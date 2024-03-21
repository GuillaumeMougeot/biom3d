#---------------------------------------------------------------------------
# Method builder
# The main purpose of this class is to easily reload a training 
#---------------------------------------------------------------------------

import sys # for printing into file (Logger)
from datetime import datetime

import os 
import shutil

import torch
from torch.utils.data import DataLoader
# from monai.data import ThreadDataLoader
# import torch.distributed as dist
import numpy as np

from biom3d import register
from biom3d import callbacks as clbk
from biom3d import utils

#---------------------------------------------------------------------------
# utils to read config's functions in the function register

def read_config(config_fct, register_cat, **kwargs):
    """Read the config function in the register category and run the corresponding function with the keyword arguments which are merged from 1. the register kwargs, 2. the config file kwargs and 3. this function kwargs.

    Parameters
    ----------
    config_fct : str
        Name of the function listed in the register.
    register_cat : Dict
        Dictionary defining one category in the register.
    **kwargs : dict, optional
        Additional keyword arguments of the function defined by the config_fct
        
    Returns
    -------
    The eventual outputs of the function.
    """
    # read the corresponding name in the register
    register_fct_ = register_cat[config_fct.fct]

    # get the register function name
    register_fct = register_fct_.fct

    # get the register function kwargs and merge them with config function kwargs 
    register_kwargs = {**register_fct_.kwargs, **config_fct.kwargs, **kwargs} 

    # run the function with its kwargs
    return register_fct(**register_kwargs) 

#---------------------------------------------------------------------------
# class to redirect prints to file and to terminal simultaneously
# source: https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting

class Logger(object):
    def __init__(self, terminal, filename):
        self.terminal = terminal
        self.log = open(filename, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass   

# class ErrorLogger(object):
#     def __init__(self, filename):
#         self.terminal = sys.stderr
#         self.log = open(filename, "a")
   
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)  

#     def flush(self):
#         pass  

#---------------------------------------------------------------------------
# for distributed data parallel

# TODO: use DDP...
# def setup_ddp(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)

# def cleanup_ddp():
#     dist.destroy_process_group()

#---------------------------------------------------------------------------
# self-supervised specific
# optimizer and parameters getter

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

#---------------------------------------------------------------------------
# Global routine

class Builder:
    """The Builder is the core of biom3d. 

    The Builder reads a configuration file and builds the components required for training and prediction.

    Please note that in this current version the training and inference can only be done with CUDA.

    Multi-GPUs training is also supported with DataParallel only (Distributed Data Parallel is not).

    Training is currently done with the SGD optimizer. If you would like to change the optimizer, you can edit `self.build_training` method.

    If both `config` and `path` are defined then Builder considers that fine-tuning is intended.

    If `path` is a list of path, then multi-model prediction will be used, training should be off/False.

    Parameters
    ----------
    config : str, dict or biom3d.utils.Dict
        Path to a Python configuration file (in either .py or .yaml format) or dictionary of a configuration file. Please refer to biom3d.config_default.py to see the default configuration file format.
    path : str, list of str
        Path to a builder folder which contains the model folder, the model configuration and the training logs.
        If path is a list of strings, then it is considered that it is intended to run multi-model predictions. Training is not compatible with this mode.
    training : bool, default=True
        Whether to load the model in training or testing mode.

    Examples
    --------
    To run a training from a configuration file do:

    >>> from importlib import import_module
    >>> cfg = import_module("biom3d.configs.unet_default").CONFIG
    >>> builder = Builder(config=cfg)
    >>> builder.run_training() # start the training

    To run a prediction from a log folder, do:
    
    >>> path = "path/to/log/folder"
    >>> builder = Builder(path=path, training=False)
    >>> builder.run_prediction_folder(dir_in="input/folder", dir_out="output/folder")
    """
    def __init__(self, 
        config=None,         # inherit from Config class, stores the global variables
        path=None,      # path to a training folder
        training=True,  # use training mode or testing?
        ):                
        # for training or fine-tuning:
        # load the config file and change some parameters if multi-gpus training
        if config is not None: 
            assert type(config)==str or type(config)==dict or type(config)==utils.Dict, "[Error] Config has the wrong type {}".format(type(config))
            if type(config)==str:
                self.config_path = config
                self.config = utils.adaptive_load_config(config)
            else:
                self.config_path = None
                self.config = config

            # if there are more than 1 GPU we augment the batch size and reduce the number of epochs
            if torch.cuda.device_count() > 1: 
                print("Let's use", torch.cuda.device_count(), "GPUs!")

                # Loop through all key-value pairs of a nested dictionary and change the batch_size 
                self.config = utils.nested_dict_change_value(self.config, 'batch_size', torch.cuda.device_count()*self.config.BATCH_SIZE)
                self.config.BATCH_SIZE *= torch.cuda.device_count()
                self.config.NB_EPOCHS = self.config.NB_EPOCHS//torch.cuda.device_count()

        # fine-tuning  
        if path is not None and config is not None:
            print("Fine-tuning mode! The path to a builder folder and a configuration file have been input.")
            
            # build the training folder
            self.build_train()

            # load the model weights
            model_dir = os.path.join(path, 'model')
            model_name = utils.load_yaml_config(os.path.join(path,"log","config.yaml")).DESC+'.pth'
            ckpt_path = os.path.join(model_dir, model_name)
            ckpt = torch.load(ckpt_path)
            print("Loading model from", ckpt_path)
            print(self.model.load_state_dict(ckpt['model'], strict=False))
        
        # training restart or prediction
        elif path is not None:
            # print(self.config)
            if training:
                self.load_train(path)
            else:
                self.load_test(path)
        
        # standard training
        else:
            assert config is not None, "[Error] config file not defined."
            # print(self.config)
            self.build_train()
        
        # if cuda is not available then deactivate USE_FP16
        if not torch.cuda.is_available():
            self.config.USE_FP16 = False

    def build_dataset(self):
        """Build the dataset.
        """
        if 'TRAIN_DATASET' in self.config.keys():
            self.train_dataset = read_config(self.config.TRAIN_DATASET, register.datasets)
        if 'TRAIN_DATALOADER' in self.config.keys():
            self.train_dataloader = read_config(self.config.TRAIN_DATALOADER, register.datasets)
        else:
            self.train_dataloader = DataLoader(self.train_dataset,**self.config.TRAIN_DATALOADER_KWARGS)

        if 'VAL_DATASET' in self.config.keys():
            self.val_dataset = read_config(self.config.VAL_DATASET, register.datasets)
        if 'VAL_DATALOADER' in self.config.keys():
            self.val_dataloader = read_config(self.config.VAL_DATALOADER, register.datasets)
        elif 'VAL_DATALOADER_KWARGS' in self.config.keys():
            self.val_dataloader = DataLoader(self.val_dataset,**self.config.VAL_DATALOADER_KWARGS)


    def build_model(self, training=True):
        """Build the model, the losses and the metrics.
        """
        
        # self-supervised models are special cases
        # 2 models must be defined: student and teacher model
        if 'Self' in self.config.MODEL.fct:
            student = read_config(self.config.MODEL, register.models)
            teacher = read_config(self.config.MODEL, register.models)

            # if torch.cuda.is_available():
            student.cuda()
            teacher.cuda()
            
            # teacher and student start with the same weights
            teacher.load_state_dict(student.state_dict())

            # there is no backpropagation through the teacher, so no need for gradients
            for p in teacher.parameters():
                p.requires_grad = False

            # student model is DataParallel
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                student = torch.nn.DataParallel(student)
            
            # verbose
            print("Student model:")
            print(student)

            # gather models
            self.model = [student, teacher]
        else: 
            self.model = read_config(self.config.MODEL, register.models)

            if torch.cuda.is_available():
                self.model.cuda()
                if torch.cuda.device_count() > 1:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    self.model = torch.nn.DataParallel(self.model)

            # TODO: use DDP...
            # if rank is not None: 
            #     self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])
            # else:
            #     self.model.cuda()
            # if training:
            #     self.model.train()
            # else:
            #     self.model.eval()

            print(self.model) # TODO: model verbose

        # losses
        if training:
            self.loss_fn = read_config(self.config.TRAIN_LOSS, register.metrics)
            if torch.cuda.is_available():
                self.loss_fn.cuda()
            self.loss_fn.train()

            if 'VAL_LOSS' in self.config.keys():
                self.val_loss_fn = read_config(self.config.VAL_LOSS, register.metrics)
                if torch.cuda.is_available():
                    self.val_loss_fn.cuda()
                self.val_loss_fn.eval()
            else: 
                self.val_loss_fn = None

            # metrics
            if 'TRAIN_METRICS' in self.config.keys():
                self.train_metrics = [read_config(v, register.metrics) for v in self.config.TRAIN_METRICS.values()]
                # print(self.train_metrics)
                for m in self.train_metrics: 
                    if torch.cuda.is_available():
                        m.cuda()
                    m.eval()
            else: self.train_metrics = []

            if 'VAL_METRICS' in self.config.keys():
                self.val_metrics = [read_config(v, register.metrics) for v in self.config.VAL_METRICS.values()]
                for m in self.val_metrics: 
                    if torch.cuda.is_available():
                        m.cuda()
                    m.eval()
            else: self.val_metrics = []

            # optimizer
            if 'LR_START' in self.config.keys() and self.config.LR_START is not None: lr = self.config.LR_START
            else: lr = 1e-2
            print("Training will start with a learning rate of", lr)
            
            # for self-supervised learning:
            if 'Self' in self.config.MODEL.fct: params = get_params_groups(self.model[0])
            else: params = [{'params': self.model.parameters()}, {'params': self.loss_fn.parameters()}]

            weight_decay = 0 if not ('WEIGHT_DECAY' in self.config.keys()) else 3e-5

            self.optim = torch.optim.SGD(
                # [{'params': self.model.parameters()}, {'params': self.loss_fn.parameters()}], 
                params, 
                lr=lr, momentum=0.99, nesterov=True, weight_decay=weight_decay)
            # self.optim = LARS(params)


    def build_callbacks(self):
        """Build the callbacks for the training process. Callback are used to monitor the training process and to update the schedulers.

        As the callbacks are often dependant on the Builder arguments, they are defined directly here.
        """

        clbk_dict = {}

        # callbacks: saver (model, summary, images, train_state), logger (prints), scheduler, model_freezer, telegram_sender
        if "LR_MILESTONES" in self.config.keys():
            self.clbk_scheduler = clbk.LRSchedulerMultiStep(
                optimizer=self.optim,
                milestones=self.config.LR_MILESTONES)
            clbk_dict["lr_scheduler"] = self.clbk_scheduler

        elif "LR_T_MAX" in self.config.keys():
            self.clbk_scheduler = clbk.LRSchedulerCosine(
                optimizer=self.optim,
                T_max=self.config.LR_T_MAX)
            clbk_dict["lr_scheduler"] = self.clbk_scheduler

        elif "LR_START" in self.config.keys():
            self.clbk_scheduler = clbk.LRSchedulerPoly(
                optimizer=self.optim,
                initial_lr=self.config.LR_START,
                max_epochs=self.config.NB_EPOCHS,
                )
            clbk_dict["lr_scheduler"] = self.clbk_scheduler

        if 'USE_FG_CLBK' in self.config.keys() and self.config.USE_FG_CLBK:
            self.clbk_fg_scheduler = clbk.ForceFGScheduler(
                dataloader=self.train_dataloader,
                initial_rate=1,
                min_rate=0.3,
                max_epochs=self.config.NB_EPOCHS,
            )
            clbk_dict["fg_scheduler"] = self.clbk_fg_scheduler
        
        if 'USE_WD_CLBK' in self.config.keys() and self.config.USE_WD_CLBK:
            self.clbk_wd_scheduler = clbk.WeightDecayScheduler(
                optimizer=self.optim,
                initial_wd=self.config.INITIAL_WD,
                final_wd=self.config.FINAL_WD,
                nb_epochs=self.config.NB_EPOCHS,
                use_poly=True,
            )
            clbk_dict["wd_scheduler"] = self.clbk_wd_scheduler
        
        if 'USE_MOMENTUM_CLBK' in self.config.keys() and self.config.USE_MOMENTUM_CLBK:
            self.clbk_momentum_scheduler = clbk.MomentumScheduler(
                initial_momentum=0.9,
                final_momentum=self.config.INITIAL_MOMENTUM,
                nb_epochs=50,
                mode='linear',
                
                # initial_momentum=self.config.INITIAL_MOMENTUM,
                # final_momentum=1.0,
                # nb_epochs=self.config.NB_EPOCHS,
                # mode='exp',
                # exponent=6.0,
            )
            clbk_dict["momentum_scheduler"] = self.clbk_momentum_scheduler
        
        if 'USE_OVERLAP_CLBK' in self.config.keys() and self.config.USE_OVERLAP_CLBK:
            self.clbk_overlap_scheduler = clbk.OverlapScheduler(
                dataloader=self.train_dataloader,
                initial_rate=1.0,
                min_rate=-1.0,
                max_epochs=self.config.NB_EPOCHS,
            )
            clbk_dict["overlap_scheduler"] = self.clbk_overlap_scheduler
        
        if 'USE_GLOBAL_SCALE_CLBK' in self.config.keys() and self.config.USE_GLOBAL_SCALE_CLBK:
            self.clbk_global_scale_scheduler = clbk.GlobalScaleScheduler(
                dataloader=self.train_dataloader,
                initial_rate=1.0,
                min_rate=0.0,
                max_epochs=self.config.NB_EPOCHS,
            )
            clbk_dict["clbk_global_scale_scheduler"] = self.clbk_global_scale_scheduler

        if 'USE_DATASET_SIZE_CLBK' in self.config.keys() and self.config.USE_DATASET_SIZE_CLBK:
            self.clbk_dataset_size_scheduler = clbk.DatasetSizeScheduler(
                dataloader=self.train_dataloader,
                model=self.model,
                max_dataset_size=len(self.train_dataloader.dataset.train_imgs),
                min_dataset_size=5,
            )
            clbk_dict["clbk_dataset_size_scheduler"] = self.clbk_dataset_size_scheduler
            
        self.clbk_modelsaver = clbk.ModelSaver(
                model=self.model,
                optimizer=self.optim, 
                path=self.model_path, 
                every_epoch=self.config.SAVE_MODEL_EVERY_EPOCH, 
                save_best=self.config.SAVE_BEST, 
                loss=self.val_loss_fn if self.val_loss_fn else self.loss_fn,
                saved_loss=self.loss_fn)
        clbk_dict["model_saver"] = self.clbk_modelsaver

        self.clbk_logsaver = clbk.LogSaver(
                log_dir=self.log_dir,
                train_loss=self.loss_fn,
                val_loss=None if not hasattr(self, 'val_loss_fn') else self.val_loss_fn,
                train_metrics=None if not hasattr(self, 'train_metrics') else self.train_metrics,
                val_metrics=None if not hasattr(self, 'val_metrics') else self.val_metrics,
                scheduler=None if not hasattr(self, 'clbk_scheduler') else self.clbk_scheduler,)
                # save_best=self.config.SAVE_BEST,
                # every_batch=10)
        clbk_dict["log_saver"] = self.clbk_logsaver
        
        if "USE_IMAGE_CLBK" in self.config.keys() and self.config.USE_IMAGE_CLBK and hasattr(self, 'val_dataloader'):
            self.clbk_imagesaver = clbk.ImageSaver(
                    image_dir=self.image_dir,
                    model=self.model,
                    val_dataloader=self.val_dataloader,
                    every_epoch=self.config.SAVE_IMAGE_EVERY_EPOCH,
                    use_sigmoid=not self.config.USE_SOFTMAX,
                    )
            clbk_dict["image_saver"] = self.clbk_imagesaver

        self.clbk_metricupdater = clbk.MetricsUpdater(
        metrics=[self.loss_fn] + self.train_metrics if self.train_metrics else [self.loss_fn], 
        batch_size=self.config.BATCH_SIZE)
        clbk_dict["metric_updater"] = self.clbk_metricupdater

        self.clbk_tensorboardsaver = clbk.TensorboardSaver(
                log_dir=self.log_dir,
                train_loss=self.loss_fn,
                val_loss=self.val_loss_fn,
                train_metrics=self.train_metrics,
                val_metrics=self.val_metrics,
                batch_size=self.config.BATCH_SIZE,
                n_batch_per_epoch=len(self.train_dataloader))
        clbk_dict["tensorboard_saver"] = self.clbk_tensorboardsaver
        
        self.clbk_logprinter = clbk.LogPrinter(
                metrics=[self.loss_fn] + self.train_metrics if self.train_metrics else [self.loss_fn], 
                nbof_epochs=self.config.NB_EPOCHS, 
                nbof_batches=len(self.train_dataloader),
                every_batch=10)
        clbk_dict["log_printer"] = self.clbk_logprinter
        
        # [DEPRECATED]
        # self.clbk_telegram = clbk.Telegram(
        #         loss=self.loss_fn,
        #         test_loss=self.val_loss_fn)
        # clbk_list += [self.clbk_telegram]

        self.callbacks = clbk.Callbacks(clbk_dict)


    def build_train(self):
        """Successively build the dataset, the model and the callbacks.
        """
        # make it deterministic 
        np.random.seed(12345)
        torch.manual_seed(12345)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(12345)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # saver folder configuration
        folder_name = self.config.DESC
        
        # add fold number if defined
        if 'FOLD' in self.config.keys():
            folder_name += '_fold'+str(self.config.FOLD)
        self.base_dir, self.image_dir, self.log_dir, self.model_dir = utils.create_save_dirs(
            self.config.LOG_DIR, folder_name, dir_names=['image', 'log', 'model'], return_base_dir=True) 
        
        # redirect all prints to file and to terminal
        logger = Logger(sys.stdout, os.path.join(self.log_dir,datetime.now().strftime("%Y%m%d-%H%M%S")+'-prints.txt'))
        sys.stdout = logger
        # sys.stderr = logger
        
        # save the config file
        if self.config_path is not None:
            basename = os.path.basename(self.config_path)
            shutil.copy(self.config_path, os.path.join(self.log_dir, basename))
        # copy csv file
        if self.config.CSV_DIR is not None:
            basename = os.path.basename(self.config.CSV_DIR)
            shutil.copy(self.config.CSV_DIR, os.path.join(self.log_dir, basename))

        utils.save_yaml_config(os.path.join(self.log_dir, 'config.yaml'), self.config) # will eventually replace the yaml file

        self.model_path = os.path.join(self.model_dir, self.config.DESC)

        # first epoch
        self.initial_epoch = 0

        # Build the method
        self.build_dataset()
        self.build_model()
        self.build_callbacks()

    def run_training(self):
        """Run the training and validation routines.
        """
        if not torch.cuda.is_available():
            print("[Warning] CUDA is not available! The training might be extremely slow. We strongly advise to use a CUDA machine to train a model. Predictions can be done using a CPU only machine.")

        if torch.cuda.is_available() and 'USE_FP16' in self.config.keys() and self.config.USE_FP16:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        self.callbacks.on_train_begin(self.initial_epoch)
        for epoch in range(self.initial_epoch, self.config.NB_EPOCHS):
            self.callbacks.on_epoch_begin(epoch)
            read_config(
                self.config.TRAINER,
                register.trainers,
                scaler=scaler,
                dataloader=self.train_dataloader,
                model=self.model,
                loss_fn=self.loss_fn,
                optimizer=self.optim,
                callbacks=self.callbacks,
                metrics=self.train_metrics,
                epoch=epoch,
                use_deep_supervision=self.config.USE_DEEP_SUPERVISION,
                )
            if (epoch % self.config.VAL_EVERY_EPOCH == 0) or (epoch == (self.config.NB_EPOCHS-1)):
                if 'VALIDATER' in self.config.keys():
                    read_config(
                        self.config.VALIDATER,
                        register.trainers,
                        dataloader=self.val_dataloader,
                        model=self.model,
                        loss_fn=self.val_loss_fn,
                        metrics=self.val_metrics,
                        use_fp16=scaler is not None,
                        use_deep_supervision=self.config.USE_DEEP_SUPERVISION,)
            self.callbacks.on_epoch_end(epoch)
        self.callbacks.on_train_end(self.config.NB_EPOCHS)
    
    # def main_ddp(self, rank, world_size): # TODO: use DDP...
    #     setup_ddp(rank, world_size)

    #     self.build_model()

    #     self.run_training()
    #     cleanup_ddp()

    # def run_training_ddp(self):
    #     torch.multiprocessing.spawn(
    #         self.main_ddp,
    #         args=(torch.cuda.device_count(),),
    #         nprocs=torch.cuda.device_count(),
    #         join=True,
    #     )

    def run_prediction_single(self, img_path=None, img=None, img_meta=None, return_logit=True):
        """Compute a prediction for one image using the predictor defined in the configuration file.
        Two input options are available: either give the image path or the image and its associated metadata.

        Parameters
        ----------
        img_path : str
            Path to the image.
        img : numpy.ndarray
            The entire image, required if the img_path is not provided.
        img_meta : dict
            Metadata of the image, required it the img_path is not provided.
        return_logit : bool, default=True
            Whether to return the logit, i.e. the model output before the final activation. 
        
        Returns
        -------
        numpy.ndarray
            Output images.
        """
        # load image
        if img_path is not None:
            img, img_meta = utils.adaptive_imread(img_path=img_path)
        else:
            assert img is not None and img_meta is not None, '[Error] If the image path is not provided, provide the image array and its metadata'
        
        print("Input shape:", img.shape)

        if type(self.config)==list: # multi-model mode!
            # check if the preprocessing are all equal, then only use one preprocessing
            # TODO: make it more flexible?
            assert np.all([config.PREPROCESSOR==self.config[0].PREPROCESSOR for config in self.config[1:]]), "[Error] For multi-model prediction, the current version of biom3d imposes that all preprocessor are identical. {}".format([config.PREPROCESSOR==self.config[0].PREPROCESSOR for config in self.config[1:]])
            
            # preprocessing
            img, img_meta = read_config(self.config[0].PREPROCESSOR, register.preprocessors, img=img, img_meta=img_meta)

            # same for postprocessors
            for i in range(len(self.config)):
                if not 'POSTPROCESSOR' in self.config[i].keys():
                    self.config[i].POSTPROCESSOR = utils.Dict(fct="Seg", kwargs=utils.Dict())

            assert np.all([config.POSTPROCESSOR==self.config[0].POSTPROCESSOR for config in self.config[1:]]), "[Error] For multi-model prediction, the current version of biom3d imposes that all postprocessors are identical. {}".format([config.POSTPROCESSOR==self.config[0].POSTPROCESSOR for config in self.config[1:]])

            logit = None # to accumulate the logit
            for i, config in enumerate(self.config):
                # prediction
                print('Running prediction for model number', i)
                out = read_config(
                    config.PREDICTOR, 
                    register.predictors,
                    img = img,
                    model = self.model[i], # prediction for model i
                    **img_meta
                    )

                # accumulate the logit
                logit = out if logit is None else logit+out

            logit /= len(self.config)

            # final post-processing
            return read_config(
                    self.config[0].POSTPROCESSOR, 
                    register.postprocessors,
                    logit = logit, 
                    return_logit = return_logit,
                    **img_meta) # all img_meta should be equal as we use the same preprocessors
        
        else: # single model prediction
            img, img_meta = read_config(self.config.PREPROCESSOR, register.preprocessors, img=img, img_meta=img_meta)
            
            print("Preprocessed shape:", img.shape)

            # prediction
            out = read_config(
                self.config.PREDICTOR, 
                register.predictors,
                img = img,
                model = self.model,
                **img_meta)
        
            print("Model output shape:", out.shape)
            
            # retro-compatibility: use "Seg" post-processor as default 
            if not 'POSTPROCESSOR' in self.config.keys():
                self.config.POSTPROCESSOR = utils.Dict(fct="Seg", kwargs=utils.Dict())
            
            # postprocessing
            if "return_logit" in self.config.POSTPROCESSOR.kwargs.keys():
                return_logit = self.config.POSTPROCESSOR.kwargs.return_logit
            return read_config(
                self.config.POSTPROCESSOR, 
                register.postprocessors,
                logit = out,
                return_logit = return_logit,
                **img_meta)

    def run_prediction_folder(self, dir_in, dir_out, return_logit=False):
        """Compute predictions for a folder of images.

        Parameters
        ----------
        dir_in : str
            Path to the input folder of images.
        dir_out : str
            Path to the output folder where the predictions will be stored.
        return_logit : bool, default=False
            Whether to save the logit, i.e. the model output before the final activation.
        """
        fnames_in = sorted(utils.abs_listdir(dir_in))
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out, exist_ok=True)
        
        # remove extension
        # fnames_out = [f[:f.rfind('.')] for f in sorted(os.listdir(dir_in))]
        # fnames_out = [os.path.basename(f).split('.')[0] for f in sorted(os.listdir(dir_in))]

        fnames_out = sorted(os.listdir(dir_in))

        # add folder path
        fnames_out = utils.abs_path(dir_out,fnames_out)

        for i, img_path in enumerate(fnames_in):
            print("running prediction for image: ", img_path)
            img, img_meta = utils.adaptive_imread(img_path)
            pred = self.run_prediction_single(img=img, img_meta=img_meta, return_logit=return_logit)

            print("Saving images in", fnames_out[i])
            utils.adaptive_imsave(fnames_out[i], pred, img_meta)
                
    def load_train(self, 
        path, 
        load_best=False): # whether to load the best model
        """Load a builder from a folder. The folder should have been created by the `self.build_train` method.
        Can be use to restart a training. 

        Parameters
        ----------
        path : str
            Path of the log folder.
        load_best : bool, default=False
            Whether to load the best model or the final model.
        """

        # define config
        self.config = utils.load_yaml_config(os.path.join(path,"log","config.yaml"))

        # setup the different paths from the folder
        self.base_dir = path
        self.image_dir = os.path.join(path, 'image')
        self.log_dir = os.path.join(path, 'log')
        self.model_dir = os.path.join(path, 'model')

        # self.log_path = os.path.join(self.log_dir, 'log.csv')
        # self.log_best_path = os.path.join(self.log_dir, 'log_best.csv')

        self.model_path = os.path.join(self.model_dir, self.config.DESC)

        # redirect all prints to file and to terminal
        logger = Logger(sys.stdout, os.path.join(self.log_dir,datetime.now().strftime("%Y%m%d-%H%M%S")+'-prints.txt'))
        sys.stdout = logger
        # sys.stderr = logger

        # call the build method
        self.build_model()

        # load the model and the optimizer and the loss if needed
        model_name_full = self.config.DESC + '_best.pth' if load_best else self.config.DESC + '.pth'
        ckpt = torch.load(os.path.join(self.model_dir, model_name_full))
        print("Loading model from", os.path.join(self.model_dir, model_name_full))
        print(self.model.load_state_dict(ckpt['model'], strict=False))
        if 'loss' in ckpt.keys(): self.loss_fn.load_state_dict(ckpt['loss'])

        # if not 'LR_START' in self.config.keys() or self.config.LR_START is None:
        self.optim.load_state_dict(ckpt['opt'])

        if 'epoch' in list(ckpt.keys()): 
            self.initial_epoch=ckpt['epoch']+1 # definitive version 
        print('Restart training at epoch {}'.format(self.initial_epoch))

        self.build_dataset()

        # load callbacks
        self.build_callbacks()

        # load the best loss for the model saver and the log saver
        # if os.path.exists(self.log_best_path):
        #     df_best = pd.read_csv(self.log_best_path)
        #     if hasattr(self, 'clbk_logsaver'):
        #         self.clbk_logsaver.best_loss= df_best.val_loss[0]
        #         print('Loads log saver')
        #     if hasattr(self, 'clbk_modelsaver'):
        #         self.clbk_modelsaver.best_loss= df_best.val_loss[0]
        #         print('Load model saver')
    
    def load_test(self, 
        path, 
        load_best=True): # whether to load the best model
        """Load a builder from a folder. The folder should have been created by the `self.build_train` method.
        Can be used to test the model on unseen data.

        Parameters
        ----------
        path : str
            Path of the log folder.
        load_best : bool, default=True
            Whether to load the best model or the final model.
        """
        # if the path is a list of path then multi-model mode
        if type(path)==list:
            print("We found a list of path for your model loading. Let's switch to multi-model mode!")
            self.config = []
            self.model = []
            for p in path:
                # define config
                config = utils.load_yaml_config(os.path.join(p,"log","config.yaml"))
                self.config += [config]

                # setup the different paths from the folder
                model_dir = os.path.join(p, 'model')
                model_name = config.DESC + '_best.pth' if load_best else config.DESC + '.pth'
                model_path = os.path.join(model_dir, model_name)

                # create the model
                self.model += [read_config(config.MODEL, register.models)]
                
                print("Loading model weights for model:", self.model[-1])

                # load the model
                if getattr(self.model[-1], "load", None) is not None: 
                    self.model[-1].load(model_path)
                else:
                    raise RuntimeError("For multi-model loading, please define a `load` method in your nn.Module definition.")
        else:
            # define config
            self.config = utils.load_yaml_config(os.path.join(path,"log","config.yaml"))

            # setup the different paths from the folder
            self.model_dir = os.path.join(path, 'model')
            model_name = self.config.DESC + '_best.pth' if load_best else self.config.DESC + '.pth'
            model_path = os.path.join(self.model_dir, model_name)

            # create the model
            self.model = read_config(self.config.MODEL, register.models) 

            print(self.model)

            # if the model has a 'load' method we use it
            if getattr(self.model, "load", None) is not None: 
                self.model.load(model_path)
            # else try to load it here... but not a good practice, might be removed in the future 
            # we keep this option to avoid forcing the definition of a 'load' method in the model.
            else: 
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                ckpt = torch.load(model_path, map_location=torch.device(device))
                print("Loading model from", model_path)

                # remove `module.` prefix
                state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}  
                # remove `backbone.` prefix induced by multicrop wrapper
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                print(self.model.load_state_dict(state_dict, strict=False))

#---------------------------------------------------------------------------