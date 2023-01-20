#---------------------------------------------------------------------------
# Method builder
# The main purpose of this class is to easily reload a training 
#---------------------------------------------------------------------------

import os
from skimage.io import imsave
import torch
from torch.utils.data import DataLoader
# from monai.data import ThreadDataLoader
import pandas as pd
# import torch.distributed as dist

from biom3d import register
from biom3d import callbacks as clbk
from biom3d import utils

#---------------------------------------------------------------------------
# utils to read config's functions in the function register

def read_config(config_fct, register_cat, **kwargs):
    """
    read the register category at register_cat and run the corresponding function 
    named in the config file
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
    """
    the original goal of the builder was to easily save and load a model training.
    """
    def __init__(self, 
        config=None,         # inherit from Config class, stores the global variables
        path=None,      # path to a training folder
        training=True,  # use training mode or testing?
        ):                

        if path is not None:
            self.config = utils.load_config(path + "/log/config.yaml")
            # print(self.config)
            if training:
                self.load_train(path)
            else:
                self.load_test(path)
        else:
            assert config is not None, "[Error] config file not defined."
            self.config = config

            # if there are more than 1 GPU we augment the batch size and reduce the number of epochs
            if torch.cuda.device_count() > 1: 
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # Loop through all key-value pairs of a nested dictionary and change the batch_size 
                # for pairs in utils.nested_dict_pairs_iterator(self.config):
                #     if 'batch_size' in pairs:
                #         save = self.config[pairs[0]]; i=1
                #         while i < len(pairs) and pairs[i]!='batch_size':
                #             save = save[pairs[i]]; i+=1
                #         save['batch_size'] = torch.cuda.device_count()*self.config.BATCH_SIZE
                self.config = utils.nested_dict_change_value(self.config, 'batch_size', torch.cuda.device_count()*self.config.BATCH_SIZE)
                self.config.BATCH_SIZE *= torch.cuda.device_count()
                self.config.NB_EPOCHS = self.config.NB_EPOCHS//torch.cuda.device_count()

            # print(self.config)
            self.build_train()

    def build_dataset(self):
        """
        build the dataset.
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
        """
        build the model, the losses and the metrics.
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


            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = torch.nn.DataParallel(self.model)

            if torch.cuda.is_available():
                self.model.cuda()

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
            # self.loss_fn = DiceBCE(name='train_loss') # normal supervision
            self.loss_fn = read_config(self.config.TRAIN_LOSS, register.metrics)
            self.loss_fn.cuda().train()
            # self.loss_fn.train()

            if 'VAL_LOSS' in self.config.keys():
                self.val_loss_fn = read_config(self.config.VAL_LOSS, register.metrics)
                self.val_loss_fn.cuda().eval()
                # self.val_loss_fn.eval()
            else: 
                self.val_loss_fn = None

            # metrics
            if 'TRAIN_METRICS' in self.config.keys():
                self.train_metrics = [read_config(v, register.metrics) for v in self.config.TRAIN_METRICS.values()]
                # print(self.train_metrics)
                for m in self.train_metrics: m.cuda().eval()
            else: self.train_metrics = []

            if 'VAL_METRICS' in self.config.keys():
                self.val_metrics = [read_config(v, register.metrics) for v in self.config.VAL_METRICS.values()]
                for m in self.val_metrics: m.cuda().eval()
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
                # lr=lr, momentum=0.99, nesterov=True)

            # self.optim = LARS(params)


    def build_callbacks(self):
        """
        build the callbacks for the training process.
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

        # self.clbk_logsaver = clbk.LogSaver(
        #         log_dir=self.log_dir,
        #         train_loss=self.loss_fn,
        #         val_loss=self.val_loss_fn,
        #         train_metrics=self.train_metrics,
        #         val_metrics=self.val_metrics,
        #         scheduler=self.clbk_scheduler,
        #         save_best=self.config.SAVE_BEST,
        #         every_batch=10)
        
        if "USE_IMAGE_CLBK" in self.config.keys() and self.config.USE_IMAGE_CLBK and hasattr(self, 'val_dataloader'):
            self.clbk_imagesaver = clbk.ImageSaver(
                    image_dir=self.image_dir,
                    model=self.model,
                    val_dataloader=self.val_dataloader,
                    every_epoch=self.config.SAVE_IMAGE_EVERY_EPOCH,
                    use_sigmoid=not self.config.USE_SOFTMAX,
                    )
            clbk_dict["image_saver"] = self.clbk_imagesaver

        
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
        
        self.clbk_metricupdater = clbk.MetricsUpdater(
                metrics=[self.loss_fn] + self.train_metrics if self.train_metrics else [self.loss_fn], 
                batch_size=self.config.BATCH_SIZE)
        clbk_dict["metric_updater"] = self.clbk_metricupdater

        self.callbacks = clbk.Callbacks(clbk_dict)


    def build_train(self):
        """
        build the method from scratch.
        """
        # saver configs
        self.base_dir, self.image_dir, self.log_dir, self.model_dir = utils.create_save_dirs(
            self.config.LOG_DIR, self.config.DESC, dir_names=['image', 'log', 'model'], return_base_dir=True) 
        utils.save_config(os.path.join(self.log_dir, 'config.yaml'), self.config) # save the config file

        self.model_path = os.path.join(self.model_dir, self.config.DESC)

        # first epoch
        self.initial_epoch = 0

        # Build the method
        self.build_dataset()
        self.build_model()
        self.build_callbacks()

    def run_training(self):
        """
        run the training and validation routines.
        """
        if 'USE_FP16' in self.config.keys() and self.config.USE_FP16:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        self.callbacks.on_train_begin()
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
                        use_fp16=scaler is not None)
            self.callbacks.on_epoch_end(epoch)
        self.callbacks.on_train_end()
    
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

    def run_prediction_single(self, img_path, return_logit=True):
        """
        TODO: this function should be defined in another file 
        compute a prediction for one image, just for visualizing
        """

        return read_config(
            self.config.PREDICTOR, 
            register.predictors,
            img_path = img_path,
            model = self.model,
            return_logit = return_logit,
            )
    
    def run_prediction_folder(self, dir_in, dir_out, return_logit=False):
        """
        compute a prediction for one image, just for visualizing
        """
        fnames_in = sorted(utils.abs_listdir(dir_in))
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out, exist_ok=True)
        
        # remove extension
        fnames_out = [f[:f.rfind('.')] for f in sorted(os.listdir(dir_in))]
        # fnames_out = [os.path.basename(f).split('.')[0] for f in sorted(os.listdir(dir_in))]

        # add folder path
        fnames_out = utils.abs_path(dir_out,fnames_out)
        

        for i, img_path in enumerate(fnames_in):
            print("running prediction for image: ", img_path)
            pred = self.run_prediction_single(img_path=img_path, return_logit=return_logit)
            print("Saving images in", fnames_out[i]+".tif")
            imsave(fnames_out[i]+".tif", pred)
    
    def load_train(self, 
        path, 
        load_best=False): # whether to load the best model
        """
        load a builder from a folder.
        the folder should have been created by the self.run_training method.
        """

        # setup the different paths from the folder
        self.base_dir = path
        self.image_dir = os.path.join(path, 'image')
        self.log_dir = os.path.join(path, 'log')
        self.model_dir = os.path.join(path, 'model')

        # self.log_path = os.path.join(self.log_dir, 'log.csv')
        # self.log_best_path = os.path.join(self.log_dir, 'log_best.csv')

        self.model_path = os.path.join(self.model_dir, self.config.DESC)

        # call the build method
        self.build_model()

        # load the model and the optimizer and the loss if needed
        model_name_full = self.config.DESC + '_best.pth' if load_best else self.config.DESC + '.pth'
        ckpt = torch.load(os.path.join(self.model_dir, model_name_full))
        print("Loading model from", os.path.join(self.model_dir, model_name_full))
        print(self.model.load_state_dict(ckpt['model'], strict=False))
        if 'loss' in ckpt.keys(): self.loss_fn.load_state_dict(ckpt['loss'])

        if not 'LR_START' in self.config.keys() or self.config.LR_START is None:
            self.optim.load_state_dict(ckpt['opt'])

        if 'epoch' in list(ckpt.keys()): # tmp
            self.initial_epoch=ckpt['epoch'] # definitive version 
        # else: # tmp
        #     with open(self.log_path, "r") as file:
        #         last_line = file.readlines()[-1] # read the last line
        #     last_line = last_line.split(',')
        #     self.initial_epoch=int(last_line[0])
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
        """
        load a builder from a folder.
        the folder should have been created by the self.run_training method.
        """

        # setup the different paths from the folder
        self.model_dir = os.path.join(path, 'model')
        self.model_path = os.path.join(self.model_dir, self.config.DESC)

        # call the build method
        self.build_model(training=False)

        # load the model and the optimizer
        model_name_full = self.config.DESC + '_best.pth' if load_best else self.config.DESC + '.pth'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(os.path.join(self.model_dir, model_name_full), map_location=torch.device(device))
        print("Loading model from", os.path.join(self.model_dir, model_name_full))

        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}  
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        print(self.model.load_state_dict(state_dict, strict=False))

#---------------------------------------------------------------------------