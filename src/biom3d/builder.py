"""
Method builder.

The main purpose of this class is to easily reload a training and do prediction.
"""


import sys # for printing into file (Logger)
from datetime import datetime, timezone

import os 
import shutil

import torch
from torch.utils.data import DataLoader
import numpy as np

from biom3d import register
from biom3d import callbacks as clbk
from biom3d import utils
from biom3d import __version__

from torch.nn import Module
from typing import Any, Callable, Iterable, Optional, TextIO
#---------------------------------------------------------------------------
# utils to read config's functions in the function register

def read_config(config_fct:str, register_cat:utils.AttrDict, **kwargs:dict[str,Any])->Any:
    """
    Read the config function in the register category and run the corresponding function.
     
    Some keyword argument are merged: 
    1. the register kwargs. 
    2. the config file kwargs. 
    3. this function kwargs.

    Parameters
    ----------
    config_fct : str
        Name of the function listed in the register.
    register_cat : AttrDict
        Dictionary defining one category in the register.
    **kwargs : dict from str to any, optional
        Additional keyword arguments of the function defined by the config_fct
        
    Returns
    -------
    any
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



class Logger(object): # Should be more versatile and in utils
    """
    Class to redirect prints to file and to terminal simultaneously.

    Source: https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting

    :ivar TextIO terminal: The default terminal (stdout).
    :ivar TextIO log: An open file (in append mode).
    """

    terminal:TextIO
    log:TextIO

    def __init__(self, terminal:TextIO, filename:str):
        """
        Initialize a logger with given parameters.

        Parameters
        ----------
        terminal: TextIO 
            The terminal to print to.
        filemame: str 
            Path to a log file. Will be opened in append mode.
        """
        self.terminal = terminal
        self.log = open(filename, "a")
   
    def write(self, message:str)->None:
        """
        Write given message in both terminal and log file.

        Parameters
        ----------
        message: str
            The message to write
        
        Returns
        -------
        None
        """
        self.terminal.write(message)
        self.log.write(message)  

    #TODO implementation needed
    def flush(self):
        """
        Flush both terminal and log file.

        Do nothing for the moment.
        """
        pass   

def get_params_groups(model:Module)->list[dict[str,Any]]:
    """
    Split model parameters into two groups: those to be regularized (e.g., with weight decay), and those not to be regularized (such as biases and normalization layers).

    Parameters
    ----------
    model : torch.nn.Module
        The model containing parameters to be grouped.

    Returns
    -------
    A list of two dict from str to any:
        - The first with key 'params' for regularized parameters.
        - The second with key 'params' and 'weight_decay': 0. for non-regularized parameters.
    
    Notes
    -----
    This is typically used when setting up an optimizer with different weight decay for
    different parameter types, such as excluding biases and normalization layers from regularization.
    """
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
    Layer-wise Adaptive Rate Scaling (LARS) optimizer.

    This optimizer is designed for large-batch training and improves convergence
    by adapting the learning rate for each layer based on the ratio of parameter
    norm to gradient norm. It is commonly used in self-supervised learning methods
    such as Barlow Twins or SimCLR.

    Based on implementation from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py

    Notes
    -----
    - LARS adapts the learning rate per layer based on the ratio:
      `eta * ||param|| / ||grad||`.
    - Biases and batch norm parameters are typically excluded from weight decay
      and LARS adaptation, based on parameter shape (1D).
    - This optimizer requires that gradients are already computed (i.e., after `loss.backward()`).

    """

    def __init__(self, params:Iterable|dict, 
                 lr:float=0.0, 
                 weight_decay:float=0.0, 
                 momentum:float=0.9, 
                 eta:float=0.001,
                 weight_decay_filter:Optional[Callable]=None, 
                 lars_adaptation_filter:Optional[Callable]=None):
        """
        Initialize the LARS.

        Parameters
        ----------
        params : iterable
            Iterable of parameters or dicts defining parameter groups.
        lr : float, default=0
            Learning rate.
        weight_decay : float, default=0
            Weight decay (L2 regularization).
        momentum : float, default=0.9
            Momentum factor.
        eta : float, default=0.001
            LARS coefficient to scale the learning rate based on layer norms.
        weight_decay_filter : callable, optional
            A function that takes a parameter and returns True if weight decay
            should be applied to it. If None, applies to all except 1D parameters.
        lars_adaptation_filter : callable, optional
            A function that takes a parameter and returns True if LARS adaptation
            should be applied. If None, applies to all except 1D parameters.
        """
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self)->None:
        """
        Perform a single optimization step.

        For each parameter with a gradient:
        - Applies optional weight decay (L2 regularization), usually skipped for biases and norm parameters.
        - Computes the LARS scaling factor (based on the ratio of parameter norm to gradient norm)
        and rescales the gradient accordingly.
        - Applies momentum using a running buffer.
        - Updates the parameter with the computed update.

        Returns
        -------
        None

        Notes
        -----
        - The step assumes gradients have already been computed (i.e., after `loss.backward()`).
        - Parameters of dimension 1 (e.g., biases, BatchNorm weights) are usually excluded from
          weight decay and LARS scaling by design.
        - No filtering logic is applied unless you manually pre-filter the parameter groups.

         
        """
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

    Please note that in this current version the training and inference can only be done with CUDA or Apple Silicon.

    Multi-GPUs training is also supported with DataParallel only (Distributed Data Parallel is not).

    Training is currently done with the SGD optimizer. If you would like to change the optimizer, you can edit `self.build_training` method.

    :ivar utils.AttrDict | list[utils.AttrDict] config: The configuration object(s), loaded from file or passed directly. A list if using multi-model mode.
    :ivar str config_path: Path to the configuration file. Only defined if provided at init or loaded from disk.
    :ivar torch.nn.Module | list[torch.nn.Module] model: The model instance(s). Always defined when training, fine-tuning, or loading for inference.

    :ivar torch.utils.data.Dataset train_dataset: Training dataset, see `biom3d.datasets`. Defined only in training mode.
    :ivar torch.utils.data.DataLoader train_dataloader: DataLoader for training, see `biom3d.datasets`. Defined only in training mode.
    :ivar torch.utils.data.Dataset val_dataset: Validation dataset, see `biom3d.datasets`. Defined only in training mode.
    :ivar torch.utils.data.DataLoader val_dataloader: DataLoader for validation, see `biom3d.datasets`. Defined only in training mode.

    :ivar biom3d.Metric loss_fn: Loss function for training, see `biom3d.metrics`. Defined only in training mode.
    :ivar biom3d.Metric val_loss_fn: Loss function for validation, see `biom3d.metrics`. Defined only if `VAL_LOSS` is in config.
    :ivar list[biom3d.Metric] train_metrics: Training metrics, see `biom3d.metrics`. Defined only if `TRAIN_METRICS` is in config.
    :ivar list[biom3d.Metric] val_metrics: Validation metrics, see `biom3d.metrics`. Defined only if `VAL_METRICS` is in config.

    :ivar torch.optim.Optimizer optim: Optimizer instance, typically SGD. Defined only in training mode.

    :ivar biom3d.callbacks.LRSchedulerMultiStep | biom3d.callbacks.LRSchedulerCosine | biom3d.callbacks.LRSchedulerPoly clbk_scheduler: Main learning rate scheduler. Defined only if scheduler is configured.
    :ivar biom3d.callbacks.ForceFGScheduler clbk_fg_scheduler: Foreground rate scheduler. Defined only if `USE_FG_CLBK` is True.
    :ivar biom3d.callbacks.WeightDecayScheduler clbk_wd_scheduler: Weight decay scheduler. Defined only if `USE_WD_CLBK` is True.
    :ivar biom3d.callbacks.MomentumScheduler clbk_momentum_scheduler: Momentum scheduler. Defined only if `USE_MOMENTUM_CLBK` is True.
    :ivar biom3d.callbacks.OverlapScheduler clbk_overlap_scheduler: Scheduler for overlap rate. Defined only if `USE_OVERLAP_CLBK` is True.
    :ivar biom3d.callbacks.GlobalScaleScheduler clbk_global_scale_scheduler: Scheduler for global scale. Defined only if `USE_GLOBAL_SCALE_CLBK` is True.
    :ivar biom3d.callbacks.DatasetSizeScheduler clbk_dataset_size_scheduler: Scheduler for dataset size. Defined only if `USE_DATASET_SIZE_CLBK` is True.

    :ivar biom3d.callbacks.ModelSaver clbk_modelsaver: Callback to save the model at each epoch. Always defined in training mode.
    :ivar biom3d.callbacks.LogSaver clbk_logsaver: Callback to save logs. Always defined in training mode.
    :ivar biom3d.callbacks.ImageSaver clbk_imagesaver: Callback to save prediction images. Defined only if `USE_IMAGE_CLBK` is True and validation loader is available.
    :ivar biom3d.callbacks.MetricsUpdater clbk_metricupdater: Callback to update metrics. Always defined in training mode.
    :ivar biom3d.callbacks.TensorboardSaver clbk_tensorboardsaver: Callback to save metrics to Tensorboard. Always defined in training mode.
    :ivar biom3d.callbacks.LogPrinter clbk_logprinter: Callback to print logs to console. Always defined in training mode.
    :ivar biom3d.callbacks.Callbacks callbacks: Aggregated container for all callbacks. Always defined in training mode.

    :ivar str base_dir: Root directory created by `build_train`. Defined only in training mode.
    :ivar str image_dir: Directory for saved prediction images. Subfolder of `base_dir`.
    :ivar str log_dir: Directory for logs and configuration files. Subfolder of `base_dir`.
    :ivar str model_dir: Directory for saved model weights. Subfolder of `base_dir`.
    :ivar str model_path: Full path to the model file used in training. Defined only in training mode.
    :ivar int initial_epoch: Starting epoch number (e.g. for resuming training). Defined only in training mode.


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
    >>> builder.run_prediction_folder(path_in="input/folder", path_out="output/folder")
    """

    def __init__(self, 
        config:str|dict|utils.AttrDict=None,         # inherit from Config class, stores the global variables
        path:str|list[str]=None,      # path to a training folder
        training:bool=True,  # use training mode or testing?
        ):    
        """
        Initialize the builder.

        If both `config` and `path` are defined then Builder considers that fine-tuning is intended.

        If `path` is a list of path, then multi-model prediction will be used, training should be off/False.

        Raises
        ------
        AssertionError
            If config is not in the correct type.
        AssertionError
            If config is None and path is None

        Parameters
        ----------
        config : str, dict or biom3d.utils.AttrDict
            Path to a Python configuration file (in either .py or .yaml format) or dictionary of a configuration file. Please refer to biom3d.config_default.py to see the default configuration file format.
        path : str, list of str
            Path to a builder folder which contains the model folder, the model configuration and the training logs.
            If path is a list of strings, then it is considered that it is intended to run multi-model predictions. Training is not compatible with this mode.
        training : bool, default=True
            Whether to load the model in training or testing mode.
        """            
        # for training or fine-tuning:
        # load the config file and change some parameters if multi-gpus training
        if config is not None: 
            assert isinstance(config,str) or isinstance(config,dict) or isinstance(config,utils.AttrDict), "[Error] Config has the wrong type {}".format(type(config))
            if isinstance(config,str):
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
            elif torch.mps.device_count() > 1: 
                print("Let's use", torch.mps.device_count(), "GPUs!")

                # Loop through all key-value pairs of a nested dictionary and change the batch_size 
                self.config = utils.nested_dict_change_value(self.config, 'batch_size', torch.mps.device_count()*self.config.BATCH_SIZE)
                self.config.BATCH_SIZE *= torch.mps.device_count()
                self.config.NB_EPOCHS = self.config.NB_EPOCHS//torch.mps.device_count()

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
            assert config is not None, "[Error] config file not defined." # Maiby do an assert config is not None and path is not None at the beginning
            # print(self.config)
            self.build_train()
        
        # if cuda is not available then deactivate USE_FP16, mps support of FP16 is still unstable
        if not torch.cuda.is_available():
            self.config.USE_FP16 = False
        

    def build_dataset(self)->None:
        """
        Build training and validation datasets and dataloaders.

        This method reads dataset and dataloader configurations from `self.config` and 
        instantiates the corresponding objects using `read_config()` or default `DataLoader`.

        If only kwargs are provided (e.g., `*_DATALOADER_KWARGS`), the dataloaders are
        constructed using standard `torch.utils.data.DataLoader`.

        Returns
        -------
        None
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


    def build_model(self, training:bool=True)->None:
        """
        Build the model, losses, metrics, and optimizer.

        Handles both standard and self-supervised learning models.
        - If using a self-supervised architecture (e.g., BYOL/SimCLR), it initializes a student-teacher pair.
        - In training mode, it also loads loss functions, evaluation metrics, and configures the optimizer.

        Parameters
        ----------
        training : bool, default=True
            Whether to initialize the model in training mode (builds losses, metrics, optimizer), or prediction..

        Returns
        -------
        None
        """        
        # self-supervised models are special cases
        # 2 models must be defined: student and teacher model
        if 'Self' in self.config.MODEL.fct:
            student = read_config(self.config.MODEL, register.models)
            teacher = read_config(self.config.MODEL, register.models)

            if torch.cuda.is_available():
                student.cuda()
                teacher.cuda()
            elif torch.backends.mps.is_available():
                student.to('mps')
                teacher.to('mps')
            
            # teacher and student start with the same weights
            teacher.load_state_dict(student.state_dict())

            # there is no backpropagation through the teacher, so no need for gradients
            for p in teacher.parameters():
                p.requires_grad = False

            # student model is DataParallel
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                student = torch.nn.DataParallel(student)
            elif torch.mps.device_count() > 1:
                print("Let's use", torch.mps.device_count(), "GPUs!")
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
            elif torch.backends.mps.is_available():
                self.model.to('mps')
                if torch.mps.device_count() > 1:
                    print("Let's use", torch.mps.device_count(), "GPUs!")
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
            elif torch.backends.mps.is_available():
                self.loss_fn.to('mps')
            self.loss_fn.train()

            if 'VAL_LOSS' in self.config.keys():
                self.val_loss_fn = read_config(self.config.VAL_LOSS, register.metrics)
                if torch.cuda.is_available():
                    self.val_loss_fn.cuda()
                if torch.backends.mps.is_available():
                    self.val_loss_fn.to('mps')
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
                    elif torch.backends.mps.is_available():
                        m.to('mps')
                    m.eval()
            else: self.train_metrics = []

            if 'VAL_METRICS' in self.config.keys():
                self.val_metrics = [read_config(v, register.metrics) for v in self.config.VAL_METRICS.values()]
                for m in self.val_metrics: 
                    if torch.cuda.is_available():
                        m.cuda()
                    elif torch.backends.mps.is_available():
                        m.to('mps')
                    m.eval()
            else: self.val_metrics = []

            # optimizer
            if 'LR_START' in self.config.keys() and self.config.LR_START is not None: lr = self.config.LR_START
            else: lr = 1e-2
            print("Training will start with a learning rate of", lr)
            
            # for self-supervised learning:
            if 'Self' in self.config.MODEL.fct: params = get_params_groups(self.model[0])
            else: params = [{'params': self.model.parameters()}, {'params': self.loss_fn.parameters()}]

            weight_decay = 0 if ('WEIGHT_DECAY' not in self.config.keys()) else 3e-5

            self.optim = torch.optim.SGD(
                # [{'params': self.model.parameters()}, {'params': self.loss_fn.parameters()}], 
                params, 
                lr=lr, momentum=0.99, nesterov=True, weight_decay=weight_decay)


    def build_callbacks(self)->None:
        """
        Build the training callbacks.

        This includes model saving, learning rate schedulers, logging, metrics monitoring,
        image saving, tensorboard logging, and various training schedulers (momentum, weight decay, etc.).

        The exact callbacks instantiated depend on keys present in `self.config`.

        Returns
        -------
        None
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
        
        self.callbacks = clbk.Callbacks(clbk_dict)


    def build_train(self)->None:
        """
        Build and initialize all components required for training.

        This includes:
        - setting random seeds for reproducibility,
        - creating directories for logs, images, and models,
        - create a Logger,
        - copying config and CSV files to log directory,
        - initializing datasets, model, and callbacks.

        Returns
        -------
        None
        """
        # make it deterministic 
        np.random.seed(12345)
        torch.manual_seed(12345)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(12345)
        elif torch.backends.mps.is_available():
            # No all in API
            torch.mps.manual_seed(12345)
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
        
        # save the config file
        if self.config_path is not None:
            basename = os.path.basename(self.config_path)
            shutil.copy(self.config_path, os.path.join(self.log_dir, basename))
        # copy csv file
        if self.config.CSV_DIR is not None:
            basename = os.path.basename(self.config.CSV_DIR)
            shutil.copy(self.config.CSV_DIR, os.path.join(self.log_dir, basename))

        self.config["TRAINED_ON_VERSION"]=__version__
        self.config["TRAINED_AT_DATE_AND_HOUR"]=datetime.now(timezone.utc)

        utils.save_yaml_config(os.path.join(self.log_dir, 'config.yaml'), self.config) # will eventually replace the yaml file

        self.model_path = os.path.join(self.model_dir, self.config.DESC)

        # first epoch
        self.initial_epoch = 0

        # Build the method
        self.build_dataset()
        self.build_model()
        self.build_callbacks()

    def run_training(self):
        """Run the training and validation routines."""
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            print("[Warning] GPU not available! The training might be extremely slow. We strongly advise to use a CUDA or Apple Silicon machine to train a model. Predictions can be done using a CPU only machine.")

        if torch.cuda.is_available() and 'USE_FP16' in self.config.keys() and self.config.USE_FP16:
            scaler = torch.amp.GradScaler('cuda')
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

    def run_prediction_single(self, 
                              handler:Optional[utils.DataHandler]=None, 
                              img:Optional[np.ndarray]=None, 
                              img_meta:Optional[dict[str,Any]]=None, 
                              return_logit:bool=True,
                              skip_preprocessing:bool=False,
                              )->np.ndarray:
        """
        Compute a prediction for a single image using the model(s) and predictor(s) specified in the config.

        Two input modes are supported:
        - Provide a `handler`, which will load the first image in its dataset.
        - Provide `img` (a NumPy array) and `img_meta` (metadata dictionary) manually.

        It is advised to provide both img and img_meta.

        Parameters
        ----------
        handler : DataHandler, optional
            A data handler that will be used to load the specific image. It will load the first image in its collection.
        img : numpy.ndarray, optional
            The image array to predict on. Required if `handler` is not provided.
        img_meta : dict, optional
            Metadata associated with the image. Required if `handler` is not provided.
        return_logit : bool, default=True
            Whether to return the model's raw output before the activation function.
        is_2d : bool, default=False
            Whether the image is 2D. If True, dimensions will be added to make it compatible with 3D models.
        skip_preprocessing : bool, default=False
            If True, skips preprocessing of the input image. It may be cause of crash if you do not provide a preprocessed like image.

        Raises
        ------
        AssertionError
            If handler, img and img_meta are None.
        AssertionError
            If preprocessor or postprocessor are different between models in multi model predictions.

        Returns
        -------
        numpy.ndarray
            The post-processed prediction output.
        """
        # load image
        if handler is not None:
            img, img_meta = handler.load(handler.images[0])
        else:
            assert img is not None and img_meta is not None, '[Error] If the handler not provided, provide the image array and its metadata'
        
        print("Input shape:", img.shape)
        num_class = self.config.NUM_CLASSES

        if isinstance(self.config,list): # multi-model mode!
            if not skip_preprocessing:
                # check if the preprocessing are all equal, then only use one preprocessing
                # TODO: make it more flexible?
                assert np.all([config.PREPROCESSOR==self.config[0].PREPROCESSOR for config in self.config[1:]]), "[Error] For multi-model prediction, the current version of biom3d imposes that all preprocessor are identical. {}".format([config.PREPROCESSOR==self.config[0].PREPROCESSOR for config in self.config[1:]])
                
                # preprocessing
                img, img_meta = read_config(self.config[0].PREPROCESSOR, register.preprocessors, img=img, img_meta=img_meta,num_classes=num_class)

            # same for postprocessors
            for i in range(len(self.config)):
                if 'POSTPROCESSOR' not in self.config[i].keys():
                    self.config[i].POSTPROCESSOR = utils.AttrDict(fct="Seg", kwargs=utils.AttrDict())

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
            if not skip_preprocessing:
                img, img_meta = read_config(self.config.PREPROCESSOR, register.preprocessors, img=img, img_meta=img_meta,num_classes=num_class)
                
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
            if 'POSTPROCESSOR' not in self.config.keys():
                self.config.POSTPROCESSOR = utils.AttrDict(fct="Seg", kwargs=utils.AttrDict())
            
            # postprocessing
            if "return_logit" in self.config.POSTPROCESSOR.kwargs.keys():
                return_logit = self.config.POSTPROCESSOR.kwargs.return_logit
            return read_config(
                self.config.POSTPROCESSOR, 
                register.postprocessors,
                logit = out,
                return_logit = return_logit,
                **img_meta)

    # TODO: Maybe rename this function to run_prediction_collection ?
    def run_prediction_folder(self, 
                              path_in:str, 
                              path_out:str, 
                              return_logit:bool=False,
                              skip_preprocessing:bool=False
                              )->str:
        """
        Compute predictions for all images in a collection and save the results.

        Parameters
        ----------
        path_in : str
            Path to the input collection containing the images.
        path_out : str
            Path to the output collection where the predictions will be saved.
        return_logit : bool, default=False
            Whether to save the model's raw output (logit) before post-processing.
        is_2d : bool, default=False
            Whether the input images are 2D. Adds extra dimensions to mimic 3D if True.
        skip_preprocessing : bool, default=False
            If True, skips image preprocessing prior to prediction.

        Returns
        -------
        str
            The path to the collection where predictions were saved.
        """
        with utils.DataHandlerFactory.get(
            path_in,
            output=path_out,
            read_only=False,
            msk_outpath = path_out,
            # model_name = self.config[-1].DESC if isinstance(self.config,list) else self.config.DESC,
        ) as handler:
            for i,_,_ in handler:
                print("running prediction for image: ", i)
                img, img_meta = handler.load(i)
                pred = self.run_prediction_single(img=img, img_meta=img_meta, return_logit=return_logit,skip_preprocessing=skip_preprocessing)
                print("Saving image...")
                fnames_out= handler.save(i,pred,"pred")
                print("Saved images in", fnames_out)
            out = handler.msk_outpath

        return out
                
    def load_train(self, path:str, load_best:bool=False)->None: 
        """
        Load a builder from a folder. The folder should have been created by the `self.build_train` method.
       
        Can be use to restart a training. 

        Parameters
        ----------
        path : str
            Path of the log folder.
        load_best : bool, default=False
            Whether to load the best model or the final model.

        Returns
        -------
        None
        """
        # define config
        self.config = utils.load_yaml_config(os.path.join(path,"log","config.yaml"))

        # setup the different paths from the folder
        self.base_dir = path
        self.image_dir = os.path.join(path, 'image')
        self.log_dir = os.path.join(path, 'log')
        self.model_dir = os.path.join(path, 'model')

        self.model_path = os.path.join(self.model_dir, self.config.DESC)

        # redirect all prints to file and to terminal
        logger = Logger(sys.stdout, os.path.join(self.log_dir,datetime.now().strftime("%Y%m%d-%H%M%S")+'-prints.txt'))
        sys.stdout = logger

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
    
    def load_test(self, path:str, load_best:bool=True)->None: 
        """
        Load a builder from a folder in testing mode. The folder must have been created during training via `build_train`.

        Can be used to load a model for inference or evaluation on unseen data.
        Supports both single-model and multi-model mode.


        Parameters
        ----------
        path : str or list of str
            Path to the training folder (or list of paths for multi-model prediction).
        load_best : bool, default=True
            If True, loads the best model checkpoint (based on validation loss).
            If False, loads the final checkpoint.

        Raises
        ------
        RuntimeError
            If a model hasn't a load method during multi model predicition. May be extended to single model prediction in the future.
        
        Returns
        -------
        None
        """
        # if the path is a list of path then multi-model mode
        if isinstance(path,list):
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
                if torch.cuda.is_available(): device = 'cuda'
                elif torch.backends.mps.is_available(): device = 'mps'
                else: device = 'cpu'
                ckpt = torch.load(model_path, map_location=torch.device(device))
                print("Loading model from", model_path)

                # remove `module.` prefix
                state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}  
                # remove `backbone.` prefix induced by multicrop wrapper
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                print(self.model.load_state_dict(state_dict, strict=False))

#---------------------------------------------------------------------------