"""
Callback are periodically called during training.

There are currently 3 different periods:
- training 
- epoch
- batch
"""

import torch
import os
from shutil import copyfile
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # bug fix: change matplotlib backend 
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext
from abc import abstractmethod
import numpy as np

from typing import Any, Literal, Optional
from biom3d.metrics import Metric

#----------------------------------------------------------------------------
# Base classes
class Callback(object):
    """
    Abstract base class used to build new callbacks.

    Callback are periodically called during training. 
    There are currently 3 different periods:
    - training 
    - epoch
    - batch
    Callbacks are called either before or after a period. 

    Each method starting by `on_` can be overridden. This method will be called at a certain time point during the training process.
    For instance, the `on_epoch_end` method will be called in the end of each epoch. 

    :ivar list[biom3d.Metric] metrics 
    """

    metrics:list[Metric]

    def __init__(self):
        """Initilization of attributes."""
        self.metrics = None

    def set_trainer(self, metrics:list[Metric])->None:
        """
        Associate metrics or training state to this callback.

        Parameters
        ----------
        metrics : list of biom3d.Metric
            list of used metrics or training information to be used by the callback.

        Returns
        ------- 
        None
        """
        self.metrics = metrics

    @abstractmethod
    def on_batch_begin(self, batch:int)->None:
        """
        Call before processing a batch.
        
        Parameters
        ----------
        batch: int
            The batch index.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def on_batch_end(self, batch:int)->None:
        """
        Call after processing a batch.

        Parameters
        ----------
        batch: int
            The batch index.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch:int)->None:
        """
        Call before processing an epoch.
                
        Parameters
        ----------
        epoch: int
            The epoch index.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def on_epoch_end(self, epoch:int)->None:
        """
        Call after processing an epoch.
                
        Parameters
        ----------
        epoch: int
            The epoch index.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def on_train_begin(self, epoch:Optional[int]=None)->None:
        """
        Call once at the beginning of training.
                
        Parameters
        ----------
        epoch: int
            The epoch index.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def on_train_end(self, epoch:Optional[int]=None)->None:
        """
        Call once at the end of training.
                
        Parameters
        ----------
        epoch: int
            The epoch index.

        Returns
        -------
        None
        """
        pass


class Callbacks(Callback):
    """
    Aggregates and manages multiple callbacks, dispatching events to all contained callbacks.

    This class acts as a container for multiple Callback instances. It forwards
    all callback events (`on_batch_begin`, `on_batch_end`, `on_epoch_begin`, 
    `on_epoch_end`, `on_train_begin`, `on_train_end`) to each callback in the
    collection.

    This allows users to easily manage several callbacks in a modular way, for example
    combining logging, checkpoint saving, metric computation, etc., all in one object.

    :ivar dict[str,Callback] callbacks: A dictionary of callback.

    Examples
    --------
    >>> cbs = Callbacks({
    ...     "logger": LogSaver(...),
    ...     "saver": ModelSaver(...),
    ... })
    >>> cbs.on_epoch_end(10)  # calls on_epoch_end(10) on both LogSaver and ModelSaver
    """

    def __init__(self, callbacks:dict[str,Any]):
        """
        Initialize from a dictionary.

        Parameters
        ----------
        callbacks : dict of str to biom3d.callback.Callback
            A dictionary of callbacks.
        """
        super().__init__()
        if isinstance(callbacks, Callbacks):
            callbacks = callbacks.callbacks
        self.callbacks = callbacks
        if callbacks is None:
            self.callbacks = {}
    
    def __getitem__(self, name:str)->Callback:
        """
        Get a callback by its name.

        Parameters
        ----------
        name : str
            The name/key of the callback to retrieve.

        Raises
        ------
        KeyError
            If the name does not exist in the callbacks dictionary.

        Returns
        -------
        Callback
            The callback instance associated with the given name.
        """ 
        return self.callbacks[name]

    def set_trainer(self, trainer:list[Metric])->None:
        """
        Set the trainer or metrics context for each callback.

        Parameters
        ----------
        trainer : list of biom3d.Metric
            Trainer or metrics list to pass to callbacks.

        Returns
        -------
        None
        """
        for callback in self.callbacks.values():
            callback.set_trainer(trainer)

    def on_batch_begin(self, batch:int)->None:
        """
        Call before processing a batch. Forwards the call to all callbacks.

        Parameters
        ----------
        batch : int
            Current batch index.
            
        Returns
        -------
        None
        """
        for callback in self.callbacks.values():
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch:int)->None:
        """
        Call after processing a batch. Forwards the call to all callbacks.

        Parameters
        ----------
        batch : int
            Current batch index.
            
        Returns
        -------
        None
        """
        for callback in self.callbacks.values():
            callback.on_batch_end(batch)

    def on_epoch_begin(self, epoch:int)->None:
        """
        Call before processing an epoch. Forwards the call to all callbacks.

        Parameters
        ----------
        epoch : int
            Current epoch index.
            
        Returns
        -------
        None
        """
        for callback in self.callbacks.values():
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch:int)->None:
        """
        Call after processing an epoch. Forwards the call to all callbacks.

        Parameters
        ----------
        epoch : int
            Current epoch index.
            
        Returns
        -------
        None
        """
        for callback in self.callbacks.values():
            callback.on_epoch_end(epoch)

    def on_train_begin(self, epoch:Optional[int]=None)->None:
        """
        Call once at the start of training. Forwards the call to all callbacks.

        Parameters
        ----------
        epoch : int, optional
            Starting epoch number, if any.
            
        Returns
        -------
        None
        """
        for callback in self.callbacks.values():
            callback.on_train_begin(epoch)

    def on_train_end(self, epoch:Optional[int]=None)->None:
        """
        Call once at the end of training. Forwards the call to all callbacks.

        Parameters
        ----------
        epoch : int, optional
            Final epoch number, if any.
            
        Returns
        -------
        None
        """
        for callback in self.callbacks.values():
            callback.on_train_end(epoch)

#----------------------------------------------------------------------------
# Savers            
class ModelSaver(Callback): # TODO: save best_only
    """
    Saves the model, optimizer state, epoch, and loss at the end of each epoch.
    
    Can also save the best model based on a monitored loss metric.
    
    :ivar torch.nn.Module | list[torch.nn.Module] model: Modele to store, if list it is assumed to be [student,teacher].
    :ivar torch.optim.Optimizer optimizer: torch optimizer.
    :ivar str path: Name of the file representing the model.
    :ivar str path_last: path + '.pth'
    :ivar str path_best: path + '_best.pth'
    :ivar int every_epoch: Period between save
    :ivar bool save_best: Whether to save best or not.
    :ivar float best_loss: Best loss value since beginning.
    :ivar biom3d.Metric loss: Loss function.
    :ivar biom3d.Metric saved_loss: Loss function to save alongside model.   
    """

    def __init__(self,
        model:torch.nn.Module, 
        optimizer:torch.optim.Optimizer, 
        path:str='unet', 
        every_epoch:int=2, 
        save_best:bool=True, 
        loss:Optional[Metric]=None, # loss used for saving the best model, generally the val loss
        saved_loss:Optional[Metric]=None, # loss being saved
        ):
        """
        Initilize the saver.

        Parameters
        ----------
        model : torch.nn.Module or a list of torch.nn.Module
            A torch module to store. If the model is a list then it is considered as been [student,teacher]
        optimizer : torch.optim.Optimizer
            A torch optimizer.
        path : str
            Name of the model to store. The `.pth` extension is automatically added and the best model is stored with `_best.pth` extension.
        every_epoch : int, default=2
            Period to save the model.
        save_best : bool, default=True
            Whether to save the best model.
        loss : biom3d.Metric
            Loss function.
        saved_loss : biom3d.Metric
            Loss saved alongside the model.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.path = path
        self.path_last = path + '.pth'
        self.path_best = path + '_best.pth'
        self.every_epoch = every_epoch
        self.save_best = save_best
        self.best_loss = float('inf')
        self.loss = loss
        self.saved_loss = saved_loss

    def on_train_begin(self, epoch:int)->None:
        """
        Call once at the beginning of training.

        If model checkpoint files already exist, creates backups by copying them
        to new filenames suffixed with the current epoch number to prevent overwriting.

        Parameters
        ----------
        epoch : int
            The starting epoch number.

        Returns
        -------
        None
        """
        if os.path.exists(self.path_last):
            copyfile(self.path_last, self.path + "_" + str(epoch) + ".pth")
        if os.path.exists(self.path_best):
            copyfile(self.path_best, self.path + "_" + str(epoch) + "_best.pth")

    def on_epoch_end(self, epoch:int)->None:
        """
        Call at the end of each epoch.

        Saves the model state, optimizer state, epoch number, and loss state.
        Saves every `every_epoch` epochs, and optionally saves the best model
        if the monitored loss improves.

        Parameters
        ----------
        epoch : int
            The current epoch number.

        Returns
        -------
        None
        """
        if isinstance(self.model, list):
            save_dict = {
                'epoch':   epoch,
                'student': self.model[0].state_dict(), 
                'teacher': self.model[1].state_dict(), 
                'opt':     self.optimizer.state_dict(),
                'loss':    self.saved_loss.state_dict(),
                }
        else: 
            save_dict = {
                    'epoch':    epoch,
                    'model':    self.model.state_dict(), 
                    'opt':      self.optimizer.state_dict(),
                    'loss':     self.saved_loss.state_dict(),
                    }
        if epoch % self.every_epoch == 0:
            torch.save(save_dict, self.path_last)
            print('Save model to {}'.format(self.path_last))

        # save best model if needed 
        if self.save_best and (self.loss.avg < self.best_loss) and (self.loss.avg != 0):
            save_dict['best_loss'] = self.loss.avg
            torch.save(save_dict, self.path_best)
            print('Save best model to {}'.format(self.path_best))
            self.best_loss = self.loss.avg
            
                

class LogSaver(Callback):
    """
    Callback that logs training and validation metrics to a CSV file at the end of each epoch.

    This logger writes a `log.csv` file in the specified log directory and appends:
    - Epoch number
    - Learning rate (if a scheduler is provided)
    - Training and validation loss
    - Training and validation metrics

    :ivar str path: Full path to the log CSV file.
    :ivar int crt_epoch: Current epoch (starts at 1).
    :ivar biom3d.Metric train_loss: Training loss object.
    :ivar biom3d.Metric val_loss: Validation loss object.
    :ivar list[biom3d.Metric] train_metrics: List of training metrics.
    :ivar list[biom3d.Metric] val_metrics: List of validation metrics.
    :ivar Optional[Callback] scheduler: Scheduler providing current learning rate.
    """

    def __init__(self, 
        log_dir:str,            # path to where the csv file will be store
        train_loss:Metric,
        val_loss:Optional[Metric]=None,
        train_metrics:Optional[Metric]=None,
        val_metrics:Optional[Metric]=None,
        scheduler:Optional[Callback]=None,
        ):
        """
        Initialize the log saver.

        Parameters
        ----------
        log_dir : str
            Path to the directory where the CSV log file will be created.
        train_loss : biom3d.Metric
            Metric object tracking training loss.
        val_loss : biom3d.Metric, optional
            Metric object tracking validation loss.
        train_metrics : list of biom3d.Metric, optional
            List of training metrics to log.
        val_metrics : list of biom3d.Metric, optional
            List of validation metrics to log.
        scheduler : Callback, optional
            Learning rate scheduler for logging the current learning rate.
        """
        self.path = os.path.join(log_dir,'log.csv')

        self.crt_epoch = 1

        self.train_loss = train_loss
        self.val_loss = val_loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

        self.scheduler = scheduler if hasattr(scheduler, 'get_last_lr') else None

        f = open(self.path, "a")
        if os.stat(self.path).st_size == 0: # if the file is empty
            self.write_file_head()
        f.close()

    def write_file_head(self)->None:
        """
        Write the header of the CSV file with appropriate column names.

        Returns
        -------
        None
        """
        # write the head of the log file
        f = open(self.path, "a")
        head = "epoch"
        if self.scheduler is not None:
            head += ',learning_rate'
        head += ",train_loss"
        if self.val_loss is not None:
            head += ",val_loss"
        if self.train_metrics is not None:
            for m in self.train_metrics:
                head += "," + m.name
        if self.val_metrics is not None:
            for m in self.val_metrics:
                head += "," + m.name
        f.write(head + "\n")
    
    def on_epoch_end(self, epoch:int)->None:
        """
        Write the header of the CSV file with appropriate column names.

        Returns
        -------
        None
        """
        f = open(self.path, "a")
        template =  str(epoch)

        # add the scheduler value
        if self.scheduler is not None:
            template += "," + str(self.scheduler.get_last_lr()[0])

        # add the learning rate and the training loss
        template += "," + str(self.train_loss.avg.item()) # TODO: save the avg value only
        
        # add the validation loss if needed
        if self.val_loss is not None:
            val_loss = self.val_loss.avg if isinstance(self.val_loss.avg,float) else self.val_loss.avg.item()
            template += "," + str(val_loss)
        
        # add the training metrics
        if self.train_metrics is not None:
            for m in self.train_metrics: template += "," + str(m.val.item())

        # adde the validation metrics
        if self.val_metrics is not None:
            for m in self.val_metrics: 
                val_m = m.avg if isinstance(m.avg,int) else m.avg.item() 
                template += "," + str(val_m)
        
        # write in the output file
        f.write(template + "\n")
        f.close()

class ImageSaver(Callback):
    """
    Callback that saves visual snapshots of input images, predictions, and ground truth masks at the end of selected epochs.

    Typically used with 3D medical images. It slices through the middle channel of the input volume
    and saves a 2D projection of input, predicted mask, and ground truth mask.

    :ivar str image_dir: Path where images will be saved.
    :ivar torch.nn.Module model: Model used for inference.
    :ivar torch.utils.data.Dataloader val_dataloader: Validation dataloader providing batches for inference.
    :ivar bool use_sigmoid: Whether to apply sigmoid (binary) or softmax (multiclass) on predictions.
    :ivar int every_epoch: Frequency (in epochs) to save snapshots.
    :ivar int plot_size: Number of images from the batch to visualize.
    :ivar bool use_fp16: Whether to use AMP/mixed precision during inference.
    """
    
    def __init__(self, 
        image_dir: str, 
        model: torch.nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        use_sigmoid: bool = True,
        every_epoch: int = 1,
        plot_size: int = 1,
        use_fp16: bool = True,
        ):
        """
        Initialize the ImageSaver callback.

        Parameters
        ----------
        image_dir : str
            Path to the directory where snapshots will be saved.
        model : torch.nn.Module
            Model used for making predictions.
        val_dataloader : torch.utils.data.DataLoader or iter
            Dataloader or iterable providing (input, target) batches.
        use_sigmoid : bool, default=True
            Whether to apply sigmoid (binary classification) or softmax (multiclass) to the predictions.
        every_epoch : int, default=1
            Snapshot saving frequency (in epochs).
        plot_size : int, default=1
            Number of examples from the batch to save in the snapshot.
        use_fp16 : bool, default=True
            Whether to enable AMP (automatic mixed precision) during inference.
        """
        self.image_dir = image_dir
        self.every_epoch = every_epoch
        self.model = model 
        self.use_sigmoid = use_sigmoid
        self.val_dataloader = val_dataloader
        self.plot_size = plot_size
        self.use_fp16 = use_fp16

    def on_epoch_end(self, epoch:int)->None:
        """
        Call at the end of each epoch. Saves snapshots of model predictions for visual inspection.

        Parameters
        ----------
        epoch : int
            The current epoch number.

        Returns
        -------
        None
        """
        if epoch % self.every_epoch == 0:
            self.model.eval() # TODO: we can do a prediction function because this code is dupicated
            with torch.no_grad():
                for i in range(self.plot_size):
                # make prediction
                    X, y = next(iter(self.val_dataloader))
                    if torch.cuda.is_available():
                        X, y = X.cuda(), y.cuda()
                    elif torch.backends.mps.is_available():
                        X, y = X.to('mps'), y.to('mps')

                    with torch.amp.autocast("cuda", enabled=self.use_fp16) if torch.cuda.is_available() else nullcontext():
                        pred = self.model(X)
                        if isinstance(pred,list):
                            pred = pred[-1]
                    if self.use_sigmoid:
                        pred = (torch.sigmoid(pred)>0.5).int()*255
                    else: 
                        pred = (pred.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)).int()*255
                    l = [X, y, pred.detach()]
                    for j in range(len(l)):
                        _,_,channel,_,_ = l[j].shape
                        l[j] = l[j][-1, -1, channel//2, ...].cpu().numpy().astype(float)
                    X, y, pred = l

                    # plot 
                    plt.figure(dpi=100)
                    # print the original
                    plt.subplot(self.plot_size,3,3*i+1)
                    plt.imshow(X)
                    plt.title('raw')
                    plt.axis('off')
                    # print the prediction
                    plt.subplot(self.plot_size,3,3*i+2)
                    plt.imshow(pred)
                    plt.title('pred')
                    plt.axis('off')
                    # print ground truth
                    plt.subplot(self.plot_size,3,3*i+3)
                    plt.imshow(y)
                    plt.title('ground_truth')
                    plt.axis('off')
                    del X, y, pred
                im_path = os.path.join(self.image_dir,'image_' + str(epoch) + '.png')
                print("Save image to {}".format(im_path))
                plt.savefig(im_path)
                plt.close()

class TensorboardSaver(Callback):
    """
    Callback to log losses and metrics to TensorBoard.

    This callback plots training and validation losses as well as metrics during training.
    Launch TensorBoard from the project directory with:
    `tensorboard --logdir=logs/`

    The following tags are used:
    - Loss/train
    - Loss/test
    - Metrics/{name_of_metric}

    :ivar SummaryWriter writer: TensorBoard summary writer.
    :ivar Metric train_loss: Training loss.
    :ivar Metric: Validation loss.
    :ivar list[Metric] train_metrics: List of training metric modules.
    :ivar list[Metric] val_metrics: List of validation metric modules.
    :ivar int batch_size: Size of training mini-batch.
    :ivar int n_batch_per_epoch: Number of batches per epoch.
    :ivar int crt_epoch: Current epoch (used for iteration tracking).
    """
    
    def __init__(self, 
        log_dir:str, 
        train_loss:Metric, 
        val_loss:Metric, 
        train_metrics:list[Metric], 
        val_metrics:list[Metric], 
        batch_size:int,
        n_batch_per_epoch:int):
        """
        Initialize the TensorboardSaver callback.

        Parameters
        ----------
        log_dir : str
            Path to the folder where TensorBoard logs will be saved.
        train_loss : Metric
            Training loss function.
        val_loss : Metric
            Validation loss function.
        train_metrics : list of Metric
            List of training metric objects (must expose .avg and .name).
        val_metrics : list of Metric
            List of validation metric objects (must expose .avg and .name).
        batch_size : int
            Mini-batch size, used to compute total number of images processed (x-axis).
        n_batch_per_epoch : int
            Number of batches in each epoch (used to track iteration count).
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.batch_size = batch_size
        self.n_batch_per_epoch = n_batch_per_epoch

        self.crt_epoch = 0
    
    def on_epoch_begin(self, epoch:int)->None:
        """
        Call before the start of an epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.

        Returns
        -------
        None
        """
        self.crt_epoch = epoch
    
    def on_epoch_end(self, epoch:int)->None:
        """
        Call after an epoch ends.

        Logs the training and validation losses, as well as metrics, to TensorBoard.

        Parameters
        ----------
        epoch : int
            Current epoch number.

        Returns
        -------
        None
        """
        n_iter = (epoch+1) * self.batch_size * self.n_batch_per_epoch
        self.writer.add_scalar('Loss/train', self.train_loss.avg, n_iter)
        if self.val_loss:
            self.writer.add_scalar('Loss/test', self.val_loss.avg, n_iter)
        if self.train_metrics:
            for m in self.train_metrics: self.writer.add_scalar('Metrics/'+m.name,m.avg,n_iter)
        if self.val_metrics:
            for m in self.val_metrics: self.writer.add_scalar('Metrics/'+m.name,m.avg,n_iter)

#----------------------------------------------------------------------------
# Printer
class LogPrinter(Callback):
    """
    Callback used to print training logs to the terminal during training.

    Logs the current epoch and periodically prints batch information with associated metrics and losses.

    :ivar list[Metric] metrics: List of metric or loss modules to print (must implement __str__).
    :ivar int nbof_epochs: Total number of epochs.
    :ivar int nbof_batches: Number of batches per epoch.
    :ivar int every_batch: Frequency (in batches) at which logs are printed.
    """

    def __init__(
        self,
        metrics:list[Metric],
        nbof_epochs:int,
        nbof_batches:int,
        every_batch:int=10):
        """
        Initialize the LogPrinter callback.

        Parameters
        ----------
        metrics : list of Metric
            List of metrics to display. Losses can be included as well.
        nbof_epochs : int
            Total number of training epochs.
        nbof_batches : int
            Number of batches in each epoch.
        every_batch : int, default=10
            Print logs every `every_batch` batches.
        """
        self.nbof_epochs = nbof_epochs
        self.nbof_batches = nbof_batches

        self.metrics = metrics

        self.every_batch = every_batch

    def on_epoch_begin(self, epoch:int)->None:
        """
        Call at the beginning of an epoch. Prints epoch progress.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).

        Returns
        -------
        None
        """
        print("Epoch [{:>3d}/{:>3d}]".format(epoch+1, self.nbof_epochs))

    def on_batch_end(self, batch:int)->None:
        """
        Call at the end of a batch. Prints batch index and associated metrics.

        Parameters
        ----------
        batch : int
            Index of the current batch.

        Returns
        -------
        None
        """
        if batch % self.every_batch == 0:
            template = "Batch [{:>3d}/{:>3d}]".format(batch, self.nbof_batches)
            for i in range(len(self.metrics)):
                template += ", {}".format(self.metrics[i])
            print(template)


#----------------------------------------------------------------------------
# schedulers

class LRSchedulerMultiStep(Callback):
    """
    Multi-step learning rate scheduler callback.

    Wraps `torch.optim.lr_scheduler.MultiStepLR` to schedule learning rate decay
    at specified epoch milestones. This scheduler multiplies the learning rate
    by `gamma` whenever an epoch hits a milestone.

    For more details, see:
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR

    :ivar torch.optim.lr_scheduler.MultiStepLR scheduler: Internal PyTorch scheduler.
    """

    def __init__(self, optimizer:torch.optim.Optimizer, milestones:list[int], gamma:float=0.1):
        """
        Initialize the MultiStepLR scheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate will be scheduled.
        milestones : list of int
            List of epoch indices at which to reduce the learning rate.
        gamma : float, default=0.1
            Multiplicative factor of learning rate decay.
        """
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, verbose=False)

    def get_last_lr(self)->list[float]:
        """
        Return the last computed learning rate by the scheduler.

        Returns
        -------
        list of float
            The most recent learning rates for each parameter group.
        """
        return self.scheduler.get_last_lr()
    
    def on_epoch_end(self, epoch:int)->None:
        """
        Call at the end of each epoch. Steps the learning rate scheduler.

        Parameters
        ----------
        epoch : int
            The index of the current epoch.

        Returns
        -------
        None
        """
        self.scheduler.step()
        print("Current learning rate: {}".format(self.scheduler.get_last_lr()))

class LRSchedulerCosine(Callback):
    """
    Cosine Annealing learning rate scheduler callback.

    This callback wraps `torch.optim.lr_scheduler.CosineAnnealingLR` to apply a cosine 
    decay to the learning rate over a predefined number of epochs.

    For more details, see:
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR

    :ivar torch.optim.lr_scheduler.CosineAnnealingLR scheduler: Internal PyTorch scheduler.
    """

    def __init__(self, 
                 optimizer: torch.optim.Optimizer, 
                 T_max: int):
        """
        Initialize the CosineAnnealingLR scheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate will be scheduled.
        T_max : int
            Maximum number of iterations (usually the total number of epochs).
        """
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=1e-6, verbose=False)

    def get_last_lr(self)->list[float]:
        """
        Return the last computed learning rate by the scheduler.

        Returns
        -------
        a list of float
            The most recent learning rates for each parameter group.
        """
        return self.scheduler.get_last_lr()
    
    def on_epoch_end(self, epoch:int)->None:
        """
        Call at the end of each epoch. Steps the learning rate scheduler.

        Parameters
        ----------
        epoch : int
            The index of the current epoch.

        Returns
        -------
        None
        """
        self.scheduler.step()
        print("Current learning rate: {}".format(self.scheduler.get_last_lr()))

class LRSchedulerPoly(Callback):
    r"""
    Polynomial learning rate scheduler.

    This scheduler decreases the learning rate following a polynomial decay formula,
    similar to what is used in nnU-Net:

    .. math::
        lr_{new} = lr_{initial} * (1 - \frac{epoch_{current}}{epoch_{max}})^{exponent}

    :ivar float initial_lr: Initial learning rate.
    :ivar int max_epochs: Total number of training epochs.
    :ivar float exponent: Exponent controlling the decay rate.
    :ivar torch.optim.Optimizer optimizer: Optimizer being scheduled.
    """

    def __init__(self, 
                 optimizer:torch.optim.Optimizer, 
                 initial_lr:float, 
                 max_epochs:int, 
                 exponent:float=0.9):
        """
        Initialize the polynomial scheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Training optimizer whose learning rate will be scheduled.
        initial_lr : float
            Initial learning rate.
        max_epochs : int
            Total number of epochs for training.
        exponent : float, optional, default=0.9
            Decay exponent controlling how fast the learning rate decreases.
        """
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.exponent = exponent
        self.optimizer = optimizer

    def on_epoch_begin(self, epoch:int)->None:
        """
        Update the learning rate at the beginning of each epoch.

        Parameters
        ----------
        epoch : int
            Current epoch index.

        Returns
        -------
        None
        """
        self.optimizer.param_groups[0]['lr'] = self.initial_lr * (1 - epoch / self.max_epochs)**self.exponent
        print("Current learning rate: {}".format(self.optimizer.param_groups[0]['lr']))

class ForceFGScheduler(Callback):
    r"""
    Foreground sampling rate scheduler using polynomial decay.

    This scheduler gradually reduces the rate at which foreground patches are sampled
    during training. It calls the `set_fg_rate` method of the dataset (accessed via the
    dataloader) to adjust the foreground rate at each epoch.

    Note: The dataset associated with the dataloader **must** implement the method:
    `set_fg_rate(rate: float)`.

    The decay follows this formula:

    .. math::
        fg_{rate} = (initial - min) * (1 - \frac{epoch}{max\_epochs})^{exponent} + min

    :ivar torch.utils.data.DataLoader dataloader: Dataloader whose dataset supports `set_fg_rate`.
    :ivar float initial_rate: Starting foreground sampling rate (e.g. 1.0 means only foreground).
    :ivar float min_rate: Final minimal foreground rate (e.g. 0.33 as in nnU-Net).
    :ivar int max_epochs: Total number of epochs.
    :ivar float exponent: Exponent for polynomial decay.
    """

    def __init__(self, 
                 dataloader:torch.utils.data.DataLoader, 
                 initial_rate:float, 
                 min_rate:float, 
                 max_epochs:int, 
                 exponent:float=0.9):
        """
        Initialize the foreground scheduler.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            A dataloader whose dataset must implement `set_fg_rate(rate: float)`.
        initial_rate : float
            Initial sampling rate of foreground (typically 1.0).
        min_rate : float
            Final foreground sampling rate (e.g., 0.33).
        max_epochs : int
            Total number of training epochs.
        exponent : float, optional, default=0.9
            Polynomial decay exponent (between 0 and 1).
        """
        self.dataloader = dataloader
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_epochs = max_epochs
        self.exponent = exponent
    
    def on_epoch_begin(self, epoch:int)->None:
        """
        Adjust foreground sampling rate at the beginning of an epoch.

        Parameters
        ----------
        epoch : int
            Current epoch index.

        Returns
        -------
        None
        """
        crt_rate = (self.initial_rate-self.min_rate) * (1 - epoch / self.max_epochs)**self.exponent + self.min_rate
        self.dataloader.dataset.set_fg_rate(crt_rate) 
        print("Current foreground rate: {}".format(crt_rate))

class OverlapScheduler(Callback):
    """
    Callback to schedule the minimum overlap rate between global and local patches using a polynomial decay.

    The overlap rate is progressively reduced during training from an initial rate to a minimum final rate.
    An overlap of 1 means the local patch is fully inside the global patch.
    An overlap of 0 or less means the local patch can be outside the global patch.
    The exact behavior depends on the dataset implementation.

    :ivar torch.utils.data.DataLoader dataloader: DataLoader whose dataset implements `set_min_overlap`.
    :ivar float initial_rate: Initial overlap rate at the start of training.
    :ivar float min_rate: Minimum overlap rate at the end of training.
    :ivar int max_epochs: Total number of training epochs.
    :ivar float exponent: Exponent for the polynomial decay (between 0 and 1).
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        initial_rate: float,
        min_rate: float,
        max_epochs: int,
        exponent: float = 0.9,
        ):
        """
        Initialize the OverlapScheduler callback.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader whose dataset implements a `set_min_overlap` method.
        initial_rate : float
            Starting overlap rate.
        min_rate : float
            Final minimal overlap rate.
        max_epochs : int
            Total number of epochs for training.
        exponent : float, default=0.9
            Exponent controlling the polynomial decay curve.
        """
        self.dataloader = dataloader
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_epochs = max_epochs
        self.exponent = exponent

        self.dataloader.dataset.set_min_overlap(self.initial_rate)
        print("Current overlap: {}".format(self.initial_rate))
    
    def on_epoch_begin(self, epoch:int)->None:
        """
        Adjust and set the minimum overlap rate at the start of an epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-based).

        Returns
        -------
        None
        """
        crt_rate = (self.initial_rate-self.min_rate) * (1 - epoch / self.max_epochs)**self.exponent + self.min_rate
        self.dataloader.dataset.set_min_overlap(crt_rate)
        print("Current overlap: {}".format(crt_rate))

class GlobalScaleScheduler(Callback):
    """
    Callback to schedule the global crop scale using a polynomial decay.

    The scale of the global crop is progressively reduced from the image size to the patch/local crop size.

    :ivar torch.utils.data.DataLoader dataloader: DataLoader whose dataset implements `set_global_crop`.
    :ivar float initial_rate: Initial global scale rate.
    :ivar float min_rate: Minimal/final global scale rate.
    :ivar int max_epochs: Total number of training epochs.
    :ivar float exponent: Exponent for polynomial decay (between 0 and 1).
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        initial_rate: float,
        min_rate: float,
        max_epochs: int,
        exponent: float = 0.9,
        ):
        """
        Initialize the GlobalScaleScheduler callback.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader whose dataset implements a `set_global_crop` method.
        initial_rate : float
            Starting global scale rate.
        min_rate : float
            Final minimal global scale rate.
        max_epochs : int
            Total number of epochs.
        exponent : float, default=0.9
            Exponent controlling the polynomial decay.
        """
        self.dataloader = dataloader
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_epochs = max_epochs
        self.exponent = exponent
    
    def on_epoch_begin(self, epoch: int) -> None:
        """
        Update the global crop scale at the beginning of an epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-based).

        Returns
        -------
        None
        """
        crt_rate = (self.initial_rate-self.min_rate) * (1 - epoch / self.max_epochs)**self.exponent + self.min_rate
        self.dataloader.dataset.set_global_crop(crt_rate)
        print("Current global crop scale: {}".format(crt_rate))

class WeightDecayScheduler(Callback):
    """
    Callback to schedule weight decay during training, useful for DINO re-implementation.

    :ivar torch.optim.Optimizer optimizer: Optimizer used for training.
    :ivar float initial_wd: Initial weight decay value.
    :ivar float final_wd: Final weight decay value.
    :ivar int nb_epochs: Total number of training epochs.
    :ivar bool use_poly: Whether to use polynomial scheduler (True) or cosine scheduler (False).
    :ivar float exponent: Exponent for polynomial decay (between 0 and 1).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_wd: float,
        final_wd: float,
        nb_epochs: int,
        use_poly: bool = False,
        exponent: float = 0.9,
        ):
        """
        Initialize the WeightDecayScheduler callback.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer whose weight decay parameter will be scheduled.
        initial_wd : float
            Starting weight decay value.
        final_wd : float
            Final weight decay value.
        nb_epochs : int
            Total number of training epochs.
        use_poly : bool, default=False
            Use polynomial decay if True; cosine decay otherwise.
        exponent : float, default=0.9
            Exponent controlling polynomial decay (only used if use_poly=True).
        """
        self.optimizer = optimizer
        self.initial_wd = initial_wd
        self.final_wd = final_wd
        self.nb_epochs = nb_epochs
        self.use_poly = use_poly 
        self.exponent = exponent
    
    def on_epoch_begin(self, epoch: int) -> None:
        """
        Update weight decay at the start of each epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-based).

        Returns
        -------
        None
        """
        if self.use_poly:
            self.optimizer.param_groups[0]["weight_decay"] = self.final_wd + (self.initial_wd - self.final_wd) * (1 - epoch / self.nb_epochs)**self.exponent
        else: 
            self.optimizer.param_groups[0]["weight_decay"] = self.final_wd + 0.5 * (self.initial_wd - self.final_wd) * (1 + np.cos(np.pi * epoch / self.nb_epochs))
        print("Current weight decay:", self.optimizer.param_groups[0]["weight_decay"] )

class MomentumScheduler(Callback):
    """
    Momentum scheduler for DINO re-implementation.

    Schedules momentum with different modes: polynomial, exponential, linear, or cosine decay.

    :ivar torch.optim.Optimizer optimizer: Optimizer (optional, if needed).
    :ivar float initial_momentum: Initial momentum value.
    :ivar float final_momentum: Final momentum value.
    :ivar int nb_epochs: Total number of epochs.
    :ivar str mode: Scheduling mode ('poly', 'exp', 'linear', or None for cosine).
    :ivar float exponent: Exponent for polynomial or exponential decay (between 0 and 1).
    :ivar float crt_momentum: Current momentum value.
    """

    def __init__(
        self,
        initial_momentum: float,
        final_momentum: float,
        nb_epochs: int,
        mode: Literal['poly','exp','linear'] | None = 'poly',
        exponent: float = 0.9
    ):
        """
        Initialize the MomentumScheduler.

        Parameters
        ----------
        initial_momentum : float
            Starting momentum value.
        final_momentum : float
            Ending momentum value.
        nb_epochs : int
            Total number of training epochs.
        mode : 'poly','exp','linear' or None, default='poly'
            Scheduling mode: 'poly', 'exp', 'linear', or None (for cosine).
        exponent : float, default=0.9
            Exponent controlling polynomial or exponential decay.
        """        
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.nb_epochs = nb_epochs
        self.crt_momentum = None
        self.mode = mode 
        self.exponent = exponent
    
    def __getitem__(self, epoch: int) -> float:
        """
        Compute momentum value at given epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-based).

        Returns
        -------
        float
            Computed momentum value.
        """
        if self.mode=='poly':
            self.crt_momentum = self.final_momentum + (self.initial_momentum - self.final_momentum) * (1 - epoch / self.nb_epochs)**self.exponent
        elif self.mode=='exp':
            self.crt_momentum = self.final_momentum + (self.initial_momentum - self.final_momentum) * (np.exp(-self.exponent*epoch / self.nb_epochs))
        elif self.mode=='linear':
            if epoch > self.nb_epochs:
                self.crt_momentum = self.final_momentum
            else:
                self.crt_momentum = self.initial_momentum + (self.final_momentum - self.initial_momentum) * epoch / self.nb_epochs
        else: # cosine
            self.crt_momentum = self.final_momentum + 0.5 * (self.initial_momentum - self.final_momentum) * (1 + np.cos(np.pi * epoch / self.nb_epochs))
        return self.crt_momentum

    def on_epoch_end(self, epoch: int) -> None:
        """
        Call at the end of each epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.

        Returns
        -------
        None
        """
        print("Current teacher momentum:", self.crt_momentum)

class DatasetSizeScheduler(Callback):
    """
    Dataset size scheduler.

    Progressively increases the size of the dataset to aid Arcface training.
    The dataloader's dataset must implement a `set_dataset_size` method.
    The model must implement a `set_num_classes` method.
    At each epoch, the dataset size is incremented by 1 (up to max_dataset_size).

    :ivar torch.utils.data.DataLoader dataloader: Dataloader with dataset implementing `set_dataset_size`.
    :ivar torch.nn.Module model: Model implementing `set_num_classes`.
    :ivar int max_dataset_size: Maximum dataset size.
    :ivar int min_dataset_size: Minimum dataset size at start of training.
    
    Notes
    -----
    No proof that this improves final performance.
    """

    def __init__(self, 
                 dataloader:torch.utils.data.DataLoader, 
                 model:torch.nn.Module, 
                 max_dataset_size:int, 
                 min_dataset_size:int=5):
        """
        Initialize the DatasetSizeScheduler.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloader whose dataset must have a `set_dataset_size` method.
        model : torch.nn.Module
            Model with a `set_num_classes` method.
        max_dataset_size : int
            Maximum size of the dataset.
        min_dataset_size : int, default=5
            Initial minimal dataset size.
        """
        self.dataloader = dataloader
        self.model = model
        self.max_dataset_size = max_dataset_size
        self.min_dataset_size = min_dataset_size

    def on_epoch_begin(self, epoch: int) -> None:
        """
        Call at the beginning of an epoch.

        Increases the dataset size and updates the model number of classes.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).

        Returns
        -------
        None
        """
        dataset_size = min(epoch+self.min_dataset_size, self.max_dataset_size)
        self.dataloader.dataset.set_dataset_size(dataset_size)
        self.model.set_num_classes(dataset_size)
        print("Current dataset size: {}".format(dataset_size))

#----------------------------------------------------------------------------
# metrics updater
class MetricsUpdater(Callback):
    """
    Update the metrics averages by calling the `update` method of each metric.

    :ivar list[Metric] metrics: List of metrics and losses to update.
    :ivar int batch_size: Batch size used to update metrics.
    """

    def __init__(self, metrics:list[Metric], batch_size:int):
        """
        Initialize the MetricsUpdater callback.

        Parameters
        ----------
        metrics : list of biom3d.Metric
            List of metrics and losses to update.
        batch_size : int
            Batch size.
        """
        self.metrics = metrics
        self.batch_size = batch_size

    def on_epoch_begin(self, epoch:Optional[int]=None)->None:
        """
        Call at the beginning of an epoch.

        Resets all metrics.

        Parameters
        ----------
        epoch : int, optional
            Current epoch index (not used).

        Returns
        -------
        None
        """
        for m in self.metrics: m.reset()

    def on_batch_end(self, batch: Optional[int] = None) -> None:
        """
        Call at the end of a batch.

        Updates all metrics with the batch size.

        Parameters
        ----------
        batch : int, optional
            Current batch index (not used).

        Returns
        -------
        None
        """
        for m in self.metrics: m.update(self.batch_size)

#----------------------------------------------------------------------------