#---------------------------------------------------------------------------
# Callback are periodically called during training. 
# There are currently 3 different periods:
# - training 
# - epoch
# - batch
#---------------------------------------------------------------------------

import torch
import os
from shutil import copyfile
# from telegram_send import send
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # bug fix: change matplotlib backend 
from torch.utils.tensorboard import SummaryWriter

import numpy as np

#----------------------------------------------------------------------------
# Base classes

class Callback(object):
    """Abstract base class used to build new callbacks.

    Callback are periodically called during training. 
    There are currently 3 different periods:
    - training 
    - epoch
    - batch
    Callbacks are called either before or after a period. 

    Each method starting by `on_` can be overridden. This method will be called at a certain time point during the training process.
    For instance, the `on_epoch_end` method will be called in the end of each epoch. 
    """

    def __init__(self):
        self.metrics = None

    def set_trainer(self, metrics):
        self.metrics = metrics

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_train_begin(self, epoch=None):
        pass

    def on_train_end(self, epoch=None):
        pass

# dict based callbacks

class Callbacks(Callback):
    """Child of biom3d.callbacks.Callback and used to compile all the callbacks together. 

    Parameters
    ----------
    callbacks : list of biom3d.callback.Callback
        A list of callbacks.
    """
    def __init__(self, callbacks):
        super().__init__()
        if isinstance(callbacks, Callbacks):
            callbacks = callbacks.callbacks
        self.callbacks = callbacks
        if callbacks is None:
            self.callbacks = {}
    
    def __getitem__(self, name): return self.callbacks[name]

    def set_trainer(self, trainer):
        for callback in self.callbacks.values():
            callback.set_trainer(trainer)

    def on_batch_begin(self, batch):
        for callback in self.callbacks.values():
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch):
        for callback in self.callbacks.values():
            callback.on_batch_end(batch)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks.values():
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks.values():
            callback.on_epoch_end(epoch)

    def on_train_begin(self, epoch=None):
        for callback in self.callbacks.values():
            callback.on_train_begin(epoch)

    def on_train_end(self, epoch=None):
        for callback in self.callbacks.values():
            callback.on_train_end(epoch)

#----------------------------------------------------------------------------
# Savers            

class ModelSaver(Callback): # TODO: save best_only
    """The model saver saves the model, the epoch, the optimizer and the loss in the end of each epoch. It can also save the best model.
    
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
    loss : torch.nn.Module
        Loss function.
    saved_loss : torch.nn.Module
        Loss saved alongside the model.
    """
    def __init__(self,
        model, 
        optimizer, 
        path='unet', 
        every_epoch=2, 
        save_best=True, 
        loss=None, # loss used for saving the best model, generally the val loss
        saved_loss=None, # loss being saved
        ):

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

    def on_train_begin(self, epoch):
        """If the model already exists, it probably means that retraining is intended.
        This method thus copies the model saved parameters before retraining it (to avoid overwriting).
        """
        if os.path.exists(self.path_last):
            copyfile(self.path_last, self.path + "_" + str(epoch) + ".pth")
        if os.path.exists(self.path_best):
            copyfile(self.path_best, self.path + "_" + str(epoch) + "_best.pth")

    def on_epoch_end(self, epoch):
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
    """Save logs in a CSV file in the end of each epoch.

    This callback creates a `log.csv` file in the log folder.
    
    Parameters
    ----------
    log_dir : str
        Path to the folder where the CSV file will be stored. 
    train_loss : torch.nn.Module
        Training loss.
    val_loss : torch.nn.Module
        Validation loss.
    train_metrics : list of torch.nn.Module
        List of training metrics.
    val_metrics : list of torch.nn.Module
        List of validation metrics.
    scheduler : biom3d.callback.Callback
        Learning rate callback.
    """
    def __init__(self, 
        log_dir,            # path to where the csv file will be store
        train_loss,
        val_loss=None,
        train_metrics=None,
        val_metrics=None,
        scheduler=None,
        # save_best=False,    # save the best metrics in another text file ? 
        # every_batch=10
        ):    # save period (in batch)

        # old documentation about save best:
        # save_best : bool, default=True
        #     Whether to save the best loss in other CSV file. This will create another CSV file that will be called `log_best.csv` in the log folder.
        # every_batch : int, default=10
        #     Batch period when to save the log.

        self.path = os.path.join(log_dir,'log.csv')

        self.crt_epoch = 1

        self.train_loss = train_loss
        self.val_loss = val_loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

        self.scheduler = scheduler if hasattr(scheduler, 'get_last_lr') else None

        # if save_best:
        #     self.save_best = save_best # self.tracked_loss = val_loss
        #     self.best_loss = float('inf')
        #     self.best_path = log_dir + '/log_best.csv'
        #     self.f_best = None

        # self.every_batch = every_batch

        f = open(self.path, "a")
        if os.stat(self.path).st_size == 0: # if the file is empty
            self.write_file_head()
        f.close()

    def write_file_head(self):
        """Write the head of the CSV file.
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

    # def on_epoch_begin(self, epoch):
    #     self.crt_epoch = epoch+1

    # def on_batch_end(self, batch):
    #     if batch % self.every_batch == 0:
    #         template =  str(self.crt_epoch)
    #         template += "," + str(batch)
    #         template += "," + str(self.scheduler.get_last_lr()[0])
    #         template += "," + str(self.train_loss.val.item()) # TODO: save the avg value only
    #         # BUG fix below: TODO, simplify it! 
    #         val_loss = self.val_loss.val if type(self.val_loss.val)==int else self.val_loss.val.item()
    #         template += "," + str(val_loss)
    #         for m in self.train_metrics: template += "," + str(m.val.item())
    #         for m in self.val_metrics: 
    #             # TODO: cf. previous comment
    #             val_m = m.avg if type(m.avg)==int else m.avg.item() 
    #             template += "," + str(val_m)
    #         self.f.write(template + "\n")
        
            # save best if needed
            # if self.save_best \
            #     and (self.best_loss > self.val_loss.val) \
            #     and (self.val_loss.avg != 0): # to avoid saving during the first epoch
            #     self.best_loss = self.val_loss.avg
            #     self.f_best = open(self.best_path, "w") # open it in write mode
            #     self.write_file_head(self.f_best)
            #     self.f_best.write(template)
            #     self.f_best.close()
    
    def on_epoch_end(self, epoch):
        f = open(self.path, "a")
        template =  str(epoch)

        # add the scheduler value
        if self.scheduler is not None:
            template += "," + str(self.scheduler.get_last_lr()[0])

        # add the learning rate and the training loss
        template += "," + str(self.train_loss.avg.item()) # TODO: save the avg value only
        
        # add the validation loss if needed
        if self.val_loss is not None:
            val_loss = self.val_loss.avg if type(self.val_loss.avg)==float else self.val_loss.avg.item()
            template += "," + str(val_loss)
        
        # add the training metrics
        if self.train_metrics is not None:
            for m in self.train_metrics: template += "," + str(m.val.item())

        # adde the validation metrics
        if self.val_metrics is not None:
            for m in self.val_metrics: 
                val_m = m.avg if type(m.avg)==int else m.avg.item() 
                template += "," + str(val_m)
        
        # write in the output file
        f.write(template + "\n")
        f.close()

class ImageSaver(Callback):
    """The image saver callback saves a small snapshot of the raw image, the prediction and the ground truth. By default, it uses the validation dataloader to load a batch of 3D images, makes a prediction with the model, then uses the first images of the batch and the prediction and, to display a 2D image, uses the first channel of each image (input, prediction, ground truth).

    Parameters
    ----------
    image_dir : str
        Path to the image snapshot folder, where the snapshots will be stored.
    model : torch.nn.Module
        A torch module that will be used to make prediction.
    val_dataloader : torch.utils.data.Dataloader or iter
        An iterator used to generate a batch of input image and mask.
    use_sigmoid : bool, default=True
        Whether to use a sigmoid or a softmax activation on the model predictions.
    every_epoch : int, default=1
        Epoch period when the image snapshot will be stored.
    plot_size : int, default=1
        Number of images to plot in the snapshot. 
    use_fp16 : bool, default=True
        Whether the model has been trained with AMP or not.
    """
    def __init__(self, 
        image_dir, 
        model,
        val_dataloader,
        use_sigmoid=True,
        every_epoch=1,
        plot_size=1,
        use_fp16=True,
        ):
        self.image_dir = image_dir
        self.every_epoch = every_epoch
        self.model = model 
        self.use_sigmoid = use_sigmoid
        self.val_dataloader = val_dataloader
        self.plot_size = plot_size
        self.use_fp16 = use_fp16

    def on_epoch_end(self, epoch):
        if epoch % self.every_epoch == 0:
            self.model.eval()
            with torch.no_grad():
                for i in range(self.plot_size):
                # make prediction
                    X, y = next(iter(self.val_dataloader))
                    # X, y = self.val_dataloader.get_sample()
                    if torch.cuda.is_available():
                        X, y = X.cuda(), y.cuda()
                    with torch.cuda.amp.autocast(self.use_fp16):
                        pred = self.model(X)
                        if type(pred)==list:
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
                    # plt.figure(figsize=(15,5))
                    plt.figure(dpi=100)
                    # print original
                    plt.subplot(self.plot_size,3,3*i+1)
                    plt.imshow(X)
                    plt.title('raw')
                    plt.axis('off')
                    # print prediction
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
    """The tensorboard callback is used to plot the loss and metrics in the tensorboard interface. To start tensorboard execute the following command in a terminal opened in the biom3d project folder: `tensorboard --logdir=logs/`.
    
    This callback labels the loss with `Loss/train` and `Loss/test` and the metrics with `Metrics/name_of_your_metric`.

    Parameters
    ----------
    log_dir : str
        Path to the log folder where the curve will be stored.
    train_loss : torch.nn.Module
        Training loss function.
    val_loss : torch.nn.Module
        Validation loss function.
    train_metrics : list of torch.nn.Module
        List of the training metrics.
    val_metrics : list of torch.nn.Module
        List of the validation metrics.
    batch_size : int
        Size of the minibatch. Is used to compute the number of iteration so that the x-axis of the curve is the number of image and not the number of batch.
    n_batch_per_epoch : int
        Number of the batch per epoch. Is used to compute the number of iteration so that the x-axis of the curve is the number of image and not the number of batch.
    """
    def __init__(self, 
        log_dir, 
        train_loss, 
        val_loss, 
        train_metrics, 
        val_metrics, 
        batch_size,
        n_batch_per_epoch):

        self.writer = SummaryWriter(log_dir=log_dir)
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.batch_size = batch_size
        self.n_batch_per_epoch = n_batch_per_epoch

        self.crt_epoch = 0
    
    def on_epoch_begin(self, epoch):
        self.crt_epoch = epoch
    
    # def on_batch_end(self, batch):
    def on_epoch_end(self, epoch):
        # n_iter = (self.n_batch_per_epoch * self.crt_epoch + batch) * self.batch_size
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
    """Print the log in the terminal prompt.

    Parameters
    ----------
    metrics : list of torch.nn.Module
        List of the metrics to display. The loss can be amoung them.
    nbof_eopchs : int
        Number of epochs in the training.
    nbof_batchs : int
        Number of batches per epoch.
    every_batch : int, default=10
        Batch period when to print the log.
    """
    def __init__(
        self,
        metrics,
        nbof_epochs,
        nbof_batches,
        every_batch=10):

        self.nbof_epochs = nbof_epochs
        self.nbof_batches = nbof_batches

        self.metrics = metrics

        self.every_batch = every_batch

    def on_epoch_begin(self, epoch):
        print("Epoch [{:>3d}/{:>3d}]".format(epoch+1, self.nbof_epochs))

    def on_batch_end(self, batch):
        if batch % self.every_batch == 0:
            template = "Batch [{:>3d}/{:>3d}]".format(batch, self.nbof_batches)
            for i in range(len(self.metrics)):
                template += ", {}".format(self.metrics[i])
            print(template)

# class Telegram(Callback):
#     """
#     Send message when training finishes
#     """
#     def __init__(self, loss, test_loss=None):
#         self.loss = loss
#         self.test_loss = test_loss
#         self.template = "Finished training with loss {} and test loss {}"
    
#     def on_train_end(self):
#         send(messages=[self.template.format(self.loss.val, self.test_loss.avg)])

#----------------------------------------------------------------------------
# schedulers

class LRSchedulerMultiStep(Callback):
    """Multi-step learning rate scheduler. Uses torch.optim.lr_scheduler.MultiStepLR

    For more details, see : https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Training optimizer.
    milestones : list of int
        List of epoch when to decay the learning rate. Must be increasing. 
    gamma : float, default=0.1
        Multiplicative factor of the learning rate decay. 
    """
    def __init__(self, optimizer, milestones, gamma=0.1):
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, verbose=False)

    def get_last_lr(self):
        """Getter for the last learning rate.
        """
        return self.scheduler.get_last_lr()
    
    def on_epoch_end(self, epoch):
        self.scheduler.step()
        print("Current learning rate: {}".format(self.scheduler.get_last_lr()))

class LRSchedulerCosine(Callback):
    """Cosine learning rate scheduler. Uses torch.optim.lr_scheduler.CosineAnnealingLR

    For more details, see : https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Training optimizer.
    T_max : int
        Maximum number of iteration. By default, you can set this to the number of epochs. 
    """
    def __init__(self, optimizer, T_max):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=1e-6, verbose=False)

    def get_last_lr(self):
        """Getter for the last learning rate.
        """
        return self.scheduler.get_last_lr()
    
    def on_epoch_end(self, epoch):
        self.scheduler.step()
        print("Current learning rate: {}".format(self.scheduler.get_last_lr()))

class LRSchedulerPoly(Callback):
    """Polynomial scheduler. Similar to nnU-Net learning rate scheduler.

    .. math::
        \\begin{aligned}
            lr_{new} = lr_{initial} * (1 - \\frac{epoch_{current}}{epoch_{max}})^{exponent}
        \\end{aligned}

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Training optimizer.
    initial_lr : float
        Initial learning rate.
    max_epochs : int
        Number of epochs in the training.
    exponent : float, default=0.9
        Must be between 0 and 1. Exponent of the polynomial decay.
    """
    def __init__(self, optimizer, initial_lr, max_epochs, exponent=0.9):
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.exponent = exponent
        self.optimizer = optimizer

    # def get_last_lr(self):
    #     return self.optimizer.param_groups[0]['lr']
    
    # def on_train_begin(self, epoch=None):
    #     self.optimizer.param_groups[0]['lr'] = self.initial_lr * (1 - epoch / self.max_epochs)**self.exponent
    #     print("Current learning rate: {}".format(self.optimizer.param_groups[0]['lr']))

    def on_epoch_begin(self, epoch):
        self.optimizer.param_groups[0]['lr'] = self.initial_lr * (1 - epoch / self.max_epochs)**self.exponent
        print("Current learning rate: {}".format(self.optimizer.param_groups[0]['lr']))


# class LRSchedulerCosine(Callback):
#     """
#     Multi-step scheduler only for now
#     """
#     def __init__(self, optimizer, T_max):
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6, last_epoch=-1, verbose=False)

#     def get_last_lr(self):
#         return self.scheduler.get_last_lr()
    
#     def on_epoch_end(self, epoch):
#         self.scheduler.step()
#         print("Current learning rate: {}".format(self.scheduler.get_last_lr()))

class ForceFGScheduler(Callback):
    """Force foreground scheduler. Uses a polynomial scheduler.
    We force the model to "see" the foreground more often in the beginning of the training and progressively reduce the foreground rate.
    The callback calls the dataloader.dataset.set_fg_rate method. The dataset must thus possess a set_fg_rate function.
    This function does not change directly the way the foreground images are presented to the deep learning model. This should be implemented in the Dataset class.

    Parameters
    ----------
    dataloader : torch.utils.data.Dataloader 
        A dataloader which dataset member possesses a `set_fg_rate` method.
    initial_rate : float 
        Initial foreground rate. If the rate is equal to 1, it means that we only present foreground. On the contrary, if the foreground rate is set to 0, then the patch will be random. (Again, this depends on how the foreground forcing is defined in the dataloader.dataset).
    min_rate : float
        Minimal/final foreground rate. Can be set to zero but it is not recommended. A good choice could be 0.33 (such as in nnU-Net implementation).
    max_epochs : int
        Number of epochs in the training.
    exponent : float, default=0.9
        Must be between 0 and 1. Exponent of the polynomial decay.
    """
    def __init__(self, dataloader, initial_rate, min_rate, max_epochs, exponent=0.9):
        self.dataloader = dataloader
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_epochs = max_epochs
        self.exponent = exponent

    # def on_train_begin(self, epoch=None):
    #     self.dataloader.dataset.set_fg_rate(self.initial_rate)
    #     print("Current foreground rate: {}".format(self.initial_rate))
    
    def on_epoch_begin(self, epoch):
        crt_rate = (self.initial_rate-self.min_rate) * (1 - epoch / self.max_epochs)**self.exponent + self.min_rate
        self.dataloader.dataset.set_fg_rate(crt_rate) 
        print("Current foreground rate: {}".format(crt_rate))

class OverlapScheduler(Callback):
    """Overlapping rate scheduler. Uses a polynomial scheduler.
    We progressively reduce the minimum overlap between the global patches and the local patches.
    An overlap of 1 means that the local patch is completely included in the global patch. An overlap of 0 or less, means that the local patch can be located outside of the global patch. The exact definition of the relationship between the local and the global patch should be defined the dataset.
    
    Parameters
    ----------
    dataloader : torch.utils.data.Dataloader 
        A dataloader which dataset member possesses a `set_min_overlap` method.
    initial_rate : float 
        Initial overlapping rate. 
    min_rate : float
        Minimal/final overlapping rate. 
    max_epochs : int
        Number of epochs in the training.
    exponent : float, default=0.9
        Must be between 0 and 1. Exponent of the polynomial decay.
    """
    def __init__(self, dataloader, initial_rate, min_rate, max_epochs, exponent=0.9):
        self.dataloader = dataloader
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_epochs = max_epochs
        self.exponent = exponent

    # def on_train_begin(self, epoch=None):
        self.dataloader.dataset.set_min_overlap(self.initial_rate)
        print("Current overlap: {}".format(self.initial_rate))
    
    def on_epoch_begin(self, epoch):
        crt_rate = (self.initial_rate-self.min_rate) * (1 - epoch / self.max_epochs)**self.exponent + self.min_rate
        self.dataloader.dataset.set_min_overlap(crt_rate)
        print("Current overlap: {}".format(crt_rate))

class GlobalScaleScheduler(Callback):
    """Global scale scheduler. Uses a polynomial scheduler.
    We progressively reduce the scale of the global_crop from image size to patch/local_crop size.

    Parameters
    ----------
    dataloader : torch.utils.data.Dataloader 
        A dataloader which dataset member possesses a `set_global_crop` method.
    initial_rate : float 
        Initial global scale rate. 
    min_rate : float
        Minimal/final global scale rate. 
    max_epochs : int
        Number of epochs in the training.
    exponent : float, default=0.9
        Must be between 0 and 1. Exponent of the polynomial decay.
    """
    def __init__(self, dataloader, initial_rate, min_rate, max_epochs, exponent=0.9):
        self.dataloader = dataloader
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_epochs = max_epochs
        self.exponent = exponent

    # def on_train_begin(self, epoch=None):
    #     self.dataloader.dataset.set_global_crop(self.initial_rate)
    #     print("Current global crop scale: {}".format(self.initial_rate))
    
    def on_epoch_begin(self, epoch):
        crt_rate = (self.initial_rate-self.min_rate) * (1 - epoch / self.max_epochs)**self.exponent + self.min_rate
        self.dataloader.dataset.set_global_crop(crt_rate)
        print("Current global crop scale: {}".format(crt_rate))

class WeightDecayScheduler(Callback):
    """Weight decay scheduler. Used for DINO re-implementation.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Training optimizer.
    initial_wd : float
        Initial weight decay.
    final_wd : float
        Final weight decay.
    nb_epochs : int
        Number of epochs in the training.
    use_poly : bool, default=True
        Whether to use polynomial scheduler instead of the default cosine scheduler.
    exponent : float, default=0.9
        Must be between 0 and 1. Exponent of the polynomial decay.
    """
    def __init__(self, optimizer, initial_wd, final_wd, nb_epochs, use_poly=False, exponent=0.9):
        self.optimizer = optimizer
        self.initial_wd = initial_wd
        self.final_wd = final_wd
        self.nb_epochs = nb_epochs
        self.use_poly = use_poly 
        self.exponent = exponent

    # def on_train_begin(self, epoch=None):
    #     self.optimizer.param_groups[0]["weight_decay"] = self.initial_wd
    #     print("Current weight decay:", self.optimizer.param_groups[0]["weight_decay"] )
    
    def on_epoch_begin(self, epoch):
        if self.use_poly:
            self.optimizer.param_groups[0]["weight_decay"] = self.final_wd + (self.initial_wd - self.final_wd) * (1 - epoch / self.nb_epochs)**self.exponent
        else: 
            self.optimizer.param_groups[0]["weight_decay"] = self.final_wd + 0.5 * (self.initial_wd - self.final_wd) * (1 + np.cos(np.pi * epoch / self.nb_epochs))
        print("Current weight decay:", self.optimizer.param_groups[0]["weight_decay"] )

class MomentumScheduler(Callback):
    """Momentum scheduler. Used for DINO re-implementation.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Training optimizer.
    initial_momentum : float
        Initial momentum decay.
    final_momentum : float
        Final momentum decay.
    nb_epochs : int
        Number of epochs in the training.
    mode : str, default='poly'
        Scheduling mode. Can be one of: 'poly', 'exp', 'linear' or None. If None, use cosine scheduler.
    exponent : float, default=0.9
        Must be between 0 and 1. Exponent of the polynomial decay.
    """
    def __init__(self, initial_momentum, final_momentum, nb_epochs, mode='poly', exponent=0.9):
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.nb_epochs = nb_epochs
        self.crt_momentum = None
        self.mode = mode 
        self.exponent = exponent
    
    def __getitem__(self, epoch):
        if self.mode=='poly':
            self.crt_momentum = self.final_momentum + (self.initial_momentum - self.final_momentum) * (1 - epoch / self.nb_epochs)**self.exponent
        elif self.mode=='exp':
            self.crt_momentum = self.final_momentum + (self.initial_momentum - self.final_momentum) * (np.exp(-self.exponent*epoch / self.nb_epochs))
        elif self.mode=='linear':
            if epoch > self.nb_epochs:
                self.crt_momentum = self.final_momentum
            else:
                self.crt_momentum = self.initial_momentum + (self.final_momentum - self.initial_momentum) * epoch / self.nb_epochs
        # elif self.mode=='mix': # linear increase and then an exponential increase
        #     if epoch > self.nb_epochs:
        #         self.crt_momentum = self.final_momentum
        #     else:
        #         self.crt_momentum = self.initial_momentum + (self.final_momentum - self.initial_momentum) * epoch / self.nb_epochs
        else: # cosine
            self.crt_momentum = self.final_momentum + 0.5 * (self.initial_momentum - self.final_momentum) * (1 + np.cos(np.pi * epoch / self.nb_epochs))
        return self.crt_momentum

    def on_epoch_end(self, epoch):
        print("Current teacher momentum:", self.crt_momentum)

class DatasetSizeScheduler(Callback):
    """Dataset size scheduler. 
    We progressively increase the size of the dataset to help the Arcface training. NO PROOF THAT IT IMPROVES THE FINAL PERFORMANCE.
    The dataloader must have a dataset that has a `set_num_classes` method.
    At each epoch the size of the dataset is incremented by 1.

    Parameters
    ----------
    dataloader : torch.utils.data.Dataloader
        Torch dataloader. The dataloader must have a dataset that has a `set_num_classes` method.
    model : torch.nn.Module
        Torch module that must have a `set_num_classes` method.
    max_dataset_size : int
        Maximum size of the dataset.
    min_dataset_size : int
        Minimum size of the dataset. Used in the beginning of the training.
    """
    def __init__(self, dataloader, model, max_dataset_size, min_dataset_size=5):
        self.dataloader = dataloader
        self.model = model
        self.max_dataset_size = max_dataset_size
        self.min_dataset_size = min_dataset_size

    def on_epoch_begin(self, epoch):
        dataset_size = min(epoch+self.min_dataset_size, self.max_dataset_size)
        self.dataloader.dataset.set_dataset_size(dataset_size)
        self.model.set_num_classes(dataset_size)
        print("Current dataset size: {}".format(dataset_size))

#----------------------------------------------------------------------------
# metrics updater

class MetricsUpdater(Callback):
    """Update the metrics averages by calling the `update` method of each metric.

    Parameters
    ----------
    metrics : list of torch.nn.Module
        List of metrics and losses to update.
    batch_size : int
        Batch size.
    """
    def __init__(self, metrics, batch_size):
        self.metrics = metrics
        self.batch_size = batch_size

    def on_epoch_begin(self, epoch=None):
        for m in self.metrics: m.reset()

    def on_batch_end(self, batch=None):
        for m in self.metrics: m.update(self.batch_size)

#----------------------------------------------------------------------------