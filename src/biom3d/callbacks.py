import torch
import os
# from telegram_send import send
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import numpy as np

#----------------------------------------------------------------------------
# Base classes

class Callback(object):
    """
    Abstract base class used to build new callbacks.
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

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

# dict based callbacks

class Callbacks(Callback):
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

    def on_train_begin(self):
        for callback in self.callbacks.values():
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks.values():
            callback.on_train_end()

#----------------------------------------------------------------------------
# Savers            
            
class ModelSaver(Callback): # TODO: save best_only
    """
    if model is a list then it is considered as been [student,teacher]
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
        self.path_last = path + '.pth'
        self.path_best = path + '_best.pth'
        self.every_epoch = every_epoch
        self.save_best = save_best
        self.best_loss = float('inf')
        self.loss = loss
        self.saved_loss = saved_loss

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
            torch.save(save_dict, self.path_best)
            print('Save best model to {}'.format(self.path_best))
            self.best_loss = self.loss.avg
            
                

class LogSaver(Callback):
    def __init__(self, 
        log_dir,            # path to where the csv file will be store
        train_loss,
        val_loss,
        train_metrics,
        val_metrics,
        scheduler=None,
        save_best=False,    # save the best metrics in another text file ? 
        every_batch=10):    # save period (in batch)

        self.path = log_dir + '/log.csv'

        self.crt_epoch = 1

        self.train_loss = train_loss
        self.val_loss = val_loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

        self.scheduler = scheduler

        if save_best:
            self.save_best = save_best # self.tracked_loss = val_loss
            self.best_loss = float('inf')
            self.best_path = log_dir + '/log_best.csv'
            self.f_best = None

        self.every_batch = every_batch

        self.f = None

    def write_file_head(self, file):
        # write the head of the log file
        head = "epoch,batch,learning_rate,train_loss,val_loss"
        for m in self.train_metrics+self.val_metrics:
            head += "," + m.name
        file.write(head + "\n")

    def on_train_begin(self):
        # create the file and edit the head 
        self.f = open(self.path, "a")
        if os.stat(self.path).st_size == 0: # if the file is empty
            self.write_file_head(self.f)

    def on_epoch_begin(self, epoch):
        self.crt_epoch = epoch+1

    def on_batch_end(self, batch):
        if batch % self.every_batch == 0:
            template =  str(self.crt_epoch)
            template += "," + str(batch)
            template += "," + str(self.scheduler.get_last_lr()[0])
            template += "," + str(self.train_loss.val.item()) # TODO: save the avg value only
            # BUG fix below: TODO, simplify it! 
            val_loss = self.val_loss.val if type(self.val_loss.val)==int else self.val_loss.val.item()
            template += "," + str(val_loss)
            for m in self.train_metrics: template += "," + str(m.val.item())
            for m in self.val_metrics: 
                # TODO: cf. previous comment
                val_m = m.avg if type(m.avg)==int else m.avg.item() 
                template += "," + str(val_m)
            self.f.write(template + "\n")
        
            # save best if needed
            if self.save_best \
                and (self.best_loss > self.val_loss.val) \
                and (self.val_loss.avg != 0): # to avoid saving during the first epoch
                self.best_loss = self.val_loss.avg
                self.f_best = open(self.best_path, "w") # open it in write mode
                self.write_file_head(self.f_best)
                self.f_best.write(template)
                self.f_best.close()

# for image saver TODO:
# use the run_prediction_single method of the Builder class itself
# EXAMPLE:
# class test:
#     def __init__(self):
#         self.a = 0
#     def t(self):
#         self.a += 1
#     def __call__(self):
#         caller(self, self.t)

# def caller(tester, fct):
#     print(tester)
#     print(tester.a)
#     fct()
#     print(tester.a)

# tester = test()
# tester()

class ImageSaver(Callback):
    def __init__(self, 
        image_dir, 
        model,
        val_dataloader,
        use_sigmoid=True,
        every_epoch=1):
        self.image_dir = image_dir
        self.every_epoch = every_epoch
        self.model = model 
        self.use_sigmoid = use_sigmoid
        self.val_dataloader = val_dataloader

    def on_epoch_end(self, epoch):
        if epoch % self.every_epoch == 0:
            self.model.eval()
            with torch.no_grad():
                # for _ in range(4): next(iterator) # to start with image number 4
                plot_size = 1
                for i in range(plot_size):
                # make prediction
                    X, y = next(iter(self.val_dataloader))
                    # X, y = self.val_dataloader.get_sample()
                    X, y = X.cuda(), y.cuda()
                    with torch.cuda.amp.autocast():
                        pred = self.model(X).detach()
                    if self.use_sigmoid:
                        pred = (torch.sigmoid(pred)>0.5).int()*255
                    else: 
                        pred = (pred.softmax(dim=1)).int()*255
                    l = [X, y, pred]
                    for j in range(len(l)):
                        _,_,channel,_,_ = l[j].shape
                        l[j] = l[j][0, 0, channel//2, ...].cpu().numpy().astype(float)
                    X, y, pred = l

                    # plot 
                    # plt.figure(figsize=(15,5))
                    plt.figure(dpi=100)
                    # print original
                    plt.subplot(plot_size,3,3*i+1)
                    plt.imshow(X)
                    plt.title('raw')
                    plt.axis('off')
                    # print prediction
                    plt.subplot(plot_size,3,3*i+2)
                    plt.imshow(pred)
                    plt.title('pred')
                    plt.axis('off')
                    # print ground truth
                    plt.subplot(plot_size,3,3*i+3)
                    plt.imshow(y)
                    plt.title('gt')
                    plt.axis('off')
                    del X, y, pred
                im_path = os.path.join(self.image_dir,'image_' + str(epoch) + '.png')
                print("Save image to {}".format(im_path))
                plt.savefig(im_path)
                plt.close()

class TensorboardSaver(Callback):
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
    
    def on_batch_end(self, batch):
        n_iter = (self.n_batch_per_epoch * self.crt_epoch + batch) * self.batch_size
        self.writer.add_scalar('Loss/train', self.train_loss.val, n_iter)
        if self.val_loss:
            self.writer.add_scalar('Loss/test', self.val_loss.avg, n_iter)
        if self.train_metrics:
            for m in self.train_metrics: self.writer.add_scalar('Metrics/'+m.name,m.val,n_iter)
        if self.val_metrics:
            for m in self.val_metrics: self.writer.add_scalar('Metrics/'+m.name,m.avg,n_iter)

#----------------------------------------------------------------------------
# Printer

class LogPrinter(Callback):
    def __init__(self, metrics, nbof_epochs, nbof_batches, every_batch=10):
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
    """
    Multi-step scheduler only for now
    """
    def __init__(self, optimizer, milestones, gamma=0.1):
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, verbose=False)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()
    
    def on_epoch_end(self, epoch):
        self.scheduler.step()
        print("Current learning rate: {}".format(self.scheduler.get_last_lr()))

class LRSchedulerCosine(Callback):
    """
    Cosine scheduler
    """
    def __init__(self, optimizer, T_max):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=1e-6, verbose=False)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()
    
    def on_epoch_end(self, epoch):
        self.scheduler.step()
        print("Current learning rate: {}".format(self.scheduler.get_last_lr()))

class LRSchedulerPoly(Callback):
    """
    Polygonal scheduler
    """
    def __init__(self, optimizer, initial_lr, max_epochs, exponent=0.9):
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.exponent = exponent
        self.optimizer = optimizer

    # def get_last_lr(self):
    #     return self.optimizer.param_groups[0]['lr']
    
    def on_train_begin(self):
        self.optimizer.param_groups[0]['lr'] = self.initial_lr
        print("Current learning rate: {}".format(self.optimizer.param_groups[0]['lr']))

    def on_epoch_end(self, epoch):
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
    """
    Force foreground scheduler
    We force the model to "see" the foreground more often in the beginning and progressively reduce it
    This function edits the dataloader.dataset.fg_rate argument --> TODO: improve that
    """
    def __init__(self, dataloader, initial_rate, min_rate, max_epochs, exponent=0.9):
        self.dataloader = dataloader
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_epochs = max_epochs
        self.exponent = exponent

    def on_train_begin(self):
        self.dataloader.dataset.set_fg_rate(self.initial_rate)
        print("Current foreground rate: {}".format(self.initial_rate))
    
    def on_epoch_end(self, epoch):
        crt_rate = (self.initial_rate-self.min_rate) * (1 - epoch / self.max_epochs)**self.exponent + self.min_rate
        self.dataloader.dataset.set_fg_rate(crt_rate) 
        print("Current foreground rate: {}".format(crt_rate))

class OverlapScheduler(Callback):
    """
    Overlap scheduler
    We progressively reduce the minimum overlap between the global patches and the local patches
    """
    def __init__(self, dataloader, initial_rate, min_rate, max_epochs, exponent=0.9):
        self.dataloader = dataloader
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_epochs = max_epochs
        self.exponent = exponent

    def on_train_begin(self):
        self.dataloader.dataset.set_min_overlap(self.initial_rate)
        print("Current overlap: {}".format(self.initial_rate))
    
    def on_epoch_end(self, epoch):
        crt_rate = (self.initial_rate-self.min_rate) * (1 - epoch / self.max_epochs)**self.exponent + self.min_rate
        self.dataloader.dataset.set_min_overlap(crt_rate)
        print("Current overlap: {}".format(crt_rate))

class GlobalScaleScheduler(Callback):
    """
    Global scale scheduler
    We progressively reduce the scale of the global_crop from image size to patch size
    """
    def __init__(self, dataloader, initial_rate, min_rate, max_epochs, exponent=0.9):
        self.dataloader = dataloader
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_epochs = max_epochs
        self.exponent = exponent

    def on_train_begin(self):
        self.dataloader.dataset.set_global_crop(self.initial_rate)
        print("Current global crop scale: {}".format(self.initial_rate))
    
    def on_epoch_end(self, epoch):
        crt_rate = (self.initial_rate-self.min_rate) * (1 - epoch / self.max_epochs)**self.exponent + self.min_rate
        self.dataloader.dataset.set_global_crop(crt_rate)
        print("Current global crop scale: {}".format(crt_rate))

class WeightDecayScheduler(Callback):
    """
    cosine scheduler of the weight decay
    if use_poly=True, use polynomial update instead of cosine
    """
    def __init__(self, optimizer, initial_wd, final_wd, nb_epochs, use_poly=False, exponent=0.9):
        self.optimizer = optimizer
        self.initial_wd = initial_wd
        self.final_wd = final_wd
        self.nb_epochs = nb_epochs
        self.use_poly = use_poly 
        self.exponent = exponent

    def on_train_begin(self):
        self.optimizer.param_groups[0]["weight_decay"] = self.initial_wd
        print("Current weight decay:", self.optimizer.param_groups[0]["weight_decay"] )
    
    def on_epoch_end(self, epoch):
        if self.use_poly:
            self.optimizer.param_groups[0]["weight_decay"] = self.final_wd + (self.initial_wd - self.final_wd) * (1 - epoch / self.nb_epochs)**self.exponent
        else: 
            self.optimizer.param_groups[0]["weight_decay"] = self.final_wd + 0.5 * (self.initial_wd - self.final_wd) * (1 + np.cos(np.pi * epoch / self.nb_epochs))
        print("Current weight decay:", self.optimizer.param_groups[0]["weight_decay"] )

class MomentumScheduler(Callback):
    """
    cosine scheduler for the momentum of the teacher model update
    if use_poly=True, use polynomial update instead of cosine
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
    """
    Dataset size scheduler
    We progressively increase the size of the dataset to help the arcface training
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
    """
    Update the metrics averages
    """
    def __init__(self, metrics, batch_size):
        self.metrics = metrics
        self.batch_size = batch_size

    def on_epoch_begin(self, epoch=None):
        for m in self.metrics: m.reset()

    def on_batch_end(self, batch=None):
        for m in self.metrics: m.update(self.batch_size)

#----------------------------------------------------------------------------