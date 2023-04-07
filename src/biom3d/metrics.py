#---------------------------------------------------------------------------
# Metrics/losses 
# Mostly for segmentation
#---------------------------------------------------------------------------

import torch 
from torch import nn 
import torch.nn.functional as F
import numpy as np

#---------------------------------------------------------------------------
# Metrics base class

class Metric(nn.Module):
    """Abstract class for all metrics. Built to store the metric value, average, and name. In biom3d structure, metrics must have a `name`, a `self.reset` method that reset the stored values to zero and a `self.update` method that update the average value with the value stored in the `val` argument.
    
    To create a new metric, sub-class the Metric class, override the `self.__init__` and the `self.forward` methods. In the `self.__init__` set the `name` argument value and in `self.forward` set `val` argument value with the current value of your metric. 

    Parameters
    ----------
    name : str, default=None
        Name of the metric. Mainly for display purpose.
    """
    def __init__(self, name=None):
        super(Metric, self).__init__()
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def forward(self, preds, trues):
        pass

    def update(self, n=1):
        with torch.no_grad():
            self.sum += self.val * n
            self.count += n
            self.avg = self.sum / self.count
    
    def str(self):
        return str(self.val)
    
    def __str__(self):
        to_print = self.avg if self.avg!=0 else self.val
        return "{} {:.3f}".format(self.name, to_print)

#---------------------------------------------------------------------------
# Metrics/losses for semantic segmentation

class Dice(Metric):
    """Dice score computation. 

    Parameters
    ----------
    use_softmax : bool, default=False
        Whether the logit include the background for softmax computation. If True, then the background channel will be removed.
    dim : tuple, default=()
        The dimensions to which the score will be computed. If the default value is used, then the score will be computed for all dimension which might cause a class imbalance. To balance your channel, prefer setting `dim=(2,3,4)` if you deal with 3D images and `dim=(2,3)` for 2D images. 
    name : str, default=None
        Name of the metric. Mainly for display purpose.
    """
    def __init__(self, use_softmax=False, dim=(), name=None):
        super(Dice, self).__init__()
        self.name = name
        self.dim = dim
        self.use_softmax = use_softmax # if use softmax then remove bg

    def forward(self, inputs, targets, smooth=1):
        if self.use_softmax:
            # inputs = inputs.softmax(dim=1)
            # for dice computation, remove the background and flatten
            inputs = inputs.softmax(dim=1)[:,1:]
            targets = targets[:,1:]
        else:
            inputs = inputs.sigmoid()

        #flatten label and prediction tensors
        # inputs = inputs.reshape(-1)
        # targets = targets.reshape(-1)

        intersection = (inputs * targets).sum(dim=self.dim)                            
        dice = (2.*intersection + smooth)/(inputs.sum(dim=self.dim) + targets.sum(dim=self.dim) + smooth)  

        self.val = 1 - dice.mean() if self.training else dice.mean()
        return self.val  
    
class DiceBCE(Metric):
    """Dice-binary cross-entropy score computation. This metric can be considered as a loss function. 

    Parameters
    ----------
    use_softmax : bool, default=False
        Whether the logit include the background for softmax computation. If True, then the background channel will be removed.
    dim : tuple, default=()
        The dimensions to which the score will be computed. If the default value is used, then the score will be computed for all dimension which might cause a class imbalance. To balance your channel, prefer setting `dim=(2,3,4)` if you deal with 3D images and `dim=(2,3)` for 2D images. 
    name : str, default=None
        Name of the metric. Mainly for display purpose.
    """
    def __init__(self, use_softmax=False, dim=(), name=None):
        super(DiceBCE, self).__init__()
        self.use_softmax = use_softmax # if use softmax then remove bg for dice computation
        self.name = name
        self.dim = dim # axis defined for the dice score 
        self.bce = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.use_softmax:
            targets_bce = targets.argmax(dim=1).long()
        
            BCE = self.bce(inputs, targets_bce)

            # for dice computation, remove the background and flatten
            # inputs = inputs.softmax(dim=1)[:,1:].reshape(-1)
            # targets = targets[:,1:].reshape(-1)
            inputs = inputs.softmax(dim=1)[:,1:]
            targets = targets[:,1:]

        else:
            # keep the background and flatten
            # inputs = inputs.reshape(-1)
            # targets = targets.reshape(-1)
            BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
            inputs = inputs.sigmoid()


        intersection = (inputs * targets).sum(dim=self.dim)                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum(dim=self.dim) + targets.sum(dim=self.dim) + smooth)
        Dice_BCE = BCE + dice_loss.mean()
        
        self.val = Dice_BCE
        return self.val 
    
class IoU(Metric):
    """Intersection over union score computation. 

    Parameters
    ----------
    use_softmax : bool, default=False
        Whether the logit include the background for softmax computation. If True, then the background channel will be removed.
    dim : tuple, default=()
        The dimensions to which the score will be computed. If the default value is used, then the score will be computed for all dimension which might cause a class imbalance. To balance your channel, prefer setting `dim=(2,3,4)` if you deal with 3D images and `dim=(2,3)` for 2D images. 
    name : str, default=None
        Name of the metric. Mainly for display purpose.
    """
    def __init__(self, use_softmax=False, dim=(), name=None):
        super(IoU, self).__init__()
        self.use_softmax = use_softmax # if use softmax then remove bg
        self.dim = dim
        self.name = name 

    def forward(self, inputs, targets, smooth=1):
        if self.use_softmax:
            # inputs = inputs.softmax(dim=1)
            inputs = inputs.softmax(dim=1)[:,1:]
            targets = targets[:,1:]
        else:
            inputs = inputs.sigmoid()

        # inputs = inputs.reshape(-1)
        # targets = targets.reshape(-1)       
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum(dim=self.dim)
        total = (inputs + targets).sum(dim=self.dim)
        union = total - intersection 
        
        iou = (intersection + smooth)/(union + smooth)
        
        self.val = 1 - iou.mean() if self.training else iou.mean()
        return self.val 

class MSE(Metric):
    """Mean Square Error loss.

    Parameters
    ----------
    name : str, default=None
        Name of the metric. Mainly for display purpose.
    """
    def __init__(self, name=None):
        super(MSE, self).__init__()
        self.name = name

    def forward(self, inputs, targets):

        self.val = torch.nn.functional.mse_loss(inputs, targets, reduction='mean')
        return self.val

class CrossEntropy(Metric):
    """Cross entropy loss.

    Parameters
    ----------
    name : str, default=None
        Name of the metric. Mainly for display purpose.
    """
    def __init__(self, name=None):
        super(CrossEntropy, self).__init__()
        self.name = name
        self.ce = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs, targets):
        self.val = self.ce(inputs, targets)
        return self.val

#---------------------------------------------------------------------------
# Metric adaptation for deep supervision

class DeepMetric(Metric):
    """Deep supervision: a metric applied to several levels of the U-Net model. During the forward pass, this metric supposes that the inputs are a list of torch.Tensor that will be individually compared to the target.

    Parameters
    ----------
    metric : biom3d.metrics.Metric
        A metric that will be applied to each level of the U-Net model.
    alphas : list of float
        A list of coefficient applied to each feature map that starts with the deepest one. It must have a len of 6.
    name : str, default=None
        Name of the metric. Mainly for display purpose.
    metric_kwargs : dict, default={}
        A python dictionary that contains the keyword arguments required to define the metric.
    """
    def __init__(self,
        metric,
        alphas, # list of coefficient applied to each feature map, starts with the deepest one, must have a len of 6
        name=None,
        metric_kwargs={}):
        super(DeepMetric, self).__init__()
        self.metric = metric(**metric_kwargs)
        self.name = name 
        self.alphas = alphas 
    
    def forward(self, inputs, targets):
        """Forward pass.

        Parameters
        ----------
        inputs : list of torch.Tensor
            A list of batch of torch.Tensor that will be passed to the metric.
        targets : torch.Tensor
            A batch of torch.Tensor that will be compared to the inputs.
        """
        # inputs must be a list of network output
        # they are here all compared to the targets
        # the last inputs is supposed to be the final one
        self.val = 0
        for i in range(len(inputs)):
            if self.alphas[i]!=0:
                self.val += self.metric(inputs[i], targets)*self.alphas[i]
        return self.val

#---------------------------------------------------------------------------
