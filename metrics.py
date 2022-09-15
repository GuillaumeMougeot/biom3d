#---------------------------------------------------------------------------
# Metrics/losses for segmentation
#---------------------------------------------------------------------------

import torch 
from torch import nn 
import torch.nn.functional as F
import numpy as np

#---------------------------------------------------------------------------
# Metrics base class

class Metric(nn.Module):
    """
    base class for metric to store its value and average and name.
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
    def __init__(self, use_softmax=False, name=None):
        super(Dice, self).__init__()
        self.name = name
        self.use_softmax = use_softmax # if use softmax then remove bg

    def forward(self, inputs, targets, smooth=1):
        if self.use_softmax:
            inputs = inputs.softmax(dim=1)[:,1:]
            targets = targets[:,1:]
        else:
            inputs = inputs.sigmoid()

        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        self.val = 1 - dice if self.training else dice
        return self.val 
    
class DiceBCE(Metric):
    """
    num_classes should be > 1 only if softmax use is intended!
    """
    def __init__(self, use_softmax=False, name=None):
        super(DiceBCE, self).__init__()
        self.use_softmax = use_softmax # if use softmax then remove bg for dice computation
        self.name = name
        self.bce = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.use_softmax:
            targets_bce = targets.argmax(dim=1).long()
            BCE = self.bce(inputs, targets_bce)

            # for dice computation, remove the background and flatten
            inputs = inputs.softmax(dim=1)[:,1:].reshape(-1)
            targets = targets[:,1:].reshape(-1)

        else:
            # keep the background and flatten
            inputs = inputs.reshape(-1)
            targets = targets.reshape(-1)
            BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
            inputs = inputs.sigmoid()


        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        Dice_BCE = BCE + dice_loss
        
        self.val = Dice_BCE
        return self.val 
    
class IoU(Metric):
    def __init__(self, use_softmax=False, name=None):
        super(IoU, self).__init__()
        self.use_softmax = use_softmax # if use softmax then remove bg
        self.name = name 

    def forward(self, inputs, targets, smooth=1):
        if self.use_softmax:
            inputs = inputs.softmax(dim=1)[:,1:]
            targets = targets[:,1:]
        else:
            inputs = inputs.sigmoid()

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)       
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
        
        self.val = 1 - IoU if self.training else IoU
        return self.val 

#---------------------------------------------------------------------------
# Metric adaptation for deep supervision

class DeepMetric(Metric):
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
        # inputs must be a list of network output
        # they are here all compared to the targets
        # the last inputs is supposed to be the final one
        self.val = 0
        for i in range(len(inputs)):
            if self.alphas[i]!=0:
                self.val += self.metric(inputs[i], targets)*self.alphas[i]
        return self.val

#---------------------------------------------------------------------------
