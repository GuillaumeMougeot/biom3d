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
        
        iou = (intersection + smooth)/(union + smooth)
        
        self.val = 1 - iou if self.training else iou
        return self.val 

class MSE(Metric):
    def __init__(self, name=None):
        super(MSE, self).__init__()
        self.name = name

    def forward(self, inputs, targets):

        self.val = torch.nn.functional.mse_loss(inputs, targets, reduction='mean')
        return self.val

class CrossEntropy(Metric):
    def __init__(self, name=None):
        super(CrossEntropy, self).__init__()
        self.name = name
        self.ce = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs, targets):
        self.val = self.ce(inputs, targets)
        return self.val

#---------------------------------------------------------------------------
# Loss for self-supervised pre-training

class Triplet(Metric):
    """
    triplet loss.
    """
    def __init__(self, alpha=0.2, name=None):
        super(Triplet, self).__init__()
        self.alpha = alpha
        self.name = name 

    def forward(self, anchor, positive, negative, smooth=1):
        # normalize
        anc_normed = F.normalize(anchor)
        pos_normed = F.normalize(positive)
        neg_normed = F.normalize(negative)

        dist_pos = torch.sum(torch.square(anc_normed-pos_normed), dim=1)
        dist_neg = torch.sum(torch.square(anc_normed-neg_normed), dim=1)

        self.val = torch.mean(F.relu(dist_pos-dist_neg+self.alpha))
        return self.val

class TripletDiceBCE(Metric):
    """
    this class is just a placeholder for the triplet loss and the DiceBCE loss
    """
    def __init__(self, alpha=0.2, use_softmax=False, name=None):
        super(TripletDiceBCE, self).__init__()

        self.name = name 
        self.triplet_loss = Triplet(alpha=alpha, name='triplet_loss')
        self.dice_loss = DiceBCE(use_softmax=use_softmax, name='dice_loss')

    def update_val(self):
        """
        utils function to update self.val value
        """
        self.val = self.triplet_loss.val + self.dice_loss.val 

#---------------------------------------------------------------------------
# ArcFace losses
# stolen from: https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py 

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class ArcFace(Metric):
    """ 
    from: https://github.com/ydwen/opensphere/blob/main/model/head/arcface.py 
    reference: <Additive Angular Margin Loss for Deep Face Recognition>
    """
    def __init__(self, s=64., m=0.5, name=None):
        super(ArcFace, self).__init__()

        self.s = s
        self.m = m
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.name = name

    def forward(self, x, y):
        with torch.no_grad():
            theta_m = torch.acos(x.clamp(-1+1e-5, 1-1e-5))
            theta_m.scatter_(1, y.view(-1, 1), self.m, reduce='add')
            theta_m.clamp_(1e-5, 3.14159)
            d_theta = torch.cos(theta_m) - x

        logits = self.s * (x + d_theta)
        self.val = self.criterion(logits, y)

        return self.val

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = F.one_hot(label, num_classes=self.out_features)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

# class ArcFace(Metric):
#     """
#     CAREFUL: this module has parameters!!
#     """
#     def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, name=None):
#         super(ArcFace, self).__init__()
#         self.name = name
#         self.arc = ArcMarginProduct(in_features=in_features, out_features=out_features, s=s, m=m, easy_margin=easy_margin)
#         self.criterion = torch.nn.CrossEntropyLoss()
        
#     def forward(self, input, label):
#         self.val = self.criterion(self.arc(input, label), label)
#         return self.val

class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'

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
