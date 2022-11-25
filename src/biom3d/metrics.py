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
        # if self.num_classes>1:
        #     self.multi = torch.arange(1,self.num_classes+1).view(1,-1,1,1,1).float()
        #     self.multi = self.multi.cuda()
        #     self.multi.requires_grad = False
        self.name = name
        self.bce = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.use_softmax:
            # targets_bce = targets.softmax(dim=1).sum(dim=1).long()
            # targets_bce = targets
            # targets_bce = targets_bce.sum(dim=1).long()
            # print(targets_bce.shape)
            targets_bce = targets.argmax(dim=1).long()
        
            BCE = self.bce(inputs, targets_bce)

            # for dice computation, remove the background and flatten
            inputs = inputs.softmax(dim=1)[:,1:].reshape(-1)
            targets = targets[:,1:].reshape(-1)

        else:
            # inputs = torch.sigmoid(inputs)    
            # targets_bce = torch.squeeze(targets)  

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
        # TMP: if there are two channels consider only the second one, 
        # the first one is supposed to be the BG
        # if inputs.shape[1]>1:
        #     inputs = inputs[:,1:,...]
        #     targets = targets[:,1:,...]
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # if self.multi:
        #     # remove the background and flatten
        #     inputs = inputs.softmax(dim=1)[:,1:,:,:,:].reshape(-1)
        #     targets = targets[:,1:,:,:,:].reshape(-1)
        # else:
            # keep the background and flatten
        if self.use_softmax:
            inputs = inputs.softmax(dim=1)[:,1:]
            targets = targets[:,1:]
        else:
            inputs = inputs.sigmoid()

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)       
        
        #flatten label and prediction tensors
        # inputs = inputs[:,1:,:,:,:].reshape(-1)
        # targets = targets[:,1:,:,:,:].reshape(-1)
        
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

# class CrossEntropy(Metric):
#     def __init__(self, name=None):
#         super(CrossEntropy, self).__init__()
#         self.name = name
#         # self.log_softmax = torch.nn.LogSoftmax(dim=-1)
#         self.criterion = torch.nn.NLLLoss()

#     def forward(self, inputs, targets):
#         # self.val = self.ce(inputs, targets)
#         # yhat = torch.sigmoid(inputs).clamp(min=1e-3, max=1-1e-3)
#         # print("metric: inputs shape and target", inputs.shape, targets)
#         # yhat = torch.log(inputs.softmax(dim=1).clamp(min=1e-3, max=1-1e-3))
#         yhat = torch.log(inputs.softmax(dim=1).type(torch.float32))
#         # y = F.one_hot(y, num_classes=self.num_class).float()
#         # self.val = -y*((1-yhat) ** self.gamma) * torch.log(yhat) - (1-y) * (yhat ** self.gamma) * torch.log(1-yhat)
#         self.val = self.criterion(yhat, targets)
#         return self.val

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
        # anc_normed = anchor
        # pos_normed = positive
        # neg_normed = negative

        dist_pos = torch.sum(torch.square(anc_normed-pos_normed), dim=1)
        dist_neg = torch.sum(torch.square(anc_normed-neg_normed), dim=1)

        # print("dist_pos",dist_pos)
        # print("dist_neg",dist_neg)
        # print("dist_pos", dist_pos)
        # print("dist_neg", dist_neg)
        # print("normed anchor", torch.norm(anc_normed, dim=1))
        # print("normed neg", torch.norm(negative, dim=1))
        # print("normed pos", torch.norm(positive, dim=1))
        # print("dim anc", anc_normed.shape)
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


# class ArcFace(Metric):
#     """ 
#     from: https://github.com/ydwen/opensphere/blob/main/model/head/arcface.py 
#     reference: <Additive Angular Margin Loss for Deep Face Recognition>
#     """
#     def __init__(self, feat_dim, num_class, s=64., m=0.5, name=None):
#         super(ArcFace, self).__init__()
#         self.feat_dim = feat_dim
#         self.num_class = num_class
#         self.s = s
#         self.m = m
#         self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
#         nn.init.xavier_normal_(self.w)

#         self.name = name
#         self.gamma = 0.2
#         self.log_softmax = torch.nn.LogSoftmax(dim=-1)
#         self.criterion = torch.nn.NLLLoss()

#     def set_num_classes(self, num_classes):
#         """
#         edit the number of classes, reset the loss trainable weights.
#         """
#         self.num_class = num_classes
#         self.w = nn.Parameter(torch.Tensor(self.feat_dim, self.num_class))
#         nn.init.xavier_normal_(self.w)

#     def forward(self, x, y):
#         with torch.no_grad():
#             self.w.data = F.normalize(self.w.data, dim=0)

#         cos_theta = F.normalize(x, dim=1).mm(self.w)
#         with torch.no_grad():
#             theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
#             theta_m.scatter_(1, y.view(-1, 1), self.m, reduce='add')
#             theta_m.clamp_(1e-5, 3.14159)
#             d_theta = torch.cos(theta_m) - cos_theta

#         logits = self.s * (cos_theta + d_theta)
#         self.val = F.cross_entropy(logits, y)

#         # yhat = torch.sigmoid(logits).clamp(min=1e-3, max=1-1e-3)
#         # yhat = torch.log(logits.softmax(dim=1)+1e-3)
#         # y = F.one_hot(y, num_classes=self.num_class).float()
#         # self.val = -y*((1-yhat) ** self.gamma) * torch.log(yhat) - (1-y) * (yhat ** self.gamma) * torch.log(1-yhat)
#         # self.val = self.criterion(yhat, y)

#         return self.val


# class ArcFace(Metric):
#     """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
#     """
#     def __init__(self, num_classes, s=30.0, margin=0.5, easy_margin=False, name=None):
#         super(ArcFace, self).__init__()
#         self.num_classes=num_classes
#         self.scale = s
#         self.cos_m = math.cos(margin)
#         self.sin_m = math.sin(margin)
#         self.theta = math.cos(math.pi - margin)
#         self.sinmm = math.sin(math.pi - margin) * margin
#         self.easy_margin = easy_margin

#         self.name = name
#         self.criterion = torch.nn.CrossEntropyLoss()


#     def forward(self, logits: torch.Tensor, labels: torch.Tensor):
#         # index = labels
#         # labels = F.one_hot(labels, num_classes=self.num_classes)
#         # # index = torch.where(labels != 0)[0] # original

#         # target_logit = logits[index, labels[index].view(-1)]


#         # sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
#         # cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
#         # if self.easy_margin:
#         #     final_target_logit = torch.where(
#         #         target_logit > 0, cos_theta_m, target_logit)
#         # else:
#         #     final_target_logit = torch.where(
#         #         target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

#         # logits[index, labels[index].view(-1)] = final_target_logit
#         # logits = logits * self.scale

#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         sine = torch.sqrt((1.0 - torch.pow(logits, 2)).clamp(0, 1))
#         phi = logits * self.cos_m - sine * self.sin_m
#         if self.easy_margin:
#             phi = torch.where(logits > 0, phi, logits)
#         else:
#             phi = torch.where(logits > self.theta, phi, logits - self.sinmm)
#         # --------------------------- convert label to one-hot ---------------------------
#         one_hot = F.one_hot(labels, num_classes=self.num_classes)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + ((1.0 - one_hot) * logits)  # you can use torch.where if your torch.__version__ is 0.4
#         output *= self.scale

#         self.val = self.criterion(output, labels)
#         return self.val


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

# class FocalLoss(Metric):

#     def __init__(self, gamma=0, eps=1e-7, name=None):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.eps = eps
#         self.ce = torch.nn.CrossEntropyLoss()
#         self.name = name

#     def forward(self, input, target):
#         logp = self.ce(input, target)
#         p = torch.exp(-logp)
#         loss = (1 - p) ** self.gamma * logp
#         self.val = loss.mean()
#         return self.val



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class FocalLoss(Metric):
    """ 
    from: https://github.com/ydwen/opensphere/blob/main/model/head/arcface.py 
    reference: <Additive Angular Margin Loss for Deep Face Recognition>
    """
    def __init__(self, feat_dim, num_class, name=None):
        super(FocalLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        # self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        # nn.init.xavier_normal_(self.w)

        self.w = nn.Linear(self.feat_dim, self.num_class, bias=False)
        self.apply(self._init_weights)

        self.name = name
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)

    def set_num_classes(self, num_classes):
        """
        edit the number of classes, reset the loss trainable weights.
        """
        self.num_class = num_classes
        # self.w = nn.Parameter(torch.Tensor(self.feat_dim, self.num_class))
        # nn.init.xavier_normal_(self.w)

        self.w = nn.Linear(self.feat_dim, self.num_class, bias=False)
        self.apply(self._init_weights)

    def forward(self, x, y):
        # with torch.no_grad():
        #     self.w.data = F.normalize(self.w.data, dim=0)
        # logits = F.normalize(x, dim=1).mm(self.w)
        # logits = x.mm(self.w)
        logits = self.w(F.normalize(x, dim=1))
        self.val = self.criterion(logits, y)

        return self.val

#---------------------------------------------------------------------------
# Adversarial metrics

from biom3d.models.encoder_efficientnet3d import EfficientNet3D
from biom3d.models.encoder_vgg import VGGEncoder, SmallEncoderBlock, EncoderBlock

class GlobalAvgPool3d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool3d, self).__init__()

    def forward(self,x):
        out = x.mean(dim=(-3,-2,-1))
        return out

class Adversarial(Metric):
    """
    this loss contains a discriminator model (i.e. a CNN) to sort
    real from fake images. 
    """
    def __init__(self, lr=1e-3, weight_decay=3e-5, name=None):
        super(Adversarial, self).__init__()
        self.name = name
        # self.cnn = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2}, in_channels=1)
        self.cnn = VGGEncoder(EncoderBlock, [4,4,4], use_emb=True, first_stride=[2,2,2])
        print(self.cnn)
        # self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.criterion = nn.CrossEntropyLoss()
        self.optim = self.optim = torch.optim.Adam(
            [{'params': self.cnn.parameters()}], 
            lr=1e-3, weight_decay=1e-4)
        self.warmup = 10 # number of epoch warmup
        
    def requires_grad(self, bool):
        for p in self.cnn.parameters():
            p.requires_grad = bool

    def forward(self, fake, real):
        fake = (torch.sigmoid(fake) > 0.5).float()
        if self.training: # discriminator update
            self.requires_grad(True)
            self.cnn.zero_grad()
            pred_real = self.cnn(real)
            pred_fake = self.cnn(fake.detach())
            errD = (torch.mean(torch.nn.ReLU()(1.0 - (pred_real - torch.mean(pred_fake)))) + torch.mean(torch.nn.ReLU()(1.0 + (pred_fake - torch.mean(pred_real)))))/2
            self.val = errD
            errD.backward()
            self.optim.step()
        else:
            self.requires_grad(False)
            pred_real = self.cnn(real)
            pred_fake = self.cnn(fake)
            errG = (torch.mean(torch.nn.ReLU()(1.0 + (pred_real - torch.mean(pred_fake)))) + torch.mean(torch.nn.ReLU()(1.0 - (pred_fake - torch.mean(pred_real)))))/2
            self.val = errG
            return errG
        # relativistic average BCE loss
        # defined in https://arxiv.org/abs/1807.00734
        # f1 = lambda x: self.criterion(x, torch.ones(x.shape[0], dtype=torch.long).cuda())
        # f2 = lambda x: self.criterion(x, torch.zeros(x.shape[0], dtype=torch.long).cuda())

        # rf = pred_real - pred_fake.mean()
        # fr = pred_fake - pred_real.mean()

        # disc_loss = (f1(rf)+f2(fr)).mean()
        # if self.training:
        #     disc_loss
        # gen_loss = (f2(rf)+f1(fr)).mean()

        # if self.training: # discriminator loss
        #     self.val = disc_loss
        # else:
        #     self.val = gen_loss
        # return disc_loss, gen_loss
        # targets = torch.cat([
        #     torch.ones(pred_real.shape[0]),
        #     torch.zeros(pred_fake.shape[0])], dim=0)
        # targets = targets.long()
        # if torch.cuda.is_available():
        #     targets = targets.cuda()
        
        # inputs = torch.cat([pred_real, pred_fake], dim=0)
    
        # self.val = self.criterion(inputs, targets)
        # return self.val

def test_adversarial():
    """
    to test the adversarial loss we try here to classify 3d images of nucleus from 3d images of pancreas 
    """
    from biom3d.datasets.semseg import SemSeg3D
    from time import time
    loss_fn = Adversarial().cuda()
    data_nucleus = SemSeg3D(
        img_dir="data/nucleus/images_resampled_sophie",
        msk_dir="data/nucleus/masks_resampled_sophie",
        folds_csv="data/nucleus/folds_sophie.csv",
        fold=0,
        train=True,
        use_aug=False,
    )
    data_pancreas = SemSeg3D(
        img_dir="data/pancreas/tif_train_img",
        msk_dir="data/pancreas/tif_train_img_labels",
        folds_csv="data/pancreas/folds_small.csv",
        fold=0,
        train=True,
        use_aug=False,
    )

    optim = torch.optim.Adam(
            [{'params': loss_fn.parameters()}], 
            lr=1e-2, weight_decay=1e-4)
    batch_size = 4
    for epoch in range(100):
        for i in range(0, min(len(data_pancreas), len(data_nucleus))-batch_size, batch_size):
            start_time = time()
            batch_nucl = []
            batch_panc = []
            for j in range(batch_size):
                batch_nucl += [data_nucleus.__getitem__(i+j)[1]]
                batch_panc += [data_pancreas.__getitem__(i+j)[1]]
            msk_nucl = torch.stack(batch_nucl).cuda()
            msk_panc = torch.stack(batch_panc).cuda()
            loss = loss_fn(msk_nucl, msk_panc)
            print("loss",loss)
            optim.zero_grad() # set gradient to zero, why is that needed?
            loss.backward() # compute gradient
            optim.step() # apply gradient 
            print(time()-start_time)
        print("end of epoch number ", epoch, "="*100)

if __name__=='__main__':
    test_adversarial()

class DiceAdversarial(Metric):
    """
    this loss contains a discriminator model (i.e. a CNN) to sort
    real from fake images. 
    """
    def __init__(self, lr=1e-3, name=None):
        super(DiceAdversarial, self).__init__()
        self.name = name
        name_adv = "adv" if name==None else name+"_adv"
        self.adv = Adversarial(lr=lr,name=name_adv)
        name_dice = "dice" if name==None else name+"_dice"
        self.dice = DiceBCE(name=name_dice)
        self.warmup = 0

    def forward(self, fake, real, pred=None):
        if self.training or pred==None:
            self.adv.train()
            self.adv(fake, real)
            self.val = self.adv.val
        else:
            self.adv.eval()
            self.val = self.adv(fake, real) + self.dice(pred, real)
            return self.val


#---------------------------------------------------------------------------
# Metrics for cotraining

class CoTrain(Metric):
    """
    CAREFUL: this module has parameters!!
    """
    def __init__(self, milestones, alphas, name=None):
        super(CoTrain, self).__init__()
        self.name = name
        self.arc = ArcFace(name="arcface")
        self.seg = DeepMetric(
            metric=DiceBCE(name="seg_loss"),
            milestones=milestones,
            alphas=alphas)
        
    def forward(self, input_enc, input_dec, mask, label, epoch):
        self.val = 0.05*self.arc(input_enc, label) + self.seg(input_dec, mask, epoch)
        return self.val

class DenoiSeg(Metric):
    def __init__(self, name=None):
        super(DenoiSeg, self).__init__()
        self.name=name
        self.denois = MSE(name+"_mse")
        self.seg = DiceBCE(name+"_dice_bce")
        self.save = 0
        self.alpha = 0.1

    def forward(self, inputs, gt_seg, gt_denois):
        inputs_seg, inputs_denois = torch.split(tensor=inputs, split_size_or_sections=1, dim=1)
        self.val = self.alpha * self.denois(inputs_denois, gt_denois)
        if gt_seg.sum()!=0:
            seg_loss = self.seg(inputs_seg, gt_seg)
            self.val += seg_loss
            self.save = seg_loss.detach()
        else:
            self.val += self.save
        return self.val 

#---------------------------------------------------------------------------
# DINO metric from: https://github.com/facebookresearch/dino


class Entropy(Metric):
    """
    Teacher entropy to see if there is no collapse
    """
    def __init__(self, 
        out_dim,
        nbof_global_crops,
        nbof_local_crops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        name=None):

        super(Entropy, self).__init__()
        self.name = name
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = nbof_global_crops+nbof_local_crops
        self.nbof_global_crops = nbof_global_crops
        self.register_buffer("center", torch.zeros(1, out_dim))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, teacher_output, student_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.nbof_global_crops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            loss = torch.sum(-q * torch.log(q), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
        self.update_center(teacher_output)
        total_loss /= n_loss_terms
        self.val = total_loss
        return self.val

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        # batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        batch_center = batch_center / (len(teacher_output))

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
   


class KLDiv(Metric):
    """
    Kullback-Leibler divergence to see if there is no collapse
    """
    def __init__(self, 
        out_dim,
        nbof_global_crops,
        nbof_local_crops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        name=None):

        super(KLDiv, self).__init__()
        self.name = name
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = nbof_global_crops+nbof_local_crops
        self.nbof_global_crops = nbof_global_crops
        self.register_buffer("center", torch.zeros(1, out_dim))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, teacher_output, student_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.nbof_global_crops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                # loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                loss = torch.sum(q * torch.log(q/F.softmax(student_out[v], dim=-1)), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        self.update_center(teacher_output)
        total_loss /= n_loss_terms
        self.val = total_loss
        return self.val

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        # batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        batch_center = batch_center / (len(teacher_output))

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        


class DINOLoss(Metric):
    def __init__(self,
            out_dim,
            nbof_global_crops,
            nbof_local_crops,
            warmup_teacher_temp,
            teacher_temp,
            warmup_teacher_temp_epochs,
            nepochs,
            student_temp=0.1,
            center_momentum=0.9,
            name=None,
        ):
        super(DINOLoss, self).__init__()
        self.name = name
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = nbof_global_crops+nbof_local_crops
        self.nbof_global_crops = nbof_global_crops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, teacher_output, student_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.nbof_global_crops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        self.val = total_loss
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        # batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        batch_center = batch_center / (len(teacher_output))

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

#---------------------------------------------------------------------------
# HRNet loss

class HRDiceBCE(Metric):
    """
    this loss is basically the DiceBCE loss applied to a list 
    num_classes should be > 1 only if softmax use is intended!
    """
    def __init__(self, use_softmax=False, weights=[0.4,1], name=None):
        super(HRDiceBCE, self).__init__()
        self.use_softmax = use_softmax # if use softmax then remove bg for dice computation
        self.weights = weights 
        self.name = name
        self.bce = torch.nn.CrossEntropyLoss(reduction='mean')

    def _forward(self, inputs, targets, smooth=1):
        ph, pw, pd = inputs.size(2), inputs.size(3), inputs.size(4)
        h, w, d = targets.size(2), targets.size(3), targets.size(4)
        if ph != h or pw != w or pd != d:
            inputs = F.interpolate(input=inputs, size=(
                h, w, d), mode='trilinear')

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
        return BCE + dice_loss

    def forward(self, inputs, targets, smooth=1):
        self.val = sum([w * self._forward(x, targets) for (w,x) in zip(self.weights, inputs)])
        return self.val 


#---------------------------------------------------------------------------
# Metric adaptation for deep supervision

# class DeepMetric(Metric):
#     def __init__(self,
#         metric,
#         milestones, # list of epoch where the alphas are applied
#         alphas, # list of alpha, must be of length len(milestones)+1, must be between 0 and 1, preferably decreasing
#         name=None,
#         **kwargs):
#         super(DeepMetric, self).__init__()
#         self.metric = metric(**kwargs)
#         self.name = name 
#         self.milestones = milestones
#         self.idx_milestone = 0
#         self.alphas = alphas 
#         self.display_interruption = True # just for verbose
    
#     def forward(self, inputs, targets, epoch=None):
#         # inputs must be a list of network output
#         # they are here all compared to the targets
#         # the last inputs is supposed to be the final one

#         metric_deep = 0 # values of deep metrics
#         for i in range(len(inputs)-1):
#             metric_deep += self.metric(inputs[i], targets)
#         metric_last = self.metric(inputs[-1], targets)

#         if len(inputs) > 1: # deepsupervision is on
#             if self.idx_milestone < len(self.milestones) and epoch >= self.milestones[self.idx_milestone]:
#                 # update the milestone
#                 self.idx_milestone += 1
#             alpha = self.alphas[self.idx_milestone]
#         else:
#             alpha = 0
#         if alpha == 0 and self.display_interruption: 
#             print("Deepsupervision has been interrupted.")
#             self.display_interruption = False 
#         self.val = metric_last*(1-alpha) + metric_deep*alpha
#         return self.val 

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
