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
            # for dice computation, remove the background and flatten
            inputs = inputs.softmax(dim=1)

            if not all([i == j for i, j in zip(inputs.shape, targets.shape)]):
                # if this is not the case then gt is probably not already a one hot encoding
                targets_oh = torch.zeros(inputs.shape, device=inputs.device)
                targets_oh.scatter_(1, targets.long(), 1)
            else:
                targets_oh = targets

            # remove background
            inputs = inputs[:,1:]
            targets = targets_oh[:,1:]
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
            BCE = self.bce(inputs, targets.argmax(dim=1).long())

            # for dice computation, remove the background and flatten
            # inputs = inputs.softmax(dim=1)[:,1:].reshape(-1)
            # targets = targets[:,1:].reshape(-1)
            inputs = inputs.softmax(dim=1)

            if not all([i == j for i, j in zip(inputs.shape, targets.shape)]):
                # if this is not the case then gt is probably not already a one hot encoding
                targets_oh = torch.zeros(inputs.shape, device=inputs.device)
                targets_oh.scatter_(1, targets.long(), 1)
            else:
                targets_oh = targets

            # remove background
            inputs = inputs[:,1:]
            targets = targets_oh[:,1:]

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
            inputs = inputs.softmax(dim=1)

            if not all([i == j for i, j in zip(inputs.shape, targets.shape)]):
                # if this is not the case then gt is probably not already a one hot encoding
                targets_oh = torch.zeros(inputs.shape, device=inputs.device)
                targets_oh.scatter_(1, targets.long(), 1)
            else:
                targets_oh = targets

            # remove background
            inputs = inputs[:,1:]
            targets = targets_oh[:,1:]
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
# nnUNet metrics

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input, target):
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

# class RobustCrossEntropyLoss(nn.NLLLoss):
#     def __init__(self):
#         super(RobustCrossEntropyLoss, self).__init__()
#         # self.log_softmax = torch.nn.LogSoftmax(dim=-1)
#         self.criterion = torch.nn.NLLLoss()

#     def forward(self, input, target):
#         if len(target.shape) == len(input.shape):
#             assert target.shape[1] == 1
#             target = target[:, 0]
#         # self.val = self.ce(inputs, targets)
#         # yhat = torch.sigmoid(inputs).clamp(min=1e-3, max=1-1e-3)
#         # print("metric: inputs shape and target", inputs.shape, targets)
#         # yhat = torch.log(inputs.softmax(dim=1).clamp(min=1e-3, max=1-1e-3))
#         yhat = torch.log(input.type(torch.float32).softmax(dim=1))
#         # y = F.one_hot(y, num_classes=self.num_class).float()
#         # self.val = -y*((1-yhat) ** self.gamma) * torch.log(yhat) - (1-y) * (yhat ** self.gamma) * torch.log(1-yhat)
#         # self.val = self.criterion(yhat, targets.long())
#         return super().forward(yhat, target.long())
#         # return self.val


class DC_and_CE_loss(Metric):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None, name=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=lambda x: F.softmax(x, 1), **soft_dice_kwargs)
        
        self.name = name

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        
        if target.shape[1] != 1:
            target = target.argmax(dim=1).long().unsqueeze(dim=1)

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        
        self.val = result
        return self.val

#---------------------------------------------------------------------------
