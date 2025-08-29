"""
Metrics/losses.

Mostly for segmentation.
"""

import torch 
from torch import Type, nn 
import torch.nn.functional as F
import numpy as np

from abc import abstractmethod
from typing import Any, Callable, Literal, Optional

#---------------------------------------------------------------------------
# Metrics base class
class Metric(nn.Module):
    """
    Abstract base class for all metrics.

    Designed to store and update metric values such as the current value, average, sum, and count.
    In the biom3d structure, all metrics must implement the following:
    - a `name` attribute for identification,
    - a `reset()` method to clear internal states,
    - a `forward()` method to compute the metric given predictions and targets,
    - an `update()` method to accumulate statistics.

    To define a new metric:
    1. Inherit from `Metric`.
    2. Set the `name` attribute in `__init__`.
    3. Implement `forward()` to compute the metric and assign to `self.val`.

    :ivar str name: Name of the metric (used in logging/display).
    :ivar float val: Current value of the metric for the last batch.
    :ivar float avg: Running average of the metric.
    :ivar float sum: Cumulative sum of the metric across all updates.
    :ivar int count: Number of updates applied (used to compute average).
    """

    def __init__(self, name:Optional[str]=None):
        """
        Initialize the base Metric class.

        Parameters
        ----------
        name : str, optional
            Name of the metric, for display/logging purposes.
        """
        super(Metric, self).__init__()
        self.name = name
        self.reset()

    def reset(self) -> None:
        """
        Reset internal metric statistics.

        Returns
        -------
        None
        """
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    @abstractmethod
    def forward(self, preds: torch.Tensor, trues: torch.Tensor) -> None:
        """
        Compute the metric value for the given predictions and targets.

        Must be implemented by subclass.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions.
        trues : torch.Tensor
            Ground truth targets.

        Returns
        -------
        None
        """
        pass

    def update(self, n: int = 1) -> None:
        """
        Update metric statistics with the current value.

        Parameters
        ----------
        n : int, default=1
            Number of samples in the current batch.

        Returns
        -------
        None
        """
        with torch.no_grad():
            self.sum += self.val * n
            self.count += n
            self.avg = self.sum / self.count
    
    def str(self) -> str:
        """
        Return string representation of the current value.

        Returns
        -------
        str
            String representation of the current .
        """
        return str(self.val)
    
    def __str__(self):
        """
        Return formatted string representation of the average or current value.

        Returns
        -------
        str
            Formatted metric string: "<name> <value>".
        """
        to_print = self.avg if self.avg!=0 else self.val
        return "{} {:.3f}".format(self.name, to_print)

#---------------------------------------------------------------------------
# Metrics/losses for semantic segmentation
class Dice(Metric):
    """
    Dice score computation metric.

    Computes the Dice coefficient between predictions and targets.
    Supports binary and multi-class segmentation with optional softmax normalization.
    Background channel can be automatically removed for multi-class setups.

    :ivar str name: Name of the metric (used in logs).
    :ivar bool use_softmax: Whether to apply softmax before Dice computation.
    :ivar tuple[int] dim: Dimensions over which Dice is computed (e.g., (2, 3) or (2, 3, 4)).
    """

    def __init__(self, use_softmax:bool=False, dim:tuple[int]=(), name:str=None):
        """
        Initialize the Dice metric.

        Parameters
        ----------
        use_softmax : bool, default=False
            Whether to apply softmax to model outputs. If True, background channel is removed.
        dim : tuple, default=()
            Dimensions along which Dice is computed. Use (2,3) for 2D or (2,3,4) for 3D images.
        name : str, optional
            Name of the metric for logging and display.
        """
        super(Dice, self).__init__()
        self.name = name
        self.dim = dim
        self.use_softmax = use_softmax # if use softmax then remove bg

    def forward(self, 
                inputs:torch.Tensor, 
                targets:torch.Tensor, 
                smooth:float=1.0,
                )->torch.Tensor:
        """
        Compute Dice coefficient between predictions and ground truth targets.

        Parameters
        ----------
        inputs : torch.Tensor
            Model outputs (logits or probabilities).
        targets : torch.Tensor
            Ground truth segmentation masks.
        smooth : float, default=1.0
            Smoothing constant to avoid division by zero.

        Returns
        -------
        self.val: torch.Tensor
            Dice score as a scalar tensor.
        """
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


        intersection = (inputs * targets).sum(dim=self.dim)                            
        dice = (2.*intersection + smooth)/(inputs.sum(dim=self.dim) + targets.sum(dim=self.dim) + smooth)  

        self.val = 1 - dice.mean() if self.training else dice.mean()
        return self.val  
    
class DiceBCE(Metric):
    """
    Dice + Binary Cross-Entropy (BCE) metric.

    This combined metric can also be used as a loss function. It computes the sum of the BCE loss 
    and the Dice loss, supporting both binary and multi-class segmentation with optional softmax activation. 
    For multi-class predictions, background is removed from Dice computation if `use_softmax` is True.

    :ivar str name: Name of the metric (used in logs).
    :ivar bool use_softmax: Whether to apply softmax and remove background for Dice computation.
    :ivar tuple[int] dim: Dimensions over which Dice is computed.
    :ivar torch.nn.CrossEntropyLoss bce: BCE loss function module.
    """

    def __init__(self, use_softmax:bool=False, dim:tuple[int]=(), name:Optional[str]=None):
        """
        Initialize the DiceBCE metric.

        Parameters
        ----------
        use_softmax : bool, default=False
            Whether to apply softmax to model outputs. If True, background is excluded from Dice.
        dim : tuple, default=()
            Dimensions along which Dice is computed (e.g., (2, 3) or (2, 3, 4)).
        name : str, optional
            Name of the metric for logging and display.
        """
        super(DiceBCE, self).__init__()
        self.use_softmax = use_softmax # if use softmax then remove bg for dice computation
        self.name = name
        self.dim = dim # axis defined for the dice score 
        self.bce = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, 
                inputs:torch.Tensor, 
                targets:torch.Tensor, 
                smooth:float=1.0,
                )->torch.Tensor:
        """
        Compute the combined Dice and BCE loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Model outputs (logits).
        targets : torch.Tensor
            Ground truth segmentation masks (can be one-hot or class indices).
        smooth : float, default=1.0
            Smoothing constant to avoid division by zero in Dice computation.

        Returns
        -------
        slef.val: torch.Tensor
            The Dice + BCE.
        """
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.use_softmax:
            BCE = self.bce(inputs, targets.argmax(dim=1).long())

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
    """
    Intersection over Union (IoU) score computation.

    This metric measures the overlap between the predicted and ground truth segmentation masks.
    Can be used as a metric or a loss function (when inverted during training). Supports both
    binary and multi-class segmentation tasks, with optional softmax activation and background removal.

    :ivar str name: Name of the metric.
    :ivar bool use_softmax: Whether to apply softmax (multi-class case).
    :ivar tuple dim: Dimensions along which the IoU is computed.
    """
    
    def __init__(self, use_softmax: bool = False, dim: tuple = (), name: Optional[str] = None):
        """
        Initialize the IoU metric.

        Parameters
        ----------
        use_softmax : bool, default=False
            Whether to apply softmax to model outputs. If True, background is excluded from IoU computation.
        dim : tuple, default=()
            Dimensions over which IoU is computed (e.g., (2, 3, 4) for 3D or (2, 3) for 2D).
        name : str, optional
            Name of the metric (for logging or display).
        """
        super(IoU, self).__init__()
        self.use_softmax = use_softmax # if use softmax then remove bg
        self.dim = dim
        self.name = name 

    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor, 
                smooth: float = 1.0,
                ) -> torch.Tensor:
        """
        Compute the IoU between predicted and target masks.

        Parameters
        ----------
        inputs : torch.Tensor
            Logits output by the model.
        targets : torch.Tensor
            Ground truth segmentation masks (either one-hot or class indices).
        smooth : float, default=1.0
            Smoothing term to avoid division by zero.

        Returns
        -------
        self.val: torch.Tensor
            Computed IoU score.
        """
        if self.use_softmax:
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
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum(dim=self.dim)
        total = (inputs + targets).sum(dim=self.dim)
        union = total - intersection 
        
        iou = (intersection + smooth)/(union + smooth)
        
        self.val = 1 - iou.mean() if self.training else iou.mean()
        return self.val 

class MSE(Metric):
    """
    Mean Squared Error (MSE) loss.

    This metric computes the average squared difference between predictions and targets.
    Commonly used as a regression loss, but can also serve as a metric in training.

    :ivar str name: Name of the metric (for logging or display).
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the MSE metric.

        Parameters
        ----------
        name : str, optional
            Name of the metric, for display or logging purposes.
        """

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error between predictions and targets.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted values.
        targets : torch.Tensor
            Ground truth values.

        Returns
        -------
        torch.Tensor
            Computed MSE loss.
        """
        self.val = torch.nn.functional.mse_loss(inputs, targets, reduction='mean')
        return self.val

class CrossEntropy(Metric):
    """
    Cross-entropy loss metric.

    This metric computes the average cross-entropy between predicted class scores and target class indices.
    Typically used for classification problems.

    :ivar str name: Name of the metric (used in logging or display).
    :ivar torch.nn.CrossEntropyLoss ce: Internal cross-entropy loss module.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the CrossEntropy metric.

        Parameters
        ----------
        name : str, optional
            Name of the metric, for display or logging purposes.
        """
        super(CrossEntropy, self).__init__()
        self.name = name
        self.ce = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-entropy loss between predictions and targets.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw class scores (logits) of shape (N, C, ...).
        targets : torch.Tensor
            Ground truth class indices of shape (N, ...).

        Returns
        -------
        torch.Tensor
            Computed cross-entropy loss.
        """
        self.val = self.ce(inputs, targets)
        return self.val

#---------------------------------------------------------------------------
# Metric adaptation for deep supervision
class DeepMetric(Metric):
    """
    Metric wrapper for deep supervision.

    Applies a given metric across multiple feature maps (from different decoder depths, e.g., in U-Net),
    combining them using a weighted sum of coefficients (`alphas`). Each feature map prediction is compared
    to the same ground truth.

    :ivar Metric metric: Base metric applied at each level.
    :ivar list[float] alphas: Weights associated with each level’s output.
    :ivar str name: Name of the metric.
    """
    
    def __init__(self,
                 metric: type[Metric],
                 alphas: list[float],
                 name: Optional[str] = None,
                 metric_kwargs: Optional[dict[str, Any]] = None):
        """
        Initialize the DeepMetric.

        Parameters
        ----------
        metric : type[Metric]
            A callable class or constructor of a `Metric` to apply at each level.
        alphas : list of float
            List of coefficients for each prediction level. Must match the number of inputs.
        name : str, optional
            Name of the metric (used for logging/display).
        metric_kwargs : dict, optional
            Additional keyword arguments for the base metric constructor.
        """
        super(DeepMetric, self).__init__()
        self.metric = metric(**metric_kwargs)
        self.name = name 
        self.alphas = alphas 
    
    def forward(self, inputs: list[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the deep supervision metric across multiple prediction levels.

        Parameters
        ----------
        inputs : list of torch.Tensor
            List of model outputs at various depths.
        targets : torch.Tensor
            Ground truth labels shared across all prediction levels.

        Returns
        -------
        self.val: torch.Tensor
            Weighted sum of metric values across levels.
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
def sum_tensor(inp:torch.Tensor, 
               axes:int|tuple[int]|list[int], 
               keepdim:bool=False,
               )->torch.Tensor:
    """
    Sum a tensor over specified axes.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor to reduce.
    axes : int or list/tuple of int
        Axes over which the tensor is summed.
    keepdim : bool, default=False
        Whether to retain reduced dimensions (with size 1).

    Returns
    -------
    inp: torch.Tensor
        Reduced tensor with summed values. No copy.
    """
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output: torch.Tensor,
                    gt: torch.Tensor,
                    axes: Optional[int| tuple[int]| list[int]] = None,
                    mask: Optional[torch.Tensor] = None,
                    square: bool = False,
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute true positives (TP), false positives (FP), false negatives (FN), and true negatives (TN) between network outputs and ground truth labels.

    Assumes the input `net_output` is in shape (B, C, ...) and `gt` is either a label map (B, ...) or
    one-hot encoded (B, C, ...). If not already in one-hot form, the function converts `gt`.

    Parameters
    ----------
    net_output : torch.Tensor
        Model prediction tensor with shape (B, C, H, W, (D)).
    gt : torch.Tensor
        Ground truth tensor. Can be label map (B, H, W, (D)) or one-hot encoded (B, C, H, W, (D)).
    axes : int or list/tuple of int, optional
        Axes to reduce over (e.g., spatial dimensions).
    mask : torch.Tensor, optional
        Optional binary mask of shape (B, 1, H, W, (D)) where 1 indicates valid pixels.
    square : bool, default=False
        Whether to square the TP/FP/FN/TN tensors before summation.

    Returns
    -------
    tp torch.Tensor
        True positives after reduction per class.
    fp torch.Tensor
        False positives after reduction per class.
    tn torch.Tensor
        True negatives after reduction per class.
    fn torch.Tensor
        False negatives after reduction per class.
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
    """
    Soft Dice loss implementation.

    This loss is useful for segmentation tasks, especially when dealing with class imbalance.
    It computes a soft version of the Dice coefficient, optionally over the batch dimension.
    
    Parameters
    ----------
    apply_nonlin : callable, optional
        Optional non-linearity to apply to the prediction (e.g., torch.softmax or torch.sigmoid).
    batch_dice : bool, default=False
        If True, compute Dice over the entire batch instead of per-sample.
    do_bg : bool, default=True
        If False, ignore background channel (assumed to be channel index 0).
    smooth : float, default=1.
        Smoothing factor added to numerator and denominator to avoid division by zero.
    """

    def __init__(self,
                 apply_nonlin: Optional[Callable] = None,
                 batch_dice: bool = False,
                 do_bg: bool = True,
                 smooth: float = 1.0):
        """
        Initialize the SoftDiceLoss.

        Parameters
        ----------
        apply_nonlin : callable, optional
            Non-linearity function to apply to the predictions (e.g., torch.sigmoid or torch.softmax).
            If None, no activation is applied.
        batch_dice : bool, default=False
            If True, computes Dice over the entire batch as a whole; otherwise computes per sample.
        do_bg : bool, default=True
            If False, excludes the background class (channel 0) from the Dice computation.
        smooth : float, default=1.0
            Smoothing factor added to numerator and denominator to avoid division by zero.
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the Soft Dice loss.

        Parameters
        ----------
        x : torch.Tensor
            Network predictions of shape (B, C, ...) where B is batch size and C number of classes.
        y : torch.Tensor
            Ground truth labels, either label maps or one-hot encoded tensors.
        loss_mask : torch.Tensor, optional
            Mask tensor with shape (B, 1, ...), where valid pixels are 1 and invalid are 0.

        Returns
        -------
        -dc: torch.Tensor
            Scalar loss value (negative mean Dice coefficient).
        """
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
    Compatibility wrapper for CrossEntropyLoss to handle target tensors with an extra singleton dimension and float dtype.

    This class modifies the target tensor shape and type to fit the expected input of
    nn.CrossEntropyLoss, which requires targets to be LongTensor without extra dimensions.
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-entropy loss with robust target handling.

        Parameters
        ----------
        input : torch.Tensor
            Predictions (logits) of shape (B, C, ...) where B is batch size and C number of classes.
        target : torch.Tensor
            Target labels which might have an extra dimension (e.g. shape (B, 1, ...)) and float dtype.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

class DC_and_CE_loss(Metric):
    """
    Combined Dice and Cross-Entropy loss metric.

    This metric combines Soft Dice loss and Cross Entropy loss, with optional weighting and masking support.
    Weights for CE and Dice do not need to sum to one and can be set independently.

    :ivar RobustCrossEntropyLoss ce: Cross entropy loss module.
    :ivar SoftDiceLoss dc: Soft Dice loss module.
    :ivar float weight_ce: Weight for the Cross Entropy loss.
    :ivar float weight_dice: Weight for the Dice loss.
    :ivar bool log_dice: Whether to log-transform the Dice loss.
    :ivar int | None ignore_label: Label to ignore during loss calculation.
    :ivar str aggregate: Method to aggregate losses (currently supports "sum" only).
    :ivar str name: Name of the metric.
    """

    def __init__(self, 
                 soft_dice_kwargs:dict[str,Any], 
                 ce_kwargs:dict[str,Any], 
                 aggregate:Literal["sum"]="sum", 
                 square_dice:bool=False, 
                 weight_ce:float=1.0, 
                 weight_dice:float=1.0,
                 log_dice:bool=False, 
                 ignore_label:Optional[int]=None,
                 name:Optional[str]=None):
        """
        Initialize the combined Dice and Cross Entropy loss metric.

        Parameters
        ----------
        soft_dice_kwargs : dict of str to any
            Keyword arguments for SoftDiceLoss.
        ce_kwargs : dict of str to any
            Keyword arguments for RobustCrossEntropyLoss.
        aggregate : "sum", default="sum"
            Aggregation method for losses. Only "sum" supported.
        square_dice : bool, default=False
            Use squared Dice loss variant (not implemented with ignore_label).
        weight_ce : float, default=1
            Weight for the Cross Entropy component.
        weight_dice : float, default=1
            Weight for the Dice component.
        log_dice : bool, default=False
            If True, apply negative log transform to Dice loss.
        ignore_label : int, optional
            Label to ignore during loss calculation.
        name : str, optional
            Name of the metric.
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

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined Dice and Cross Entropy loss.

        Parameters
        ----------
        net_output : torch.Tensor
            Model predictions (logits), shape (batch_size, num_classes, ...).
        target : torch.Tensor
            Ground truth tensor, shape (batch_size, 1, ...) or one-hot encoded.

        Returns
        -------
        self.val: torch.Tensor
            Combined loss scalar.
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
