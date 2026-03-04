"""
This module define some simple metrics.

The metrics defined here can't be used as loss function, contrarly to the module biom3d.metrics.
"""

from typing import Callable, Optional
from biom3d.utils import one_hot_fast
import numpy as np

def iou(inputs:np.ndarray, targets:np.ndarray, smooth:float=1.0)->float:
    """
    Calculate the Intersection over Union (IoU) score between two binary masks.

    Parameters
    ----------
    inputs : numpy.ndarray
        Binary array representing the predicted mask.
    targets : numpy.ndarray
        Binary array representing the ground truth mask.
    smooth : float, default=1.0
        Smoothing factor to avoid division by zero.

    Returns
    -------
    float
        IoU score between inputs and targets.
    """
    inter = (inputs & targets).sum()
    union = (inputs | targets).sum()
    return (inter+smooth)/(union+smooth)

def dice(inputs:np.ndarray, 
         targets:np.ndarray, 
         smooth:float=1.0, 
         axis:tuple[int]=(-3,-2,-1),
         )->float:   
    """
    Compute the Dice coefficient between inputs and targets.

    Parameters
    ----------
    inputs : numpy.ndarray
        Binary array or one-hot encoded mask of predictions.
    targets : numpy.ndarray
        Binary array or one-hot encoded mask of ground truth.
    smooth : float, default=1.0
        Smoothing factor to avoid division by zero.
    axis : tuple of int, default is last three axes (supposed spatial axis)
        Axes along which to compute the Dice score.

    Returns
    -------
    float
        Mean Dice score over specified axes.
    """
    inter = (inputs & targets).sum(axis=axis)   
    dice = (2.*inter + smooth)/(inputs.sum(axis=axis) + targets.sum(axis=axis) + smooth)  
    return dice.mean()

def absolute_volume_difference(pred, gt, voxel_volume=1.0, relative=False):
    vol_pred = pred.sum(axis=(-3,-2,-1)) * voxel_volume
    vol_gt   = gt.sum(axis=(-3,-2,-1)) * voxel_volume
    diff = np.abs(vol_pred - vol_gt)
    if relative:
        return diff / (vol_gt + 1e-8)
    else:
        return diff
    
def expected_calibration_error(probs, labels, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels)

    bins = np.linspace(0,1,n_bins+1)
    ece = 0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.sum() > 0:
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()
            ece += mask.mean() * abs(acc - conf)
    return ece

class MONAIMetricFactory:
    """
    MONAI metric wrapper that mimics simple NumPy metric behavior.
    Designed for batch size = 1 and evaluation only.
    """

    def __init__(self, metric_name, average_classes=False, **metric_kwargs):
        import monai.metrics as metrics

        if not hasattr(metrics, metric_name):
            raise ValueError(f"{metric_name} not found in monai.metrics")

        metric_class = getattr(metrics, metric_name)

        # Force non-reduced behavior
        metric_kwargs.setdefault("reduction", "none")
        metric_kwargs.setdefault("get_not_nans", False)
        metric_kwargs.setdefault("include_background", False)
        if metric_name == "HausdorffDistanceMetric":
            metric_kwargs.setdefault("percentile", 95)

        self.metric = metric_class(**metric_kwargs)
        self.average_classes = average_classes

    def __call__(self, inputs, targets):
        import torch

        # Convert to tensor
        inputs = torch.as_tensor(inputs)
        targets = torch.as_tensor(targets)

        # Ensure (B,C,...) shape
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0).unsqueeze(0)
            targets = targets.unsqueeze(0).unsqueeze(0)
        elif inputs.ndim == 4:
            inputs = inputs.unsqueeze(0)
            targets = targets.unsqueeze(0)

        self.metric.reset()
        self.metric(y_pred=inputs, y=targets)
        out = self.metric.aggregate()
        self.metric.reset()

        out = out.detach().cpu().squeeze(0)  # remove batch dim

        if self.average_classes:
            return out.mean().item()
        else:
            return out.numpy()

def versus_one(fct:Callable, 
               input_img:np.ndarray, 
               target_img:np.ndarray, 
               num_classes:int, 
               single_class:Optional[int]=None,
               is_sigmoid_output:bool=False,
               )->float|None:
    """
    Compare input and target images using a given metric function.

    This function:
    - Converts label images to one-hot encoding if needed.
    - Optionally selects a single class channel.
    - Binarizes masks.
    - Removes background class channel if present.
    - Checks shape compatibility.
    - Applies the provided comparison function `fct` on processed masks.

    Parameters
    ----------
    fct : callable
        A function that takes two binary masks and returns a metric score (e.g., IoU or Dice).
    input_img : numpy.ndarray
        Input image as label indices or one-hot encoded mask.
    target_img : numpy.ndarray
        Target (ground truth) image as label indices or one-hot encoded mask.
    num_classes : int
        Number of classes expected in input and target images.
    single_class : int or None, optional
        Index of class to compare individually. If None, compares all classes.
    is_sigmoid_output : bool, optional
        Whether the first dimension is a label dimension (CZYX), typically output of sigmoid layer.
        For instance a 3D image could have the following dimensions: (2, 64, 111, 110)
        Otherwise, dimensions are supposed to be ZYX.

    Returns
    -------
    float or None
        The score returned by `fct`.

    Notes
    -----
    If shapes after processing don't match, prints an error message and returns None.
    Copy the images so no side effect.
    """
    img1 = input_img.copy()
    img1 = img1.squeeze() 
    if not is_sigmoid_output:
        img1 = one_hot_fast(img1.astype(np.uint8), num_classes)[1:,...]
    if single_class is not None:
        img1 = img1[single_class,...]
    img1 = (img1 > 0).astype(int)
    
    img2 = target_img.copy()
    img2 = img2.squeeze() 
    if not is_sigmoid_output:
        img2 = one_hot_fast(img2.astype(np.uint8), num_classes)[1:,...]
    if single_class is not None:
        img2 = img2[single_class,...]
    img2 = (img2 > 0).astype(int)
    
    # remove background if needed
    if img1.shape[0]==(img2.shape[0]+1):
        img1 = img1[1:]
    if img2.shape[0]==(img1.shape[0]+1):
        img2 = img2[1:]
    
    if sum(img1.shape)!=sum(img2.shape):
        print("bug:sum(img1.shape)!=sum(img2.shape):")
        print("img1.shape", img1.shape)
        print("img2.shape", img2.shape)
        return # TODO should raise error
    out = fct(img1, img2)
    return out