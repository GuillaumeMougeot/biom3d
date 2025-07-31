from biom3d.utils import one_hot_fast
import numpy as np

# metric definition
def iou(inputs, targets, smooth=1):
    inter = (inputs & targets).sum()
    union = (inputs | targets).sum()
    return (inter+smooth)/(union+smooth)

def dice(inputs, targets, smooth=1, axis=(-3,-2,-1)):   
    """Dice score between inputs and targets.
    """
    inter = (inputs & targets).sum(axis=axis)   
    dice = (2.*inter + smooth)/(inputs.sum(axis=axis) + targets.sum(axis=axis) + smooth)  
    return dice.mean()

def versus_one(fct, input_img, target_img, num_classes, single_class=None):
    """
    comparison function between input image  and target images and using the criterion defined by fct
    """
    img1 = input_img.copy()
    if len(img1.shape)==3:
        img1 = one_hot_fast(img1.astype(np.uint8), num_classes)[1:,...]
    if single_class is not None:
        img1 = img1[single_class,...]
    img1 = (img1 > 0).astype(int)
    
    img2 = target_img.copy()
    if len(img2.shape)==3:
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
        return
    out = fct(img1, img2)
    return out