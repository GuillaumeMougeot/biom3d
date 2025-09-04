"""Dataloader with batch_generator. Follow the nnUNet augmentation pipeline."""

import random
import numpy as np
import pandas as pd

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform

from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from typing import Any, Hashable, Optional, Iterable

from biom3d.utils import DataHandlerFactory, DataHandler, get_folds_train_test_df

#---------------------------------------------------------------------------
# random crop and pad with batchgenerator

def centered_crop(img:np.ndarray, 
                  msk:np.ndarray, 
                  center:Iterable[int], 
                  crop_shape:Iterable[int], 
                  margin:Iterable[float]=np.zeros(3),
                  )->tuple[np.ndarray,np.ndarray]:
    """
    Do a crop, forcing the location voxel to be located in the center of the crop.

    Parameters
    ----------
    img: numpy.ndarray
        Image data.
    msk: numpy.ndarray
        Mask data.
    center: iterable of int
        Center voxel location for cropping.
    crop_shape: iterable of int
        Shape of the crop.
    margin: iterable of float, default=np.zeros(3)
        Margin around the center location.

    Returns
    -------    
    crop_img : numpy.ndarray
        The cropped image, centered around center.
    crop_msk: numpy.ndarray
        The cropped mask, centered around center.
    """
    center = np.array(center)
    crop_shape = np.array(crop_shape)
    margin = np.array(margin)
    
    # middle of the crop
    start = np.maximum(0,center-crop_shape//2+margin).astype(int)

    # assert that the end will not be out of the crop
    end = crop_shape+start
    
    idx = [slice(0,img.shape[0])]+[slice(s[0], s[1]) for s in zip(start, end)]
    idx = tuple(idx)
    
    crop_img = img[idx]
    crop_msk = msk[idx]
    return crop_img, crop_msk

def located_crop(img:np.ndarray,
                 msk:np.ndarray,
                 location:Iterable[int], 
                 crop_shape:Iterable[int], 
                 margin:Iterable[float]=np.zeros(3),
                 )->tuple[np.ndarray,np.ndarray]:
    """Do a crop, forcing the location voxel to be located in the crop.
    
    Parameters
    ----------
    img : numpy.ndarray
        Image data.
    msk : numpy.ndarray
        Mask data.
    location : iterable of int
        Specific voxel location to include in the crop.
    crop_shape : iterable of int
        Shape of the crop.
    margin : iterable of float, default=np.zeros(3)
        Margin around the location.

    Returns
    -------
    crop_img : numpy.ndarray
        Cropped image data, containing the specified location voxel within the crop.
    crop_msk : numpy.ndarray
        Cropped mask data, corresponding to the cropped image region.
    """
    img_shape = np.array(img.shape)[1:]
    location = np.array(location)
    crop_shape = np.array(crop_shape)
    margin = np.array(margin)
    
    lower_bound = np.maximum(0,location-crop_shape+margin)
    higher_bound = np.maximum(lower_bound+1,np.minimum(location-margin, img_shape-crop_shape))
    start = np.random.randint(low=lower_bound, high=np.maximum(lower_bound+1,higher_bound))
    end = start+crop_shape
    
    idx = [slice(0,img.shape[0])]+[slice(s[0], s[1]) for s in zip(start, end)]
    idx = tuple(idx)
    
    crop_img = img[idx]
    crop_msk = msk[idx]
    return crop_img, crop_msk

def foreground_crop(img:np.ndarray, 
                    msk:np.ndarray, 
                    final_size:Iterable[int], 
                    fg_margin:Iterable[float], 
                    fg:Optional[dict[int,np.ndarray]]=None, 
                    use_softmax:bool=True,
                    )->tuple[np.ndarray,np.ndarray]:
    """Do a foreground crop.
    
    Parameters
    ----------
    img : numpy.ndarray
        Image data.
    msk : numpy.ndarray
        Mask data.
    final_size : iterable of int
        Final size of the cropped image and mask.
    fg_margin : iterable of float
        Margin around the foreground location.
    fg : dict of int to numpy.ndarray, optional
        Foreground information.
    use_softmax : bool, default=True
        If True, assumes softmax activation.

    Returns
    -------
    img : numpy.ndarray
        Cropped image data, focused on the foreground region.
    msk : numpy.ndarray
        Cropped mask data, corresponding to the cropped image region.
        
    """
    if fg is not None:
        locations = fg[random.choice(list(fg.keys()))]
    else:
        if tuple(msk.shape)[0]==1:
            # then we consider that we don't have a one hot encoded label
            rnd_label = random.randint(1,msk.max()+1)
            locations = np.argwhere(msk[0] == rnd_label)
        else:
            # then we have a one hot encoded label
            rnd_label = random.randint(int(use_softmax),tuple(msk.shape)[0]-1)
            locations = np.argwhere(msk[rnd_label] == 1)

    if np.array(locations).size==0: # bug fix when having empty arrays 
        img, msk = random_crop(img, msk, final_size)
    else:
        center=random.choice(locations) # choose a random voxel of this label
        img, msk = centered_crop(img, msk, center, final_size, fg_margin)
    return img, msk

def random_crop(img:np.ndarray,
                msk:np.ndarray, 
                crop_shape:Iterable[int]
                )->tuple[np.ndarray,np.ndarray]:
    """
    Randomly crop a portion of size prop of the original image size.
    
    Parameters
    ----------
    img : numpy.ndarray
        Image data.
    msk : numpy.ndarray
        Mask data.
    crop_shape : array_like
        Shape of the crop.

    Raises
    ------
    AssertionError:
        If img and crop_shape doesn't have the same number of dimensions.

    Returns
    -------
    crop_img : numpy.ndarray
        Cropped image data.
    crop_msk : numpy.ndarray
        Cropped mask data.
    """  
    img_shape = np.array(img.shape)[1:]
    assert len(img_shape)==len(crop_shape),"[Error] Not the same dimensions! Image shape {}, Crop shape {}".format(img_shape, crop_shape)
    start = np.random.randint(0, np.maximum(1,img_shape-crop_shape))
    end = start+crop_shape
    
    idx = [slice(0,img.shape[0])]+[slice(s[0], s[1]) for s in zip(start, end)]
    idx = tuple(idx)
    
    crop_img = img[idx]
    crop_msk = msk[idx]
    return crop_img, crop_msk

def centered_pad(img:np.ndarray, 
                 final_size:np.ndarray, 
                 msk:Optional[np.ndarray]=None,
                 )->np.ndarray|tuple[np.ndarray,np.ndarray]:
    """
    Centered pad an img and msk to fit the final_size.
    
    Parameters
    ----------
    img : numpy.ndarray
        Image data.
    final_size : array_like
        Final size after padding.
    msk : numpy.ndarray, optional
        Mask data.

    Returns
    -------
    pad_img: numpy.ndarray
        Padded image.
    pad_mask: numpy.ndarray, optional
        Padded image
    """
    final_size = np.array(final_size)
    img_shape = np.array(img.shape[1:])
    
    start = (final_size-np.array(img_shape))//2
    start = start * (start > 0)
    end = final_size-(img_shape+start)
    end = end * (end > 0)
    
    pad = np.append([[0,0]], np.stack((start,end),axis=1), axis=0)
    pad_img = np.pad(img, pad, 'constant', constant_values=0)
    
    if msk is not None:
        pad_msk = np.pad(msk, pad, 'constant', constant_values=0)
        return pad_img, pad_msk
    else: 
        return pad_img

def random_crop_pad(img:np.ndarray, 
                    msk:np.ndarray, 
                    final_size:Iterable[int], 
                    fg_rate:float=0.33, 
                    fg_margin:Iterable[float]=np.zeros(3), 
                    fg:Optional[dict[str,np.ndarray]]=None, 
                    use_softmax:bool=True,
                    )->tuple[np.ndarray,np.ndarray]:
    """
    Random crop and pad if needed.
    
    Parameters
    ----------
    img : numpy.ndarray
        Image data.
    msk : numpy.ndarray
        Mask data.
    final_size : iterable of int
        Final size after cropping and padding.
    fg_rate : float, default=0.33
        Probability of focusing the crop on the foreground.
    fg_margin : iterable of float, optional
        Margin around the foreground location.
    fg : dict of int to numpy.ndarray, optional
        Foreground information.
    use_softmax : bool, default=True
        If True, assumes softmax activation; otherwise sigmoid is used.
    Returns
    -------
    img : numpy.ndarray
        Cropped and padded image data.
    msk : numpy.ndarray
        Cropped and padded mask data.
    """
    if isinstance(img,list): # then batch mode
        imgs, msks = [], []
        for i in range(len(img)):
            img_, msk_ = random_crop_pad(img[i], msk[i], final_size)
            imgs += [img_]
            msks += [msk_]
        return np.array(imgs), np.array(msks) # can convert to array as they should have now all the same shape
    
    # choose if using foreground centrered or random alignement
    force_fg = random.random()
    if fg_rate>0 and force_fg<fg_rate:
        img, msk = foreground_crop(img, msk, final_size, fg_margin, fg=fg, use_softmax=use_softmax)
    else:
        # or random crop
        img, msk = random_crop(img, msk, final_size)
        
    # pad if needed
    if np.any(np.array(img.shape)[1:]-final_size)!=0:
        img, msk = centered_pad(img=img, msk=msk, final_size=final_size)
    return img, msk

class RandomCropAndPadTransform(AbstractTransform):
    """
    BatchGenerator transform for random cropping and padding.

    :ivar str data_key: Key used to access data in dictionary.
    :ivar str label_key: Key used to access label in dictionary.
    :ivar float fg_rate: Foreground rate, probability of focusing crop on foreground.
    :ivar Iterable[int] crop_size: Size of the crop.
    """

    data_key: str
    label_key: str
    fg_rate: float
    crop_size: Iterable[int]

    def __init__(self, 
                 crop_size:Iterable[int], 
                 fg_rate:float=0.33, 
                 data_key:str="data", 
                 label_key:str="seg"):
        """
        Batch generator transform for random cropping and padding.

        Parameters
        ----------
        crop_size : iterable of int
            Size of the crop.
        fg_rate : float, default=0.33
            Probability of focusing the crop on the foreground.
        data_key : str, default="data"
            Key for the data in the data dictionary.
        label_key : str, default="seg"
            Key for the label in the data dictionary.
        """
        self.data_key = data_key
        self.label_key = label_key
        self.fg_rate = fg_rate
        self.crop_size = crop_size

    def __call__(self, **data_dict:dict[str,Any])->dict[str,Any]:
        """
        Apply random cropping and padding transform to the data dictionary.

        Parameters
        ----------
        **data_dict : dict
            Dictionary containing data arrays. Must contain keys matching
            `self.data_key` and `self.label_key` which correspond to
            the input data and segmentation mask respectively.

        Returns
        -------
        data_dict: dict
            The modified data dictionary with cropped and padded data and mask.
        """
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        data, seg = random_crop_pad(data, seg, self.crop_size, self.fg_rate)

        data_dict[self.data_key] = data
        data_dict[self.label_key] = seg

        return data_dict

#---------------------------------------------------------------------------
# image reader

def imread(handler:DataHandler,
           img:str, 
           msk:str, 
           loc:Optional[str]=None,
           is3d:bool=True,
           )->tuple[np.ndarray,np.ndarray,np.ndarray|None]:
    """
    Read all data with the provided DataHandler.

    Parameters
    ----------
    handler: DataHandler
        The DataHandler used to read data.
    img: str
        The path to the image.
    msk: str
        The path to the mask.
    loc: str, optional
        The path to the foreground. If None, no foreground will be returned.
    is3d: bool, default=True
        If image is in 3D

    Returns
    -------
    img: numpy.ndarray
        The image.
    msk: numpy.ndarray
        The mask.
    fg: numpy.ndarray, optional
        The foreground, or None.
    """
    img,_ = handler.load(img)
    msk,_ = handler.load(msk)
    if loc is not None : fg,_ = handler.load(loc)

    if len(img.shape) == 3 if is3d else 2:
        img = np.expand_dims(img,0)
    if len(msk.shape) == 3 if is3d else 2:
        msk = np.expand_dims(msk,0)

    assert (is3d and len(msk.shape)==4 and len(img.shape)==4) or (not is3d and len(msk.shape)==3 and len(img.shape)==3), "[Error] Your data has the wrong dimension."
    return img, msk, fg if loc is not None else None

class DataReader(AbstractTransform):
    """Read the data and add it to dictionary.

    :ivar str data_key: Key used to access data in dictionary.
    :ivar str label_key: Key used to access label in dictionary.
    :ivar str loc_key: Key used to access foreground in dictionary.
    :ivar bool is3d: If images are in 3d, not used yet.
    :ivar DataHandler handler: DataHandler used to read data.
    """

    data_key: str
    label_key: str
    loc_key: str
    is3d: bool
    handler: DataHandler
    
    def __init__(self, 
                 handler:DataHandler,
                 is3d:bool=True, 
                 data_key:str="data", 
                 label_key:str="seg",
                 loc_key:str='loc'):
        """Read the data and add it to dictionary.

        Parameters
        ----------
        handler : DataHandler
            DataHandler used to read data
        is3d : bool
            If images are in 3d, not used yet.
        data_key : str
            Key used to access data in dictionary.
        label_key : str
            Key used to access label in dictionary.
        loc_key : str
            Key used to access foreground in dictionary.
        """
        self.is3d = is3d
        self.data_key = data_key
        self.label_key = label_key
        self.loc_key= loc_key
        self.handler=handler
    
    def __call__(self, **data_dict:dict[str,Any])->dict[str,Any]:
        """
        Add data to the data_dict.

        Parameters
        ----------
        **data_dict : dict
            Dictionary containing data arrays. Must contain keys matching
            `self.data_key`, `self.label_key` and `self.loc_key` which correspond to
            the input data, segmentation mask and foreground respectively.

        Returns
        -------
        data_dict: dict
            The modified data dictionary with raw data added to their keys.
        """
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        loc = data_dict.get(self.loc_key)
        if isinstance(data,list):
            for i in range(len(data)):
                data[i], seg[i],loc[i] = imread(self.handler,data[i], seg[i],loc[i])
        else:
            data, seg,loc = imread(self.handler,data, seg,loc)

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg
        if loc is not None:
            data_dict[self.loc_key] = loc
            
        return data_dict
    
#---------------------------------------------------------------------------
# training and validation augmentations

def get_bbox(patch_size:Iterable[int],
             final_patch_size:Iterable[int],
             annotated_classes_key:Hashable,
             data_shape: np.ndarray, 
             force_fg: bool, 
             class_locations: Optional[dict],
             overwrite_class: Optional[int| tuple[int, ...]] = None, 
             verbose: bool = False
            )->tuple[list[int],list[int]]:
    """
    Compute bounding box coordinates for cropping a patch from the data, optionally focusing on foreground regions.

    Parameters
    ----------
    patch_size : iterable of int
        Desired patch size to crop (dimensions).
    final_patch_size : iterable of int
        Current size of the patch after any previous cropping or resizing.
    annotated_classes_key : hashable
        Key identifying the annotated class in `class_locations`.
    data_shape : numpy.ndarray
        Shape of the full data volume or image from which the patch is cropped.
    force_fg : bool
        If True, ensures the patch contains at least one voxel of foreground classes.
    class_locations : dict or None
        Dictionary mapping class labels (int or tuple) to lists/arrays of voxel coordinates for that class.
        Required if `force_fg` is True.
    overwrite_class : int or tuple of int, optional
        If set, forces the patch to focus on this class instead of randomly selected foreground class.
    verbose : bool, default=False
        If True, prints diagnostic messages.

    Raises
    ------
    AssertionError:
        If class_locations is None and force_fg is True. Or overwrite_class not in class_locations

    Returns
    -------
    bbox_lbs : list of int
        Lower bounds (start indices) of the bounding box along each dimension.
    bbox_ubs : list of int
        Upper bounds (end indices) of the bounding box along each dimension.
    
    Notes
    -----
    - The function calculates how much padding is needed if `final_patch_size` is smaller than `patch_size`.
    - If `force_fg` is True, it attempts to center the bounding box on a randomly selected voxel of a foreground class.
    - If no foreground voxel is found, it falls back to random cropping.
    """
    # Force patch_size to have a get_item method
    patch_size = np.array(patch_size)
    # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
    # locations for the given slice
    need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
    dim = len(data_shape)

    for d in range(dim):
        # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
        # always
        if need_to_pad[d] + data_shape[d] < patch_size[d]:
            need_to_pad[d] = patch_size[d] - data_shape[d]

    # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
    # define what the upper and lower bound can be to then sample form them with np.random.randint
    lbs = [- need_to_pad[i] // 2 for i in range(dim)]
    ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - patch_size[i] for i in range(dim)]

    # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
    # at least one of the foreground classes in the patch
    if not force_fg:
        bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
    else:
        assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
        if overwrite_class is not None:
            assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                'have class_locations (missing key)'
        # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
        # class_locations keys can also be tuple
        eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

        # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
        # strange formulation needed to circumvent
        tmp = [i == annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
        if any(tmp) and len(eligible_classes_or_regions) > 1:
                eligible_classes_or_regions.pop(np.nonzero(tmp)[0][0])

        if len(eligible_classes_or_regions) == 0:
            # this only happens if some image does not contain foreground voxels at all
            selected_class = None
            if verbose:
                print('Case does not contain any foreground classes')
        else:
            # I hate myself. Future me aint gonna be happy to read this
            # 2022_11_25: had to read it today. Wasn't too bad
            # 2025_08_07, Clement : speak for yourself
            selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class

        voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

        if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
            selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
            # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
            # Make sure it is within the bounds of lb and ub
            # i + 1 because we have first dimension 0!
            bbox_lbs = [max(lbs[i], selected_voxel[i] - patch_size[i] // 2) for i in range(dim)]
        else:
            # If the image does not contain any foreground classes, we fall back to random cropping
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

    bbox_ubs = [bbox_lbs[i] + patch_size[i] for i in range(dim)]

    return bbox_lbs, bbox_ubs
    
class nnUNetRandomCropAndPadTransform(AbstractTransform):
    """
    Random cropping and padding transform for nnU-Net-style data augmentation.

    Applies random crop centered around a foreground voxel with a certain probability (fg_rate),
    and pads the data and label to the desired augmented crop size.

    :ivar Iterable[int] aug_crop_size : Final shape after cropping and padding (target shape).
    :ivar Iterable[int] crop_size : Crop size for network input (may differ from aug_crop_size).
    :ivar float fg_rate : Probability of forcing the crop to focus on the foreground class.
    :ivar str data_key : Key for the input data in the data dictionary.
    :ivar str label_key : Key for the segmentation labels in the data dictionary.
    :ivar str class_loc_key : Key for the precomputed voxel locations per class in the data dictionary.
    """

    data_key:str
    label_key:str
    class_loc_key:str
    fg_rate:float
    crop_size:Iterable[int]
    aug_crop_size:Iterable[int]
    
    def __init__(self,
                 aug_crop_size:Iterable[int], 
                 crop_size:Iterable[int], 
                 fg_rate:float=0.33,
                 data_key:str="data", 
                 label_key:str="seg",
                 class_loc_key:str="loc",
                ):
        """
        Random cropping and padding transform for nnU-Net-style data augmentation.

        Parameters
        ----------
        aug_crop_size : iterable of int
            Final shape after cropping and padding (target shape).
        crop_size : iterable of int
            Crop size for network input (may differ from aug_crop_size).
        fg_rate : float, default=0.33
            Probability of forcing the crop to focus on the foreground class.
        data_key : str, default="data"
            Key for the input data in the data dictionary.
        label_key : str, default="seg"
            Key for the segmentation labels in the data dictionary.
        class_loc_key : str, default="loc"
            Key for the precomputed voxel locations per class in the data dictionary.
        """
        self.data_key = data_key
        self.label_key = label_key
        self.class_loc_key = class_loc_key
        self.fg_rate = fg_rate
        self.crop_size = crop_size
        self.aug_crop_size = aug_crop_size

    def __call__(self, **data_dict:dict[str,Any])->dict:
        """
        Apply the crop and pad transform to a batch of data.

        Parameters
        ----------
        **data_dict : dict
            Dictionary containing data arrays. Must contain keys matching
            `self.data_key`, `self.label_key` and `self.loc_key` which correspond to
            the input data, segmentation mask and foreground respectively.

        Returns
        -------
        **data_dict : dict
            Updated data dictionary with cropped and padded arrays.
        """
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        loc = data_dict.get(self.class_loc_key)
        dim=len(data[0].shape[1:])
        
        data_channel = data[0].shape[0]
        seg_channel = seg[0].shape[0]
        
        data_all = np.zeros([len(data), data_channel]+list(self.aug_crop_size), dtype=np.float32)
        seg_all = np.zeros([len(seg), seg_channel]+list(self.aug_crop_size), dtype=np.int16)
        
        for j,(d,s,l) in enumerate(zip(data,seg,loc)):
            shape = np.array(d.shape[1:])

            bbox_lbs, bbox_ubs = get_bbox(
                final_patch_size=self.crop_size,
                patch_size=self.aug_crop_size,
                annotated_classes_key=list(l.keys()),
                data_shape=shape, 
                force_fg=random.random()<self.fg_rate, 
                class_locations=l,
                overwrite_class = None, 
                verbose = False
            )
            
            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data_channel)] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            d = d[this_slice]

            this_slice = tuple([slice(0, seg_channel)] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            s = s[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(d, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(s, ((0, 0), *padding), 'constant', constant_values=-1)

        data_dict[self.data_key] = data_all
        data_dict[self.label_key] = seg_all

        return data_dict

class Convert2DTo3DTransform(AbstractTransform):
    """
    Reverts Convert3DTo2DTransform by transforming a 4D array (b, c * x, y, z) back to 5D  (b, c, x, y, z).

    :ivar list[str] | tuple[str] apply_to_keys: Key of the data dictionary to convert, default=('data','seg')
    """

    apply_to_keys:list[str]| tuple[str]

    def __init__(self, 
                 apply_to_keys: list[str]| tuple[str] = ('data', 'seg'),
                 ):
        """
        Reverts Convert3DTo2DTransform by transforming a 4D array (b, c * x, y, z) back to 5D  (b, c, x, y, z).

        Parameters
        ----------
        apply_to_keys: list or tuple of str, default=('data','seg')
            Key of the data dictionary to convert
        """
        self.apply_to_keys = apply_to_keys

    def __call__(self, **data_dict:dict[str,Any])->dict:
        """
        Apply the conversion to a batch of data.

        Parameters
        ----------
        **data_dict : dict
            Dictionary containing data arrays. Must contain keys matching
            `self.aply_to_keys` which correspond to the input data and segmentation mask.

        Raises
        ------
        AssertionError:
            If a key of apply_to_keys is not in data_dict.

        Returns
        -------
        **data_dict : dict
            Updated data dictionary with 3D transform.
        """
        for k in self.apply_to_keys:
            shape_key = f'orig_shape_{k}'
            assert shape_key in data_dict.keys(), f'Did not find key {shape_key} in data_dict. Shitty. ' \
                                                  f'Convert2DTo3DTransform only works in tandem with ' \
                                                  f'Convert3DTo2DTransform and you probably forgot to add ' \
                                                  f'Convert3DTo2DTransform to your pipeline. (Convert3DTo2DTransform ' \
                                                  f'is where the missing key is generated)'
            original_shape = data_dict[shape_key]
            current_shape = data_dict[k].shape
            data_dict[k] = data_dict[k].reshape((original_shape[0], original_shape[1], original_shape[2],
                                                 current_shape[-2], current_shape[-1]))
        return data_dict

class Convert3DTo2DTransform(AbstractTransform):
    """
    Transforms a 5D array (b, c, x, y, z) to a 4D array (b, c * x, y, z) by overloading the color channel.

    :ivar list[str] | tuple[str] apply_to_keys: Key of the data dictionary to convert, default=('data','seg')
    """

    def __init__(self, apply_to_keys: list[str]| tuple[str] = ('data', 'seg')):
        """
        Transform a 5D array (b, c, x, y, z) to a 4D array (b, c * x, y, z) by overloading the color channel.

        Parameters
        ----------
        apply_to_keys: list or tuple of str, default=('data','seg')
            Key of the data dictionary to convert
        """
        self.apply_to_keys = apply_to_keys

    def __call__(self, **data_dict:dict[str,Any])->dict:
        """
        Apply the conversion to a batch of data.

        Parameters
        ----------
        **data_dict : dict
            Dictionary containing data arrays. Must contain keys matching 
            `self.aply_to_keys` which correspond to the input data and segmentation mask.

        Raises
        ------
        AssertionError:
            If a key of apply_to_keys is not in data_dict

        Returns
        -------
        **data_dict : dict
            Updated data dictionary with 2D transform.
        """
        for k in self.apply_to_keys:
            shp = data_dict[k].shape
            assert len(shp) == 5, 'This transform only works on 3D data, so expects 5D tensor (b, c, x, y, z) as input.'
            data_dict[k] = data_dict[k].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
            shape_key = f'orig_shape_{k}'
            assert shape_key not in data_dict.keys(), f'Convert3DTo2DTransform needs to store the original shape. ' \
                                                      f'It does that using the {shape_key} key. That key is ' \
                                                      f'already taken. Bummer.'
            data_dict[shape_key] = shp
        return data_dict

class DictToTuple(AbstractTransform):
    """
    Return a data and seg instead of a dictionary.

    :ivar str data_key: Key for the input data in the dictionary, default="data"
    :ivar str label_key: Key for the label/segmentation in the dictionary, default="seg"
    """

    data_key:str
    label_key:str

    def __init__(self, data_key:str="data", label_key:str="seg"):
        """
        Transform that extracts `data` and `seg` from a dictionary and returns them as a tuple.

        Parameters
        ----------
        data_key : str, default="data"
            Key for the input data in the dictionary.
        label_key : str, default="seg"
            Key for the label/segmentation in the dictionary.
        """
        self.data_key = data_key
        self.label_key = label_key
    
    def __call__(self, **data_dict:dict[str,Any])->tuple[Any,Any]:
        """
        Extract data_key and label_key from a dictionary and returns them as a tuple.

        Parameters
        ----------
        **data_dict : dict
            Dictionary containing data arrays. Must contain keys matching 
            `self.data_key` and `self.label_key` which correspond to the input data and segmentation mask.

        Returns
        -------
        data
            Dictionary entry for data_key
        seg
            Dictionary entry for label_key
        """
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        if isinstance(seg,list): seg = seg[0]
        return data, seg
    
class DownsampleSegForDSTransform2(AbstractTransform):
    """
    Transform that generates downsampled versions of a segmentation map for deep supervision.

    This transform stores the results in `data_dict[output_key]` as a list of segmentations, each scaled 
    according to a corresponding entry in `ds_scales`.

    :ivar tuple | List ds_scales: Scaling factors per deep supervision level. Each entry can be a float (same scaling for all axes) or a tuple of floats (individual scaling per axis).
    :ivar int order: Interpolation order to use for resizing (0 = nearest neighbor).
    :ivar str input_key: Key to access the input segmentation in `data_dict`.
    :ivar str output_key: Key under which to store the output list of downsampled segmentations.
    :ivar tuple[int] axes: Axes along which to apply the downsampling. If None, assumes axes are (2, 3, 4), i.e., skips batch and channel.
    
    """

    axes:tuple[int]
    output_key:str
    input_key:str
    order:int
    ds_scales:list| tuple

    def __init__(self, 
                 ds_scales: list | tuple,
                 order: int = 0, 
                 input_key: str = "seg",
                 output_key: str = "seg", 
                 axes: Optional[tuple[int]] = None):
        """
        Transform that generates downsampled versions of a segmentation map for deep supervision.

        This transform stores the results in `data_dict[output_key]` as a list of segmentations, each scaled 
        according to a corresponding entry in `ds_scales`.

        Each entry in ds_scales specified one deep supervision
        output and its resolution relative to the original data, for example 0.25 specifies 1/4 of the original shape.
        ds_scales can also be a tuple of tuples, for example ((1, 1, 1), (0.5, 0.5, 0.5)) to specify the downsampling
        for each axis independently

        Parameters
        ----------
        ds_scales : list or tuple
            Scaling factors per deep supervision level. Each entry can be a float (same scaling for all axes) or a 
            tuple of floats (individual scaling per axis).
        order : int, default=0
            Interpolation order to use for resizing (0 = nearest neighbor).
        input_key : str, default="seg"
            Key to access the input segmentation in `data_dict`.
        output_key : str, default="seg"
            Key under which to store the output list of downsampled segmentations.
        axes : tuple of int, optional
            Axes along which to apply the downsampling. If None, assumes axes are (2, 3, 4), i.e., skips batch and channel.
        """
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, **data_dict:dict[str,Any])->dict:
        """
        Apply the downsampling to a batch of data.

        Parameters
        ----------
        **data_dict : dict
            Dictionary containing input data. Must contain `input_key`.

        Raises
        ------
        AssertionError:
            If a element of ds_scales has not the same length as axes

        Returns
        -------
        **data_dict : dict
            Modified `data_dict` with `output_key` storing a list of downsampled segmentations.
        """
        if self.axes is None:
            axes = list(range(2, len(data_dict[self.input_key].shape)))
        else:
            axes = self.axes

        output = []
        for s in self.ds_scales:
            if not isinstance(s, (tuple, list)):
                s = [s] * len(axes)
            else:
                assert len(s) == len(axes), f'If ds_scales is a tuple for each resolution (one downsampling factor ' \
                                            f'for each axis) then the number of entried in that tuple (here ' \
                                            f'{len(s)}) must be the same as the number of axes (here {len(axes)}).'

            if all(i == 1 for i in s):
                output.append(data_dict[self.input_key])
            else:
                new_shape = np.array(data_dict[self.input_key].shape).astype(float)
                for i, a in enumerate(axes):
                    new_shape[a] *= s[i]
                new_shape = np.round(new_shape).astype(int)
                out_seg = np.zeros(new_shape, dtype=data_dict[self.input_key].dtype)
                for b in range(data_dict[self.input_key].shape[0]):
                    for c in range(data_dict[self.input_key].shape[1]):
                        out_seg[b, c] = resize_segmentation(data_dict[self.input_key][b, c], new_shape[2:], self.order)
                output.append(out_seg)
        data_dict[self.output_key] = output
        return data_dict
    
def get_training_transforms(aug_patch_size: np.ndarray| tuple[int],
                            patch_size: np.ndarray | tuple[int],
                            fg_rate: float,
                            rotation_for_DA: dict,
                            deep_supervision_scales: list | tuple | None,
                            mirror_axes: tuple[int, ...],
                            handler:DataHandler,
                            do_dummy_2d_data_aug: bool,
                            order_resampling_data: int = 3,
                            order_resampling_seg: int = 1,
                            border_val_seg: int = -1,
                            use_data_reader: bool = True,
                            ) -> AbstractTransform:
    """
    Create a composed transform pipeline for training data augmentation, following the nnU-Net conventions.

    Parameters
    ----------
    aug_patch_size : numpy.ndarray or tuple of int
        Size of the patch used during augmentation (may be larger than `patch_size`).
    patch_size : numpy.ndarray or tuple of int
        Final cropped patch size used for training.
    fg_rate : float
        Probability of cropping patches that contain foreground voxels.
    rotation_for_DA : dict
        Dictionary specifying rotation angles for data augmentation. Should contain keys 'x', 'y', and 'z'.
    deep_supervision_scales : list, tuple or None
        List of scales for deep supervision. Used to downsample segmentation masks accordingly.
    mirror_axes : tuple[int, ...]
        Axes along which to apply mirroring (e.g., (0, 1, 2)).
    handler : DataHandler
        DataHandler used to load images. Used only if use_data_reader is True
    do_dummy_2d_data_aug : bool
        If True, applies dummy 2D data augmentation (by slicing 3D volumes).
    order_resampling_data : int, default=3
        Interpolation order used for resampling image data.
    order_resampling_seg : int, default=1
        Interpolation order used for resampling segmentation masks.
    border_val_seg : int, default=-1
        Border value used for segmentation padding.
    use_data_reader : bool, default=True
        If True, includes the DataReader transform in the pipeline.

    Returns
    -------
    AbstractTransform
        A composed transformation pipeline to be applied to training data.
    """
    tr_transforms = []
    
    if use_data_reader:
        tr_transforms.append(DataReader(handler))
    
    tr_transforms.append(nnUNetRandomCropAndPadTransform(aug_patch_size, 
                 patch_size, 
                 fg_rate,
                 data_key="data", 
                 label_key="seg",
                 class_loc_key="loc",))
    
    if do_dummy_2d_data_aug:
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None
    
    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
        do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
        p_rot_per_axis=1,  # todo experiment with this
        do_scale=True, scale=(0.7, 1.4),
        border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
        random_crop=False,  # random cropping is part of our dataloaders
        p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=False  # todo experiment with this
    ))

    if do_dummy_2d_data_aug:
        tr_transforms.append(Convert2DTo3DTransform())
        
    tr_transforms.append(CenterCropTransform(patch_size, data_key='data', label_key='seg'))

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if deep_supervision_scales is not None:
        tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                          output_key='target'))
    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms.append(DictToTuple('data', 'target'))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_validation_transforms(patch_size: np.ndarray | tuple[int],
                              fg_rate: float,
                              handler:DataHandler,
                              deep_supervision_scales: list | tuple | None = None,
                              use_data_reader: bool = True,
                              ) -> AbstractTransform:
    """
    Create a composed transformation pipeline for validation data, following the nnU-Net conventions.

    Parameters
    ----------
    patch_size : numpy.ndarray or tuple of int
        Size of the patch used for cropping and padding.
    fg_rate : float
        Probability of focusing on foreground regions when cropping.
    handler : DataHandler
        DataHandler used to load images. Used only if use_data_reader is True
    deep_supervision_scales : list, tuple or None, optional
        List of scales for deep supervision. If provided, segmentation masks will be downsampled accordingly.
    use_data_reader : bool, default=True
        If True, includes the DataReader transform to load data from disk.

    Returns
    -------
    AbstractTransform
        A composed transform pipeline to be applied during validation.
    """
    val_transforms = []

    if use_data_reader:
        val_transforms.append(DataReader(handler))

    val_transforms.append(nnUNetRandomCropAndPadTransform(patch_size, 
                 patch_size, 
                 fg_rate,
                 data_key="data", 
                 label_key="seg",
                 class_loc_key="loc",))
    
    val_transforms.append(RemoveLabelTransform(-1, 0))

    val_transforms.append(RenameTransform('seg', 'target', True))


    if deep_supervision_scales is not None:
        val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                           output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms.append(DictToTuple('data', 'target'))
    val_transforms = Compose(val_transforms)
    return val_transforms

#---------------------------------------------------------------------------
# dataloader

class BatchGenDataLoader(SlimDataLoaderBase):
    """
    Similar as torchio.SubjectsDataset but can be use with an unlimited amount of steps.

    :ivar str img_path: Path to collection containing the images.
    :ivar str msk_path: Path to collection containing the masks.
    :ivar str | None fg_path: Path to collection containing foreground information.
    :ivar int batch_size: Size of the batches.
    :ivar int nbof_steps: Number of steps per epoch.
    :ivar numpy.ndarray indices: A array of unsigned int representing the possibles index for images.
    :ivar int current_position: Index of the actual image.
    :ivar bool was_initialized: If the batch generator was initialized, used for safeguarding.
    """

    img_path:str
    msk_path:str
    fg_path:Optional[str]
    batch_size:int
    nbof_steps:int
    load_data:bool
    indices:np.ndarray
    current_position:int
    was_initialized:bool
    
    def __init__(
        self,
        img_path:str,
        msk_path:str,
        batch_size:int, 
        nbof_steps:int,
        fg_path:Optional[str] = None,
        folds_csv:Optional[str] = None, 
        fold :int      = 0, 
        val_split:float  = 0.25,
        train :bool     = True,
        load_data:bool = False,
        
        # batch_generator parameters
        num_threads_in_mt=12, 
    ):
        """
        Similar as torchio.SubjectsDataset but can be use with an unlimited amount of steps.
        
        Parameters
        ----------
        img_path : str
            Path to collection containing the images.
        msk_path : str
            Path to collection containing the masks.
        batch_size : int
            Size of the batches.
        nbof_steps : int
            Number of steps per epoch.
        fg_path : str, optional
            Path to collection containing foreground information.
        folds_csv : str, optional
            CSV file containing fold information for dataset splitting.
        fold : int, optional
            Current fold number for training/validation splitting.
        val_split : float, optional
            Proportion of data to be used for validation.
        train : bool, optional
            If True, use the dataset for training; otherwise, use it for validation.
        load_data : bool, optional
            if True, loads the all dataset into computer memory (faster but more memory expensive). ONLY COMPATIBLE WITH .npy PREPROCESSED IMAGES
        num_threads_in_mt : int, optional
            Number of threads in multi-threaded augmentation.
        """
        self.img_path = img_path
        self.msk_path = msk_path
        self.fg_path = fg_path

        self.batch_size = batch_size

        self.nbof_steps = nbof_steps

        self.load_data = load_data

        handler = DataHandlerFactory.get(
            self.img_path,
            read_only=True,
            msk_path = msk_path,
            fg_path = fg_path,
        )
        
        # get the training and validation names 
        if folds_csv is not None:
            df = pd.read_csv(folds_csv)
            trainset, testset = get_folds_train_test_df(df, verbose=False)

            self.fold = fold
            
            self.val_imgs = trainset[self.fold]
            del trainset[self.fold]
            self.train_imgs = []
            for i in trainset: self.train_imgs += i

        else: # tmp: validation split = 50% by default
            all_set = handler.extract_inner_path(handler.images)
            val_split = np.round(val_split * len(all_set)).astype(int)
            if val_split == 0: val_split=1
            self.train_imgs = all_set[val_split:]
            self.val_imgs = all_set[:val_split]
            testset = []
        
        self.train = train
        if self.train:
            print("current fold: {}\n \
            length of the training set: {}\n \
            length of the validation set: {}\n \
            length of the testing set: {}".format(fold, len(self.train_imgs), len(self.val_imgs), len(testset)))

        self.fnames = self.train_imgs if self.train else self.val_imgs
        handler.open(
            img_path = img_path,
            msk_path = msk_path,
            fg_path = fg_path,
            img_inner_paths_list = self.fnames,
            msk_inner_paths_list = self.fnames,
            fg_inner_paths_list = [f[:f.find('.')]+'.pkl' for f in self.fnames],
        )

        # print train and validation image names
        print("{} images: {}".format("Training" if self.train else "Validation", self.fnames))
        
        def generate_data(handler:DataHandler)->list[dict[str,np.ndarray|str]]:
            """Load data, if self.load_data is False, it will only load their path."""
            data=[]
            nonlocal load_data
            if load_data:
                for i,m,f in handler:
                    fg=None
                    # file names
                    img = handler.load(i)[0]
                    msk = handler.load(m)[0]
                    if self.fg_path is not None:
                        fg  = handler.load(f)[0]
                    data += [{'data': img, 'seg': msk, 'loc': fg}]
            else:
                for i,m,f in handler:
                    data += [{'data': i, 'seg': m, 'loc': f}]
            return data

        data = generate_data(handler)
        super(BatchGenDataLoader, self).__init__(
            data,
            batch_size,
            num_threads_in_mt,
        )
        self.indices = np.arange(len(self._data))

        self.current_position = 0
        self.was_initialized = False

    def reset(self)->None:
        """
        Reset the internal state of the batch generator.

        Resets the current position in the epoch and marks the generator as initialized.

        Raises
        ------
        AssertionError
            If `self.indices` is not set.

        Returns
        -------
        None
        """
        assert self.indices is not None

        self.current_position = 0

        self.was_initialized = True

    def get_indices(self)->np.ndarray:
        """
        Retrieve a random batch of indices from the dataset.

        Returns
        -------
        numpy.ndarray
            A NumPy array of randomly sampled indices with shape (batch_size,).

        Raises
        ------
        StopIteration
            If the number of allowed steps per epoch is exceeded.
        """
        if not self.was_initialized:
            self.reset()

        indices = np.random.choice(self.indices, self.batch_size, replace=True)
        
        if self.current_position < self.nbof_steps:
            self.current_position += self.number_of_threads_in_multithreaded
            return indices
        else:
            self.was_initialized=False
            raise StopIteration

    def generate_train_batch(self)->dict:   
        """
        Generate a training batch from the dataset.

        Returns
        -------
        dict
            A dictionary with the following keys:
            - 'data': List of input data arrays for the batch.
            - 'seg': List of corresponding segmentation masks.
            - 'loc': List of class location dictionaries (for foreground sampling, etc).
        """     
        indices = self.get_indices()
        
        batch_list = [self._data[i] for i in indices]
        batch = {
            'data':[data['data'] for data in batch_list],
            'seg': [data['seg'] for data in batch_list],
            'loc': [data['loc'] for data in batch_list],
        }

        return batch

#---------------------------------------------------------------------------
# multi-threading

def get_patch_size(final_patch_size:list[int]| tuple[int]| np.ndarray, 
                   rot_x:float|tuple[float]|list[float], 
                   rot_y:float|tuple[float]|list[float], 
                   rot_z:float|tuple[float]|list[float], 
                   scale_range:tuple[float]|list[float],
                   )->np.ndarray:
    """
    Compute the required patch size to accommodate rotation and scaling augmentations.

    This function determines the maximum patch size needed after applying possible
    rotations and scaling to ensure that the original patch fits entirely within the
    transformed space (i.e., no cropping due to rotation).

    Parameters
    ----------
    final_patch_size : list/tuple/ndarray of int 
        The desired final patch size before any augmentations.
        Should be 2D (for 2D images) or 3D (for volumetric data).

    rot_x : float or tuple/list of float
        Rotation angle(s) in radians around the x-axis.
        If a tuple or list, the maximum absolute value is used.

    rot_y : float or tuple/list of float
        Rotation angle(s) in radians around the y-axis.
        Ignored if input is 2D.

    rot_z : float or tuple/list of float
        Rotation angle(s) in radians around the z-axis.
        Ignored if input is 2D.

    scale_range : tuple or list of float
        Range of possible scaling factors applied during augmentation.
        The minimum value is used to compute the worst-case required patch size.

    Returns
    -------
    final_shape: numpy.ndarray of int
        The adjusted patch size that ensures the transformed patch still contains
        the original field of view, accounting for rotation and scaling.

    Notes
    -----
    - The maximum allowed rotation is clipped to 90° (π/2 radians) for numerical stability.
    - The patch size is increased to accommodate potential rotation "corners"
      that extend beyond the original bounds.
    """
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)

    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)  

def configure_rotation_dummy_da_mirroring_and_inital_patch_size(patch_size:Iterable[int],
                                )->tuple[dict[str, tuple[float, float]], bool, np.ndarray, tuple[int, ...]]:
    """
    Configure rotation parameters, dummy 2D data augmentation, mirroring axes, and compute the initial patch size.

    This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.

    Parameters
    ----------
    patch_size: iterabloe of int
        Patch size as a tuple, array, list,...

    Raises
    ------
    RuntimeError:
        If patch_size not in 2 or 3 dimension

    Returns
    -------
    rotation_for_DA: dict of str to tuple of float
        A rotation for data augmentation.

    do_dummy_2d_data_aug: bool
        Whether a dummy 2d data augmentation has been done

    initial_patch_size : numpy.ndarray
        Path to foregrounds output collection.
    """
    patch_size=np.array(patch_size)
    dim = len(patch_size)
    # TODO rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
    if dim == 2:
        do_dummy_2d_data_aug = False
        # TODO revisit this parametrization
        if max(patch_size) / min(patch_size) > 1.5:
            rotation_for_DA = {
                'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                'y': (0, 0),
                'z': (0, 0)
            }
        else:
            rotation_for_DA = {
                'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                'y': (0, 0),
                'z': (0, 0)
            }
        mirror_axes = (0, 1)
    elif dim == 3:
        # TODO this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
        # order of the axes is determined by spacing, not image size
        do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > 3
        if do_dummy_2d_data_aug:
            # why do we rotate 180 deg here all the time? We should also restrict it
            rotation_for_DA = {
                'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                'y': (0, 0),
                'z': (0, 0)
            }
        else:
            rotation_for_DA = {
                'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            }
        mirror_axes = (0, 1, 2)
    else:
        raise RuntimeError(f"Patch must be of 3 or 2 dimension, found : '{dim}'")

    # TODO this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
    #  old nnunet for now)
    initial_patch_size = get_patch_size(patch_size[-dim:],
                                        *rotation_for_DA.values(),
                                        (0.85, 1.25))
    if do_dummy_2d_data_aug:
        initial_patch_size[0] = patch_size[0]

    print(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')

    return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

class MTBatchGenDataLoader(MultiThreadedAugmenter):
    """
    Multi-threaded data loader for efficient data augmentation and loading.

    :ivar int length: Number of batches.
    """

    def __init__(
        self,
        img_path:str,
        msk_path:str,
        patch_size:Iterable[int],
        batch_size:int, 
        nbof_steps:int,
        fg_path:Optional[str]     = None,
        folds_csv:Optional[str]  = None, 
        fold :int      = 0, 
        val_split:float  = 0.25,
        train:bool      = True,
        load_data:bool = False,
        fg_rate:float     = 0.33,
        num_threads_in_mt:int=12, 
        **kwargs,
    ):
        """
        Multi-threaded data loader for efficient data augmentation and loading.

        Parameters
        ----------
        img_path : str
            Path to a collection containing the images.
        msk_path : str
            Path to a collection containing the masks.
        patch_size : iterable of int
            The size of the patches to be extracted.
        batch_size : int
            Size of the batches.
        nbof_steps : int
            Number of steps per epoch.
        fg_path : str, optional
            Path to a collection containing foreground information. For the moment it is not optional (need to fix that).
        folds_csv : str, optional
            CSV file containing fold information for dataset splitting.
        fold : int, default=0
            Current fold number for training/validation splitting.
        val_split : float, default=0.25
            Proportion of data to be used for validation.
        train : bool, default=True
            If True, use the dataset for training; otherwise, use it for validation.
        load_data : bool, default=False
            If True, loads the entire dataset into computer memory.
        fg_rate : float, default=0.33
            Foreground rate for cropping.
        num_threads_in_mt : int, default=12
            Number of threads in multi-threaded augmentation.
        **kwargs
            Just to handle other parameters.

        Raises
        ------
        ValueError:
            If fg_path is None
        """
        if fg_path is None : raise ValueError("Batchgen module need foregrounds, ensure the preprocessing does it and that the path is included in the config file.")
        gen = BatchGenDataLoader(
            img_path,
            msk_path,
            batch_size,
            nbof_steps,
            fg_path,
            folds_csv, 
            fold,
            val_split,  
            train,
            load_data, 

            # batch_generator parameters
            num_threads_in_mt,
        )

        self.length = nbof_steps

        handler = DataHandlerFactory.get(
            img_path,
            read_only=True,
            msk_path = msk_path,
            fg_path = fg_path,
        )
        
        if train:
            rotation_for_DA, do_dummy_2d_data_aug, aug_patch_size, mirror_axes=configure_rotation_dummy_da_mirroring_and_inital_patch_size(patch_size)
            transform = get_training_transforms(
                                aug_patch_size=aug_patch_size,
                                patch_size=patch_size,
                                fg_rate = fg_rate,
                                rotation_for_DA=rotation_for_DA,
                                deep_supervision_scales=None,
                                mirror_axes=mirror_axes,
                                handler=handler,
                                do_dummy_2d_data_aug=do_dummy_2d_data_aug,
                                use_data_reader=not load_data,
                                )
        else:
            transform = get_validation_transforms(
                                patch_size=patch_size,
                                handler=handler,
                                fg_rate = fg_rate,
                                use_data_reader=not load_data,
                                )
        
        super(MTBatchGenDataLoader, self).__init__(
            gen,
            transform,
            num_threads_in_mt,
            batch_size
        )

    def __len__(self)->int:
        """Return the number of batches in the batch generator."""
        return self.length
    

#---------------------------------------------------------------------------
