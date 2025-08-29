"""
Sampling and data augmentation functions.

Data augmentation not implemented yet...
"""
# TODO: finish this module (or remove)
from typing import Iterable, Optional
import numpy as np
import torchio as tio

# TODO: same code as some function in dataloaders, try to call it to avoid code duplication
def centered_pad(img:np.ndarray, 
                 final_size:Iterable[int], 
                 msk:Optional[np.ndarray]=None,
                 )->np.ndarray|tuple[np.ndarray,np.ndarray]:
    """
    Centered pad an img and msk to fit the final_size.

    Parameters
    ----------
    img: numpy.ndarray
        The image to pad.
    final_size: iterable of int
        The size of the image after the pad.
    msk: numpy.ndarray, optional
        The mask to pad.

    Returns
    -------
    img: numpy.ndarray
        Padded image.
    msk: numpy.ndarray, optional
        Padded mask.
    """
    final_size = np.array(final_size)
    img_shape = np.array(img.shape[1:])
    
    start = (final_size-np.array(img_shape))//2
    start = start * (start > 0)
    end = final_size-(img_shape+start)
    end = end * (end > 0)
    
    pad = np.append([[0,0]], np.stack((start,end),axis=1), axis=0)
    pad_img = np.pad(img, pad, 'constant', constant_values=0)
    if msk is not None: pad_msk = np.pad(msk, pad, 'constant', constant_values=0)
    
    if msk is not None:
        return pad_img, pad_msk
    else: 
        return pad_img

class SmartPatch:
    """
    Randomly crop and resize the images to a certain crop_shape.

    This class provide two functionalities:
    - `global_crop_resize`: method performs a random crop and resize.
    - `local_crop_resize`: performs a second random crop that overlaps with the global one, with a minimum overlap ratio defined by `min_overlap`.

    :ivar numpy.ndarray local_crop_shape: Shape of local crop
    :ivar numpy.ndarray global_crop_shape: Minimal crop size
    :ivar numpy.ndarray | float global_crop_scale: Value between 0 and 1. Factor multiplying (img_shape - global_crop_min_shape) and added to the global_crop_min_shape. A value of 1 means that the maximum shape of the global crop will be the image shape. A value of 0 means that the maximum value will be the global_crop_min_shape. 
    :ivar numpy.ndarray global_crop_shape: shape of local crop
    :ivar numpy.ndarray global_crop_min_shape_scale: Factor multiplying the minimal global_crop_shape, 1.0 is a good default
    :ivar float alpha: 1 - min_overlap; used internally to determine maximum allowed center displacement.
    :ivar ndarra | None global_crop_center: The center coordinates of the global crop, once computed.        
    """

    def __init__(
        self,
        local_crop_shape,
        global_crop_shape,
        min_overlap,
        global_crop_scale=1.0,
        global_crop_min_shape_scale=1.0,
        ):
        """
         Initialize a SmartPatch object.

        Parameters
        ----------
        local_crop_shape : list or tuple of 3 ints
            Shape of the local crop.
        global_crop_shape : list or tuple of 3 ints
            Minimal shape for the global crop.
        min_overlap : float
            Value between 0 and 1. Minimum required overlap between local and global crops.
        global_crop_scale : float or list/tuple of 3 floats, default=1.0
            Scaling factor(s) applied to (image_shape - global_crop_min_shape). Controls how large
            the global crop can be beyond its minimum shape.
            - 1.0 means the crop can reach the full image size.
            - 0.0 means the crop stays at minimal shape.
        global_crop_min_shape_scale : float or list/tuple of 3 floats, default=1.0
            Scaling factor(s) applied to `global_crop_shape` to define the minimum global crop shape.
        """        
        self.local_crop_shape = np.array(local_crop_shape)
        self.global_crop_shape = np.array(global_crop_shape)
        self.global_crop_scale = np.array(global_crop_scale)
        self.global_crop_min_shape_scale = np.array(global_crop_min_shape_scale)
        self.alpha = 1  - min_overlap
        
        # internal arguments
        self.global_crop_center = None
        
    def global_crop_resize(self, img:np.ndarray, 
                           msk:Optional[np.ndarray]=None,
                           )->np.ndarray|tuple[np.ndarray,np.ndarray]:
        """
        Perform a random global crop and resize on the input image (and optional mask).

        The crop shape is randomly selected between a minimum shape (scaled by `global_crop_min_shape_scale`) 
        and a maximum shape controlled by `global_crop_scale` and the image size.

        The crop is then extracted and resized to the fixed `global_crop_shape`.

        Parameters
        ----------
        img : numpy.ndarray
            Input image tensor with shape (C, H, W, D).
        msk : numpy.ndarray, optional
            Optional mask tensor with the same spatial dimensions as img.

        Returns
        -------
        crop_img: numpy.ndarray
            Cropped and resized image.
        crop_msk: numpy.ndarray, optional
            Cropped and resized mask, if `msk` is provided.
        """
        img_shape = np.array(img.shape)[1:]
        
        # determine crop shape
        min_crop_shape = np.round(self.global_crop_shape * self.global_crop_min_shape_scale).astype(int)
        min_crop_shape = np.minimum(min_crop_shape, img_shape)
        crop_shape = np.random.randint(min_crop_shape, (img_shape-min_crop_shape)*self.global_crop_scale+min_crop_shape+1)
        
        # determine crop coordinates
        rand_start = np.random.randint(0, np.maximum(1,img_shape-crop_shape))
        rand_end = crop_shape+rand_start
        
        self.global_crop_center = (rand_end-rand_start)//2 + rand_start
        
        # crop
        crop_img = img[:,
                        rand_start[0]:rand_end[0], 
                        rand_start[1]:rand_end[1], 
                        rand_start[2]:rand_end[2]]
        
        if msk is not None:
            crop_msk = msk[:,
                            rand_start[0]:rand_end[0], 
                            rand_start[1]:rand_end[1], 
                            rand_start[2]:rand_end[2]]
    
        # temp: resize must be done!
        if not np.array_equal(crop_img.shape[1:], self.global_crop_shape):
            if msk is not None:
                sub = tio.Subject(img=tio.ScalarImage(tensor=crop_img), msk=tio.LabelMap(tensor=crop_msk))
                sub = tio.Resize(self.global_crop_shape)(sub)
                crop_img, crop_msk = sub.img.tensor, sub.msk.tensor
            else:
                crop_img = tio.Resize(self.global_crop_shape)(crop_img)
        
        # returns
        if msk is not None:
            return crop_img, crop_msk
        else:
            return crop_img

    def local_crop_pad(self, img:np.ndarray, 
                           msk:Optional[np.ndarray]=None,
                           )->np.ndarray|tuple[np.ndarray,np.ndarray]:
        """
        Perform a local crop centered near the global crop center with padding if needed.

        This method requires `global_crop_resize` to have been called before, so that
        `self.global_crop_center` is defined.

        The local crop overlaps with the global crop by at least the configured minimum overlap.

        Parameters
        ----------
        crop_img : numpy.ndarray
            Input image tensor with shape (C, H, W, D).
        crop_msk : numpy.ndarray, optional
            Optional mask tensor with the same spatial dimensions as img.

        Raises
        ------
        AssertionError
            If `global_crop_resize` has not been called before.

        Returns
        -------
        crop_img: numpy.ndarray
            Cropped and resized image.
        crop_msk: numpy.ndarray, optional
            Cropped and resized mask, if `msk` is provided.
        """
        assert self.global_crop_center is not None, "Error! self.global_crop_resize must be called once before self.local_crop_pad."
        
        img_shape = np.array(img.shape)[1:]
        crop_shape = self.local_crop_shape
        
        # determine crop coordinates
        # we make sure that the crop shape overlap with the global crop shape by at least min_overlap
        centers_max_dist = np.round(crop_shape * self.alpha).astype(np.uint8) + (self.global_crop_shape-crop_shape)//2
        local_center_low = np.maximum(crop_shape//2, self.global_crop_center-centers_max_dist)
        local_center_high = np.minimum(img_shape - crop_shape//2, self.global_crop_center+centers_max_dist)
        local_center_high = np.maximum(local_center_high, local_center_low+1)

        local_crop_center = np.random.randint(low=local_center_low, high=local_center_high)
        
        # local
        start = local_crop_center - (self.local_crop_shape//2)
        start = np.maximum(0,start)
        end = start + self.local_crop_shape
        
        crop_img = img[:,
                    start[0]:end[0], 
                    start[1]:end[1], 
                    start[2]:end[2]]
        
        if msk is not None:
            crop_msk = msk[:,
                        start[0]:end[0], 
                        start[1]:end[1], 
                        start[2]:end[2]]
        
        # pad if needed
        if not np.array_equal(crop_img.shape[1:], self.local_crop_shape):
            if msk is not None:
                crop_img, crop_msk = centered_pad(img=crop_img, final_size=self.local_crop_shape, msk=crop_msk)
            else:
                crop_img = centered_pad(img=crop_img, final_size=self.local_crop_shape)
        
        # returns
        if msk is not None:
            return crop_img, crop_msk
        else:
            return crop_img

    def local_crop_resize(self,
                          img: np.ndarray,
                          msk: Optional[np.ndarray] = None,
                          ) -> np.ndarray| tuple[np.ndarray, np.ndarray]:
        """
        Perform a local crop with random size and resize, overlapping the global crop.

        This method requires `global_crop_resize` to have been called before, so that
        `self.global_crop_center` is defined.

        The crop size is randomly selected within `self.local_crop_scale` fraction of the image size,
        and positioned to ensure a minimum overlap with the global crop.

        Parameters
        ----------
        img : numpy.ndarray
            Input image tensor with shape (C, H, W, D).
        msk : numpy.ndarray, optional
            Optional mask tensor with the same spatial dimensions as img.

        Raises
        ------
        AssertionError
            If `global_crop_resize` has not been called before.

        Returns
        -------
        crop_img : numpy.ndarray
            Input image tensor with shape (C, H, W, D).
        crop_msk : numpy.ndarray, optional
            Optional mask tensor with the same spatial dimensions as img.
        """
        assert self.global_crop_center is not None, "Error! self.global_crop_resize must be called once before self.local_crop_resize."

        img_shape = np.array(img.shape)[1:]

        # determine crop shape
        crop_shape = np.random.randint(self.local_crop_scale[0] * img_shape, self.local_crop_scale[1] * img_shape+1)
        
        # determine crop coordinates
        # we make sure that the crop shape overlap with the global crop shape by at least min_overlap
        centers_max_dist = np.round(crop_shape * self.alpha).astype(np.uint8) + (self.global_crop_shape-crop_shape)//2
        local_center_low = np.maximum(crop_shape//2, self.global_crop_center-centers_max_dist)
        local_center_high = np.minimum(img_shape - crop_shape//2, self.global_crop_center+centers_max_dist)
        local_center_high = np.maximum(local_center_high, local_center_low+1)

        local_crop_center = np.random.randint(low=local_center_low, high=local_center_high)
        
        start = local_crop_center - (self.local_crop_shape//2)
        start = np.maximum(0,start)
        end = start + self.local_crop_shape
        
        crop_img = img[:,
                    start[0]:end[0], 
                    start[1]:end[1], 
                    start[2]:end[2]]
        
        if msk is not None:
            crop_msk = msk[:,
                        start[0]:end[0], 
                        start[1]:end[1], 
                        start[2]:end[2]]
        
        # resize if needed
        if not np.array_equal(crop_img.shape[1:], self.local_crop_shape):
            if msk is not None:
                sub = tio.Subject(img=tio.ScalarImage(tensor=crop_img), msk=tio.LabelMap(tensor=crop_msk))
                sub = tio.Resize(self.global_crop_shape)(sub)
                crop_img, crop_msk = sub.img.tensor, sub.msk.tensor
            else:
                crop_img = tio.Resize(self.global_crop_shape)(crop_img)
        
        # returns
        if msk is not None:
            return crop_img, crop_msk
        else:
            return crop_img