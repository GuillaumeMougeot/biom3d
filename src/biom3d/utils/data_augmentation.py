import numpy as np
import torchio as tio


# ----------------------------------------------------------------------------
# data augmentation utils
# not used yet...
def centered_pad(img, final_size, msk=None):
    """
    centered pad an img and msk to fit the final_size
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
    The global_crop_resize method performs a random crop and resize.
    The local_crop_resize method performs a random crop and resize making sure that the crop 
    is overlapping (to a certain extent, defined by the min_overlap parameter) with the global
    crop previously performed. 
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
        Parameters
        ----------
        global_crop_shape : list or tuple of size == 3
            Minimal crop size
        global_crop_scale : float, default=1.0
            Value between 0 and 1. Factor multiplying (img_shape - global_crop_min_shape) and added to the global_crop_min_shape. A value of 1 means that the maximum shape of the global crop will be the image shape. A value of 0 means that the maximum value will be the global_crop_min_shape. 
        global_crop_min_shape_factor : float, default=1.0
            (DEPRECATED?) Factor multiplying the minimal global_crop_shape, 1.0 is a good default
        
        """
        
        self.local_crop_shape = np.array(local_crop_shape)
        self.global_crop_shape = np.array(global_crop_shape)
        self.global_crop_scale = np.array(global_crop_scale)
        self.global_crop_min_shape_scale = np.array(global_crop_min_shape_scale)
        self.alpha = 1  - min_overlap
        
        # internal arguments
        self.global_crop_center = None
        
    def global_crop_resize(self, img, msk=None):
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

    def local_crop_pad(self, img, msk=None):
        """
        global_crop_resize must be called at least once before calling local_crop_pad
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

    def local_crop_resize(self, img, msk=None):
        """
        global_crop_resize must be called at least once before calling local_crop_resize
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