
#---------------------------------------------------------------------------
# Dataset primitives for 3D segmentation dataset
# solution: patch approach with the whole dataset into memory 
#---------------------------------------------------------------------------

import os
import numpy as np 
import torchio as tio
from torch.utils.data import Dataset
import pandas as pd 
from skimage.io import imread

from biom3d.utils import get_folds_train_test_df


#---------------------------------------------------------------------------
# utilities to random crops

def random_crop(img, crop_shape):
    """
    randomly crop a portion of size prop of the original image size.
    """
    img_shape = np.array(img.shape)[1:]
    # rand_start = np.array([random.randint(0,c) for c in np.maximum(0,(img_shape-crop_shape))])
    rand_start = np.random.randint(0, np.maximum(1,img_shape-crop_shape))
    rand_end = crop_shape+rand_start

    crop_img = img[:,
                    rand_start[0]:rand_end[0], 
                    rand_start[1]:rand_end[1], 
                    rand_start[2]:rand_end[2]]
    return crop_img

def random_crop_resize(img, scale, final_size):
    """
    crop the image randomly first and the resize the crop to the final_size.
    the first crop is performed on the original image 'img' with a scale defined by 'scale'. 
    The 'scale' size is computed using the original image 'img' shape.
    """
    original_shape = np.array(img.shape[1:])
    scale = np.array(scale)
    crop_shape = np.random.randint(low=scale[0]*original_shape, high=scale[1]*original_shape+1)
    crop_img = random_crop(img, crop_shape=crop_shape)
    resize_img = tio.Resize(final_size)(crop_img)
    return resize_img

# def random_crop_resize_local(img, scale, local_final_size, ):
    
class RandomCropResize:
    """
    Randomly crop and resize the images to a certain crop_shape.
    The global_crop_resize method performs a random crop and resize.
    The local_crop_resize method performs a random crop and resize making sure that the crop 
    is overlapping (to a certain extent, defined by the min_overlap parameter) with the global
    crop previously performed. 
    
    Args:
        local_crop_shape, 
        local_crop_scale, 
        global_crop_shape, 
        global_crop_scale, 
        min_overlap, minimal overlap between local and global crop 
                     (1:local is fully included inside global, <=0:local can be outside global)
    """
    def __init__(
        self,
        local_crop_shape,
        local_crop_scale,
        global_crop_shape,
        global_crop_scale,
        min_overlap,
        ):
        
        self.local_crop_shape = np.array(local_crop_shape)
        self.local_crop_scale = np.array(local_crop_scale)
        self.global_crop_shape = np.array(global_crop_shape)
        self.global_crop_scale = np.array(global_crop_scale)
        self.alpha = 1  - min_overlap
        
        # internal arguments
        self.global_crop_center = None
        
    def global_crop_resize(self, img):
        img_shape = np.array(img.shape)[1:]
        crop_shape = np.random.randint(self.global_crop_scale[0] * img_shape, self.global_crop_scale[1] * img_shape+1)
                
        rand_start = np.random.randint(0, np.maximum(1,img_shape-crop_shape))
        rand_end = crop_shape+rand_start
        
        self.global_crop_center = (rand_end-rand_start)//2 + rand_start
        
        crop_img = img[:,
                        rand_start[0]:rand_end[0], 
                        rand_start[1]:rand_end[1], 
                        rand_start[2]:rand_end[2]]
    
        # temp: resize must be done!
        if not np.array_equal(crop_img.shape[1:], self.global_crop_shape):
            crop_img = tio.Resize(self.global_crop_shape)(crop_img)
        
        return crop_img

    def local_crop_resize(self, img):
        """
        global_crop_resize must be called at least once before calling local_crop_resize
        """
        if self.global_crop_center is None:
            print("Error! self.global_crop_resize must be called once before self.local_crop_resize.")
            return img
        
        img_shape = np.array(img.shape)[1:]
        crop_shape = np.random.randint(self.local_crop_scale[0] * img_shape, self.local_crop_scale[1] * img_shape+1)

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
        
        # temp: resize must be done!
        if not np.array_equal(crop_img.shape[1:], self.local_crop_shape):
            crop_img = tio.Resize(self.local_crop_shape)(crop_img)
        
        return crop_img

#---------------------------------------------------------------------------

class Dino(Dataset):
    """
    with DataLoader
    """
    def __init__(
        self,
        img_dir,
        batch_size, 
        global_patch_size,
        local_patch_size,
        nbof_steps,
        nbof_global_patch =2, 
        nbof_local_patch =8, # must be dividable by nbof_global_patch
        folds_csv  = None, 
        fold       = 0, 
        val_split  = 0.25,
        train      = True,
        use_aug    = False,
        ):

        self.img_dir = img_dir

        self.batch_size = batch_size
        self.global_patch_size = global_patch_size
        self.local_patch_size = local_patch_size
        self.nbof_local_patch = nbof_local_patch
        self.nbof_global_patch = nbof_global_patch
        
        ps = np.array(self.global_patch_size)
        max_dim = 0.6
        min_dim = 0.2
        self.global_patch_scale = (np.where(ps.max()/ps<3, max_dim, 1.0), np.ones(3))
        self.local_patch_scale = (np.where(ps.max()/ps<3, min_dim, max_dim), np.where(ps.max()/ps<3, max_dim, 1.0))
        
        self.nbof_steps = nbof_steps

        
        # get the training and validation names 
        if folds_csv is not None:
            df = pd.read_csv(folds_csv)
            trainset, _ = get_folds_train_test_df(df, verbose=False)

            self.fold = fold
            
            self.val_imgs = trainset[self.fold]
            del trainset[self.fold]
            self.train_imgs = []
            for i in trainset: self.train_imgs += i

        else: # tmp: validation split = 50% by default
            all_set = os.listdir(img_dir)
            val_split = np.round(val_split * len(all_set)).astype(int)
            if val_split == 0: val_split=1
            self.train_imgs = all_set[val_split:]
            self.val_imgs = all_set[:val_split]
            
        self.train = train
        print("current fold: {}\n \
        length of the training set: {}\n \
        length of the validation set: {}\n \
        is training mode active?: {}".format(fold, len(self.train_imgs), len(self.val_imgs), self.train))

        self.use_aug = use_aug

        # data augmentation
        if self.use_aug:
            self.transform = tio.Compose([
                tio.RandomFlip(p=1, axes=(0,1,2)),


                # intensity augmentations
                tio.RandomBiasField(p=0.15, coefficients=0.2),
                tio.RandomBlur(p=0.2, std=(0.5,1.5)),
                tio.RandomNoise(p=0.2, std=(0,0.1)),
                # tio.RandomSwap(p=0.2, patch_size=ps//8),
                tio.RandomGamma(p=0.3, log_gamma=(-0.35,0.4)),
            ])
            
        self.crop_resize = RandomCropResize(
            local_crop_shape=self.local_patch_size,
            local_crop_scale=self.local_patch_scale,
            global_crop_shape=self.global_patch_size,
            global_crop_scale=self.global_patch_scale,
            min_overlap=0.7,
        )
    
    def __len__(self):
        # return len(self.train_imgs) if self.training else len(self.val_imgs)
        return self.nbof_steps*self.batch_size
    
    def __getitem__(self, idx):
        
        # load the image
        # crop a big crop
        # crop several small crops
        
        length = len(self.train_imgs) if self.train else len(self.val_imgs)
        fnames = self.train_imgs if self.train else self.val_imgs
        img_fname = fnames[idx%length]

        # file names
        img_path = os.path.join(self.img_dir, img_fname)

        # read the images
        img = imread(img_path)

        # expand dim if needed
        if len(img.shape)==3:
            img = np.expand_dims(img, 0)
            

        # random crop and pad
        crops = []
        # crops.append(random_crop_resize(img, scale=(0.7,1.0), final_size=self.global_patch_size))
        # crops.append(random_crop_resize(img, scale=self.global_patch_scale, final_size=self.global_patch_size))
        for i in range(self.nbof_global_patch): 
            crops.insert(i, self.crop_resize.global_crop_resize(img))
            for _ in range(self.nbof_local_patch//self.nbof_global_patch):
                crops.append(self.crop_resize.local_crop_resize(img))
        
        

        # data augmentation
        if self.use_aug:
            crops = [self.transform(c) for c in crops]

        return crops

#---------------------------------------------------------------------------