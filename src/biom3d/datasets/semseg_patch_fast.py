#---------------------------------------------------------------------------
# Dataset primitives for 3D segmentation dataset
# solution: patch approach with the whole dataset into memory 
#---------------------------------------------------------------------------

from json import load
import os
import numpy as np 
import torchio as tio
import random 
import torch
from torch.utils.data import Dataset, DataLoader
# from monai.data import CacheDataset
import pandas as pd 
from tqdm import tqdm 
from skimage.io import imread

from biom3d.utils import centered_pad, get_folds_train_test_df

#---------------------------------------------------------------------------
# utilities to random crops
    
# create a patch centered on this voxel
def centered_crop(img, msk, center, crop_shape):
    """
    centered crop a portion of size prop of the original image size.
    """
    crop_shape=np.array(crop_shape)
    start = center-crop_shape//2
    end = crop_shape+start

    crop_img = img[:,
                start[0]:end[0], 
                start[1]:end[1], 
                start[2]:end[2]]
    crop_msk = msk[:,
                start[0]:end[0], 
                start[1]:end[1], 
                start[2]:end[2]]
    return crop_img, crop_msk

def foreground_crop(img, msk, foreground, crop_shape, margin=np.zeros(3)):
    """
    crop the img and msk so that the foreground voxel is located somewhere (random) in the cropped patch
    the margin argument adds a margin to force the foreground voxel to be located nearer the middle of the patch 
    """
    lower_bound = np.maximum(0,foreground-np.array(crop_shape)+margin)
    start = np.random.randint(low=lower_bound, high=np.maximum(lower_bound+1,foreground-margin))
    end = start+crop_shape
    
    crop_img = img[:,
                start[0]:end[0], 
                start[1]:end[1], 
                start[2]:end[2]]
    crop_msk = msk[:,
                start[0]:end[0], 
                start[1]:end[1], 
                start[2]:end[2]]
    return crop_img, crop_msk

def random_crop(img, msk, crop_shape):
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
    crop_msk = msk[:,
                    rand_start[0]:rand_end[0], 
                    rand_start[1]:rand_end[1], 
                    rand_start[2]:rand_end[2]]
    return crop_img, crop_msk

# def centered_pad(img, msk, final_size):
#     """
#     centered pad an img and msk to fit the final_size
#     """
#     final_size = np.array(final_size)
#     img_shape = np.array(img.shape[1:])
    
#     start = (final_size-np.array(img_shape))//2
#     end = final_size-(img_shape+start)
#     end = end * (end > 0)
    
#     pad = np.append([[0,0]], np.stack((start,end),axis=1), axis=0)
#     pad_img = np.pad(img, pad, 'constant', constant_values=0)
#     pad_msk = np.pad(msk, pad, 'constant', constant_values=0)
    
# #     if ((final_size-np.array(pad_img.shape)) < 0).any(): # keeps only the negative values
# #         pad_img = pad_img[:,:final_size[0],:final_size[1],:final_size[2]]
# #         pad_msk = pad_msk[:,:final_size[0],:final_size[1],:final_size[2]]
#     return pad_img, pad_msk

# def random_pad(img, msk, final_size):
#     """
#     [CAREFUL!] I THINK THIS FUNCTION HAS SOME BUGS (WITH SMALL IMAGES)
#     randomly pad an image with zeros to reach the final size. 
#     if the image is bigger than the expected size, then the image is cropped.
#     """
#     img_shape = np.array(img.shape)[1:]
#     size_range = (final_size-img_shape) * (final_size-img_shape > 0) # needed if the original image is bigger than the final one
#     # rand_start = np.array([random.randint(0,c) for c in size_range])
#     rand_start = np.random.randint(0,np.maximum(1,size_range))

#     rand_end = final_size-(img_shape+rand_start)
#     rand_end = rand_end * (rand_end > 0)

#     pad = np.append([[0,0]],np.vstack((rand_start, rand_end)).T,axis=0)
#     pad_img = np.pad(img, pad, 'constant', constant_values=0)
#     pad_msk = np.pad(msk, pad, 'constant', constant_values=0)
#     # pad_img = torch.nn.functional.pad(img, tuple(pad.flatten().tolist()), 'constant', value=0)

#     # crop the image if needed
#     if ((final_size-np.array(pad_img.shape)) < 0).any(): # keeps only the negative values
#         pad_img = pad_img[:,:final_size[0],:final_size[1],:final_size[2]]
#         pad_msk = pad_msk[:,:final_size[0],:final_size[1],:final_size[2]]
#     return pad_img, pad_msk

def random_crop_pad(img, msk, final_size, fg_rate=0.33, fg_margin=np.zeros(3), remove_bg=False):
    """
    random crop and pad if needed.
    """
    # choose if using foreground centrered or random alignement
    force_fg = random.random()
    if fg_rate>0 and force_fg<fg_rate:
        start_rnd = 1 if remove_bg else 0
        rnd_label = random.randint(start_rnd,msk.shape[0]-1) # choose a random label
        
        locations = np.argwhere(msk[rnd_label] == 1)
        
        if locations.size==0: # bug fix when having empty arrays 
            img, msk = random_crop(img, msk, final_size)
        else:
            center=random.choice(locations) # choose a random voxel of this label
            img, msk = foreground_crop(img, msk, center, final_size, fg_margin)
    else:
        # or random crop
        img, msk = random_crop(img, msk, final_size)
        
    # pad if needed
    if np.any(np.array(img.shape)[1:]-final_size)!=0:
        img, msk = centered_pad(img=img, msk=msk, final_size=final_size)
    return img, msk

def random_crop_resize(img, msk, crop_scale, final_size, fg_rate=0.33, fg_margin=np.zeros(3), remove_bg=False):
    """
    random crop and resize if needed.
    Args:
    crop_scale: >=1
    """
    final_size = np.array(final_size)
        
    # determine crop shape
    max_crop_shape = np.round(final_size * crop_scale).astype(int)
    crop_shape = np.random.randint(final_size, max_crop_shape+1)
    
    # choose if using foreground centrered or random alignement
    force_fg = random.random()
    if fg_rate>0 and force_fg<fg_rate:
        start_rnd = 1 if remove_bg else 0
        rnd_label = random.randint(start_rnd,msk.shape[0]-1) # choose a random label
        
        locations = np.argwhere(msk[rnd_label] == 1)
        
        if locations.size==0: # bug fix when having empty arrays 
            img, msk = random_crop(img, msk, crop_shape)
        else:
            center=random.choice(locations) # choose a random voxel of this label
            img, msk = foreground_crop(img, msk, center, crop_shape, fg_margin)
    else:
        # or random crop
        img, msk = random_crop(img, msk, crop_shape)
    
    # resize if needed
    if np.any(np.array(img.shape)[1:]-final_size)>0:
        sub = tio.Subject(img=tio.ScalarImage(tensor=img), msk=tio.LabelMap(tensor=msk))
        sub = tio.Resize(final_size)(sub)
        img, msk = sub.img.tensor, sub.msk.tensor
    elif np.any(np.array(img.shape)[1:]-final_size)<0:
        img, msk = centered_pad(img, msk, final_size)
    return img, msk

#---------------------------------------------------------------------------

class SemSeg3DPatchFast(Dataset):
    """
    with DataLoader
    """
    def __init__(
        self,
        img_dir,
        msk_dir,
        batch_size, 
        patch_size,
        nbof_steps,
        folds_csv  = None, 
        fold       = 0, 
        val_split  = 0.25,
        train      = True,
        use_aug    = True,
        aug_patch_size = None,
        fg_rate = 0.33, # if > 0, force the use of foreground, needs to run some pre-computations (note: better use the foreground scheduler)
        crop_scale = 1.0, # if > 1, then use random_crop_resize instead of random_crop_pad
        use_softmax = True, # if true, means that the output is one_hot encoded for softmax use
        load_data = False, # if True, loads the all dataset into computer memory (faster but more memory expensive)
        ):

        self.img_dir = img_dir
        self.msk_dir = msk_dir

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.aug_patch_size = aug_patch_size

        self.nbof_steps = nbof_steps

        self.load_data = load_data
        
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
            all_set = os.listdir(img_dir)
            val_split = np.round(val_split * len(all_set)).astype(int)
            if val_split == 0: val_split=1
            self.train_imgs = all_set[val_split:]
            self.val_imgs = all_set[:val_split]
            testset = []
        
        if self.load_data:
            print("Loading the whole dataset into computer memory...")
            def load_data(imgs_fnames):
                imgs_data = []
                msks_data = []
                for idx in range(len(imgs_fnames)):
                    # file names
                    img_path = os.path.join(self.img_dir, imgs_fnames[idx])
                    msk_path = os.path.join(self.msk_dir, imgs_fnames[idx])

                    # load img and msks
                    imgs_data += [imread(img_path)]
                    msks_data += [imread(msk_path)]
                return imgs_data, msks_data

            self.train_imgs_data, self.train_msks_data = load_data(self.train_imgs)
            self.val_imgs_data, self.val_msks_data = load_data(self.val_imgs)
            print("Done!")
            
        self.train = train
        print("current fold: {}\n \
        length of the training set: {}\n \
        length of the validation set: {}\n \
        length of the testing set: {}\n \
        is training mode active?: {}".format(fold, len(self.train_imgs), len(self.val_imgs), len(testset), self.train))

        # print train and validation image names
        print("Training images:", self.train_imgs)
        print("Validation images:", self.val_imgs)

        self.use_aug = use_aug

        if self.use_aug:
            ps = np.array(self.patch_size)

            # [aug] flipping probabilities
            flip_prop=ps.min()/ps
            flip_prop/=flip_prop.sum()

            # [aug] 'axes' for tio.RandomAnisotropy
            anisotropy_axes=tuple(np.arange(3)[ps/ps.min()>3].tolist())

            # [aug] 'degrees' for tio.RandomAffine
            if np.any(ps/ps.min()>3): # then use dummy_2d
                norm = ps*3/ps.min()
                softmax=np.exp(norm)/sum(np.exp(norm))
                degrees=softmax.min()*90/softmax
                # degrees = 180*ps.min()/ps
            else:
                degrees = (-30,30)

            # [aug] 'cropping'
            # the affine transform is computed on bigger patches than the other transform
            # that's why we need to crop the patch after potential affine augmentation
            start = (np.array(self.aug_patch_size)-np.array(self.patch_size))//2
            end = self.aug_patch_size-(np.array(self.patch_size)+start)
            cropping = (start[0],end[0],start[1],end[1],start[2],end[2])
            
            # the foreground-crop-function forces the foreground to be in the center of the patch
            # so that, when doing the second centrering crop, the foreground is still present in the patch,
            # that's why there is a margin here
            self.fg_margin = start 


            self.transform = tio.Compose([
                # spatial augmentations
                # tio.RandomFlip(p=0.8, axes=(0), flip_probability=flip_prop[0]),
                # tio.RandomFlip(p=0.8, axes=(1), flip_probability=flip_prop[1]),
                # tio.RandomFlip(p=0.8, axes=(2), flip_probability=flip_prop[2]),
                # tio.RandomFlip(p=0.2, axes=(0,1,2)),
                # tio.RandomAnisotropy(p=0.25, axes=anisotropy_axes, downsampling=(1,2)),
                tio.RandomAffine(p=0.25, scales=(0.7,1.4), degrees=degrees, translation=0),
                tio.Crop(p=1, cropping=cropping),

                tio.RandomFlip(p=1, axes=(0,1,2)),
                # tio.RandomElasticDeformation(p=0.2, num_control_points=4, locked_borders=1),
                # tio.OneOf({
                #     tio.RandomAffine(scales=0.1, degrees=10, translation=0): 0.8,
                #     tio.RandomElasticDeformation(): 0.2,
                # }),
                

                # intensity augmentations
                # tio.RandomMotion(p=0.2),
                # tio.RandomGhosting(p=0.2),
                # tio.RandomSpike(p=0.15),
                tio.RandomBiasField(p=0.15, coefficients=0.2),
                tio.RandomBlur(p=0.2, std=(0.5,1.5)),
                tio.RandomNoise(p=0.2, std=(0,0.1)),
                # tio.RandomSwap(p=0.2, patch_size=ps//8),
                tio.RandomGamma(p=0.3, log_gamma=(-0.35,0.4)),
            ])

        # self.fg_rate = fg_rate if self.train else 1
        self.fg_rate = fg_rate
        self.use_softmax = use_softmax
        self.crop_scale = crop_scale 
        assert self.crop_scale >= 1, "[Error] crop_scale must be higher or equalt to 1"
    
    def set_fg_rate(self,value):
        """
        setter function for the foreground rate class parameter
        """
        self.fg_rate = value
    
    def __len__(self):
        # return len(self.train_imgs) if self.training else len(self.val_imgs)
        return self.nbof_steps*self.batch_size
    
    def __getitem__(self, idx):

        if self.load_data:
            if self.train:
                img = self.train_imgs_data[idx%len(self.train_imgs_data)]
                msk = self.train_msks_data[idx%len(self.train_msks_data)]
            else: 
                img = self.val_imgs_data[idx%len(self.val_imgs_data)]
                msk = self.val_msks_data[idx%len(self.val_msks_data)]
        else:
            fnames = self.train_imgs if self.train else self.val_imgs
            # img_fname = np.random.choice(fnames)
            img_fname = fnames[idx%len(fnames)]
            
            # file names
            img_path = os.path.join(self.img_dir, img_fname)
            msk_path = os.path.join(self.msk_dir, img_fname)

            # read the images
            img = imread(img_path)
            msk = imread(msk_path)

        # remove bg channel if use_softmax is False
        if not self.use_softmax:
            msk = msk[1:]

        # random crop and pad
        final_size = self.aug_patch_size if self.use_aug else self.patch_size
        fg_margin = self.fg_margin if self.use_aug else np.zeros(3)
        if self.train and self.crop_scale > 1:
            img, msk = random_crop_resize(
                img,
                msk,
                crop_scale=self.crop_scale,
                final_size=final_size,
                fg_rate=self.fg_rate,
                fg_margin=fg_margin,
                remove_bg=self.use_softmax,
                )
        else:
            img, msk = random_crop_pad(
                img,
                msk,
                final_size=final_size,
                fg_rate=self.fg_rate,
                fg_margin=fg_margin,
                remove_bg=self.use_softmax,
                )

        # data augmentation
        if self.use_aug:
            sub = tio.Subject(img=tio.ScalarImage(tensor=img), msk=tio.LabelMap(tensor=msk))
            sub = self.transform(sub)
            img, msk = sub.img.tensor, sub.msk.tensor
        
            # to float for msk
            msk = msk.float()

        elif not self.use_softmax:
            msk = msk.astype(float)
        
        return img, msk

#---------------------------------------------------------------------------