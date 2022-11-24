
#---------------------------------------------------------------------------
# Arcface dataset
#---------------------------------------------------------------------------

import os
import numpy as np 
import torchio as tio
from torch.utils.data import Dataset
# from monai.data import CacheDataset
import pandas as pd 
from skimage.io import imread

from biom3d.utils import get_folds_train_test_df, RandomCropResize

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

def random_pad(img, final_size):
    """
    randomly pad an image with zeros to reach the final size. 
    if the image is bigger than the expected size, then the image is cropped.
    """
    img_shape = np.array(img.shape)[1:]
    size_range = (final_size-img_shape) * (final_size-img_shape > 0) # needed if the original image is bigger than the final one
    # rand_start = np.array([random.randint(0,c) for c in size_range])
    rand_start = np.random.randint(0,np.maximum(1,size_range))

    rand_end = final_size-(img_shape+rand_start)
    rand_end = rand_end * (rand_end > 0)

    pad = np.append([[0,0]],np.vstack((rand_start, rand_end)).T,axis=0)
    pad_img = np.pad(img, pad, 'constant', constant_values=0)
    # pad_img = torch.nn.functional.pad(img, tuple(pad.flatten().tolist()), 'constant', value=0)

    # crop the image if needed
    if ((final_size-img_shape) < 0).any(): # keeps only the negative values
        pad_img = pad_img[:,:final_size[0],:final_size[1],:final_size[2]]
    return pad_img

def centered_pad(img, msk, final_size):
    """
    centered pad an img and msk to fit the final_size
    """
    final_size = np.array(final_size)
    img_shape = np.array(img.shape[1:])
    
    start = (final_size-np.array(img_shape))//2
    end = final_size-(img_shape+start)
    end = end * (end > 0)
    
    pad = np.append([[0,0]], np.stack((start,end),axis=1), axis=0)
    pad_img = np.pad(img, pad, 'constant', constant_values=0)
    pad_msk = np.pad(msk, pad, 'constant', constant_values=0)
    
#     if ((final_size-np.array(pad_img.shape)) < 0).any(): # keeps only the negative values
#         pad_img = pad_img[:,:final_size[0],:final_size[1],:final_size[2]]
#         pad_msk = pad_msk[:,:final_size[0],:final_size[1],:final_size[2]]
    return pad_img, pad_msk

def random_crop_pad(img, final_size):
    """
    random crop and pad if needed.
    """
    # or random crop
    img = random_crop(img, final_size)

    # pad if needed
    if (np.array(img.shape)[1:]-final_size).sum()!=0:
        img = centered_pad(img, final_size)
    return img

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

#---------------------------------------------------------------------------

class ArcFace(Dataset):
    """
    with DataLoader
    """
    def __init__(
        self,
        img_dir,
        batch_size, 
        patch_size,
        nbof_steps,
        folds_csv  = None, 
        fold       = 0, 
        val_split  = 0.25,
        train      = True,
        use_aug    = True,
        aug_patch_size = None,
        initial_global_crop_scale = 1.0,
        ):

        self.img_dir = img_dir

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.aug_patch_size = aug_patch_size

        self.nbof_steps = nbof_steps
        
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
            # if val_split == 0: val_split=1
            self.train_imgs = all_set[val_split:]
            self.val_imgs = all_set[:val_split]
            testset = []
        
            
        self.train = train
        print("current fold: {}\n \
        length of the training set: {}\n \
        length of the validation set: {}\n \
        length of the testing set: {}\n \
        is training mode active?: {}".format(fold, len(self.train_imgs), len(self.val_imgs), len(testset), self.train))

        self.set_dataset_size(len(self.train_imgs) if self.train else len(self.val_imgs))
        # self.set_dataset_size(5)

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

        self.crt_global_crop_scale = initial_global_crop_scale
        self.set_global_crop(self.crt_global_crop_scale)
    
    def set_dataset_size(self, dataset_size):
        """
        set the number of images in the dataset. It is useful for the dataset size scheduler.
        the formula is: dataset_size = min(epoch + 2, len(fnames))
        """
        self.dataset_size = dataset_size
    
    def set_global_crop(self, scale=None):

        self.crop_resize = RandomCropResize(
            local_crop_shape=self.aug_patch_size if self.use_aug else self.patch_size,
            global_crop_shape=self.aug_patch_size if self.use_aug else self.patch_size,
            global_crop_scale=self.crt_global_crop_scale if scale is None else scale,
            min_overlap=1.0, # can be anything as local_patching is not used here
        )
        self.crt_global_crop_scale=self.crt_global_crop_scale if scale is None else scale
    
    def __len__(self):
        # return len(self.train_imgs) if self.training else len(self.val_imgs)
        return self.nbof_steps*self.batch_size
    
    def __getitem__(self, idx):
        
        fnames = self.train_imgs if self.train else self.val_imgs
        # img_fname = np.random.choice(fnames)
        img_fname = fnames[idx%self.dataset_size]

        # file names
        img_path = os.path.join(self.img_dir, img_fname)

        # read the images
        img = imread(img_path)

        # random crop and pad
        # final_size = self.aug_patch_size if self.use_aug else self.patch_size
        # img = random_crop_pad(img,final_size=final_size)
        # img = random_crop_resize(img, scale=(0.7,1.0), final_size=final_size)
        img = self.crop_resize.global_crop_resize(img)

        # data augmentation
        if self.use_aug:
            img = self.transform(img)

        # print("arcface, datasetset_size", idx%self.dataset_size)
        
        return img, idx%self.dataset_size

#---------------------------------------------------------------------------