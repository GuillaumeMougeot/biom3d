#---------------------------------------------------------------------------
# Dataset primitives for 3D segmentation dataset
# solution: patch approach with the whole dataset into memory 
#---------------------------------------------------------------------------

import os
import pickle
import numpy as np 
import torchio as tio
import random 
from torch.utils.data import Dataset
# from monai.data import CacheDataset
import pandas as pd 
# from tifffile import imrea
from biom3d.utils import centered_pad, get_folds_train_test_df, DataHandlerFactory

#---------------------------------------------------------------------------
# utilities to random crops

def centered_crop(img, msk, center, crop_shape, margin=np.zeros(3)):
    """Do a crop, forcing the location voxel to be located in the center of the crop.
    """
    img_shape = np.array(img.shape)[1:]
    center = np.array(center)
    assert np.all(center>=0) and np.all(center<img_shape), "[Error] Center must be located inside the image. Center: {}, Image shape: {}".format(center, img_shape)
    crop_shape = np.array(crop_shape)
    margin = np.array(margin)
    
    # middle of the crop
    start = (center-crop_shape//2+margin).astype(int)

    # assert that the end will not be out of the crop
    end = start+crop_shape

    # we make sure that we are not out of the image shape
    start = np.maximum(0,start)
    
    idx = [slice(0,img.shape[0])]+list(slice(s[0], s[1]) for s in zip(start, end))
    idx = tuple(idx)
    
    crop_img = img[idx]
    crop_msk = msk[idx]
    return crop_img, crop_msk

def located_crop(img, msk, location, crop_shape, margin=np.zeros(3)):
    """Do a crop, forcing the location voxel to be located in the crop.
    """
    img_shape = np.array(img.shape)[1:]
    location = np.array(location)
    crop_shape = np.array(crop_shape)
    margin = np.array(margin)
    
    lower_bound = np.maximum(0,location-crop_shape+margin)
    higher_bound = np.maximum(lower_bound+1,np.minimum(location-margin, img_shape-crop_shape))
    start = np.random.randint(low=lower_bound, high=np.maximum(lower_bound+1,higher_bound))
    end = start+crop_shape
    
    idx = [slice(0,img.shape[0])]+list(slice(s[0], s[1]) for s in zip(start, end))
    idx = tuple(idx)
    
    crop_img = img[idx]
    crop_msk = msk[idx]
    return crop_img, crop_msk

def foreground_crop(img, msk, final_size, fg_margin, fg=None, use_softmax=True):
    """Do a foreground crop.
    """
    if fg is not None and len(list(fg.keys()))>0:
        locations = fg[random.choice(list(fg.keys()))]
    else:
        if tuple(msk.shape)[0]==1:
            # then we consider that we don't have a one hot encoded label
            rnd_label = random.randint(1,msk.max() if msk.max()>0 else 1)
            locations = np.argwhere(msk[0] == rnd_label)
        else:
            # then we have a one hot encoded label
            rnd_label = random.randint(int(use_softmax),tuple(msk.shape)[0]-1)
            locations = np.argwhere(msk[rnd_label] == 1)

    if np.array(locations).size==0: # bug fix when having empty arrays 
        img, msk = random_crop(img, msk, final_size, force_in=False)
    else:
        center=random.choice(locations) # choose a random voxel of this label
        img, msk = centered_crop(img, msk, center, final_size, fg_margin)
    return img, msk

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
    
    if msk is not None:
        pad_msk = np.pad(msk, pad, 'constant', constant_values=0)
        return pad_img, pad_msk
    else: 
        return pad_img

def random_crop(img, msk, crop_shape, force_in=True):
    """
    randomly crop a portion of size prop of the original image size.
    """ 
    img_shape = np.array(img.shape)[1:]
    assert len(img_shape)==len(crop_shape),"[Error] Not the same dimensions! Image shape {}, Crop shape {}".format(img_shape, crop_shape)
    
    if force_in: # force the crop to be located in image shape range
        start = np.random.randint(0, np.maximum(1,img_shape-crop_shape))
        end = start+crop_shape
        
        idx = [slice(0,img.shape[0])]+list(slice(s[0], s[1]) for s in zip(start, end))
        idx = tuple(idx)
        
        crop_img = img[idx]
        crop_msk = msk[idx]
    else: # the crop will be chosen randomly and then padded if needed
        # the crop might be too small but will be padded with zeros
        start = np.random.randint(0, img_shape)
        crop_img, crop_msk = centered_crop(img=img, msk=msk, center=start, crop_shape=crop_shape)
        # pad if needed
        if np.any(np.array(crop_img.shape)[1:]-crop_shape)!=0:
            crop_img, crop_msk = centered_pad(img=crop_img, msk=crop_msk, final_size=crop_shape)

    return crop_img, crop_msk

def random_crop_pad(img, msk, final_size, fg_rate=0.33, fg_margin=np.zeros(3), fg=None, use_softmax=True):
    """
    random crop and pad if needed.
    """
    if type(img)==list: # then batch mode
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
        img, msk = random_crop(img, msk, final_size, force_in=False)
        
    # pad if needed
    if np.any(np.array(img.shape)[1:]-final_size)!=0:
        img, msk = centered_pad(img=img, msk=msk, final_size=final_size)
    return img, msk

def random_crop_resize(img, msk, crop_scale, final_size, fg_rate=0.33, fg_margin=np.zeros(3)):
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
        rnd_label = random.randint(0,msk.shape[0]-1) # choose a random label
        
        locations = np.argwhere(msk[rnd_label] == 1)
        
        if locations.size==0: # bug fix when having empty arrays 
            img, msk = random_crop(img, msk, crop_shape)
        else:
            center=random.choice(locations) # choose a random voxel of this label
            img, msk = located_crop(img, msk, center, crop_shape, fg_margin)
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

class LabelToLong:
    def __init__(self, label_name):
        self.label_name = label_name
        
    def __call__(self, subject):
        if self.label_name in subject.keys():
            subject[self.label_name].set_data(subject[self.label_name].data.long())
        return subject

#---------------------------------------------------------------------------

class SemSeg3DPatchFast(Dataset):
    """
    with DataLoader
    """
    def __init__(
        self,
        img_path,
        msk_path,
        batch_size, 
        patch_size,
        nbof_steps,
        folds_csv  = None, 
        fold       = 0, 
        val_split  = 0.25,
        train      = True,
        use_aug    = True,
        aug_patch_size = None,
        fg_path  = None,
        fg_rate = 0.33, # if > 0, force the use of foreground, needs to run some pre-computations (note: better use the foreground scheduler)
        crop_scale = 1.0, # if > 1, then use random_crop_resize instead of random_crop_pad
        load_data = False, # if True, loads the all dataset into computer memory (faster but more memory expensive)
        use_softmax = True,
        ):

        self.img_path = img_path
        self.msk_path = msk_path
        self.fg_path = fg_path

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.aug_patch_size = aug_patch_size

        self.nbof_steps = nbof_steps

        self.load_data = load_data

        self.handler = DataHandlerFactory.get(
            self.img_path,
            read_only=True,
            img_path = img_path,
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

        else: 
            all_set = self.handler.extract_inner_path(self.handler.images)
            assert len(all_set) > 0, "[Error] Incorrect path for folder of images or your folder is empty."
            np.random.shuffle(all_set) # shuffle all_set
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

        self.handler.open(
            img_path = img_path,
            msk_path = msk_path,
            fg_path = fg_path,
            img_inner_path_list = self.fnames,
            msk_inner_path_list = self.fnames,
            fg_inner_path_list = self.fnames,
        )

        # print train and validation image names
        print("{} images: {}".format("Training" if self.train else "Validation", self.fnames))
        
        if self.load_data:
            print("Loading the whole dataset into computer memory...")
            def load_data():
                nonlocal fg_path
                imgs_data = []
                msks_data = []
                fg_data   = []
                for i,m,f in self.handler:
                    # load img and msks
                    imgs_data += [self.handler.load(i)[0]]
                    msks_data += [self.handler.load(m)[0]]

                    # load foreground 
                    if fg_path is not None:
                        fg_data += [self.handler.load(m)[0]]
                return imgs_data, msks_data, fg_data

            self.imgs_data, self.msks_data, self.fg_data = load_data()
            print("Done!")

        self.use_aug = use_aug

        if self.use_aug:
            ps = np.array(self.patch_size)

            # [aug] flipping probabilities
            flip_prop=ps.min()/ps
            flip_prop/=flip_prop.sum()

            # [aug] 'axes' for tio.RandomAnisotropy
            anisotropy_axes=tuple(np.arange(len(ps))[ps/ps.min()>3].tolist())
            if len(anisotropy_axes)==0:
                anisotropy_axes=tuple(np.arange(len(ps)).tolist())

            # [aug] 'degrees' for tio.RandomAffine
            if np.any(ps/ps.min()>3): # then use dummy_2d
                degrees = tuple(180 if p==ps.argmin() else 0 for p in range(len(ps)))
            else:
                degrees = (-45,45)

            # [aug] 'cropping'
            # the affine transform is computed on bigger patches than the other transform
            # that's why we need to crop the patch after potential affine augmentation
            start = (np.array(self.aug_patch_size)-np.array(self.patch_size))//2
            end = self.aug_patch_size-(np.array(self.patch_size)+start)
            cropping = (start[0],end[0],start[1],end[1],start[2],end[2])
            
            # the foreground-crop-function forces the foreground to be in the center of the patch
            # so that, when doing the second centrering crop, the foreground is still present in the patch,
            # that's why there is a margin here
            self.fg_margin = np.zeros(len(patch_size))

            self.rotate = tio.Compose([
                                 tio.RandomAffine(scales=0, degrees=degrees, translation=0, default_pad_value=0),
                                 tio.Crop(cropping=cropping),
                                 LabelToLong(label_name='msk')])
            self.transform = tio.Compose([

                tio.Compose([tio.RandomAffine(scales=(0.7,1.4), degrees=0, translation=0),
                             LabelToLong(label_name='msk')
                            ], p=0.2),

                # spatial augmentations
                tio.RandomAnisotropy(p=0.1, axes=anisotropy_axes, downsampling=(1,1.5)),
                tio.RandomFlip(p=1, axes=(0,1,2)),
                tio.RandomBiasField(p=0.15, coefficients=0.2),
                tio.RandomBlur(p=0.2, std=(0.5,1)),
                tio.RandomNoise(p=0.2, std=(0,0.1)),
                tio.RandomSwap(p=0.2, patch_size=ps//8),
                tio.RandomGamma(p=0.3, log_gamma=(-0.35,0.4)),
            ])

        self.fg_rate = fg_rate
        self.crop_scale = crop_scale 
        assert self.crop_scale >= 1, "[Error] crop_scale must be higher or equal to 1"

        self.use_softmax = use_softmax
        self.batch_idx = 0
    
    def set_fg_rate(self,value):
        """
        setter function for the foreground rate class parameter
        """
        self.fg_rate = value

    def _do_fg(self):
        """
        determines whether to force the foreground depending on the batch idx
        """
        return not self.batch_idx < round(self.batch_size * (1 - self.fg_rate))
    
    def _update_batch_idx(self):
        self.batch_idx += 1
        if self.batch_idx >= self.batch_size:
            self.batch_idx = 0
    
    def __len__(self):
        return self.nbof_steps*self.batch_size
    
    def __getitem__(self, idx):

        if self.load_data:
            img = self.imgs_data[idx%len(self.imgs_data)]
            msk = self.msks_data[idx%len(self.msks_data)]
            if len(self.fg_data)>0: fg = self.fg_data[idx%len(self.fg_data)]
            else: fg = None
        else:
            img_fname = self.fnames[idx%len(self.fnames)]
            idx=idx%len(self.fnames)

            # read the images
            img = self.handler.load(self.handler.images[idx])[0]
            msk = self.handler.load(self.handler.masks[idx])[0]

            # read foreground data
            if self.fg_path is not None:
                fg = self.handler.load(self.handler.fg[idx])[0]
            else:
                fg = None

        # random crop and pad
        # rotation augmentation requires a larger patch size
        do_rot = random.random() < 0.2 # rotation proba = 0.2 
        final_size = self.aug_patch_size if self.use_aug and do_rot else self.patch_size
        fg_margin = self.fg_margin if self.use_aug else np.zeros(3)
        if self.train and self.crop_scale > 1:
            img, msk = random_crop_resize(
                img,
                msk,
                crop_scale=self.crop_scale,
                final_size=final_size,
                fg_rate=self.fg_rate,
                fg_margin=fg_margin,
                )
        else:
            img, msk = random_crop_pad(
                img,
                msk,
                final_size=final_size,
                fg_rate=int(self._do_fg()),
                fg_margin=fg_margin,
                fg = fg,
                use_softmax=self.use_softmax
                )
            self._update_batch_idx()

        # data augmentation
        if self.use_aug:
            sub = tio.Subject(img=tio.ScalarImage(tensor=img), msk=tio.LabelMap(tensor=msk))
            if do_rot: sub = self.rotate(sub)
            sub = self.transform(sub)
            img, msk = sub.img.tensor, sub.msk.tensor
        
            # to float for msk
            msk = msk.float()
        else:
            # convert mask to float for validation
            msk = msk.astype(float)
        
        return img, msk

#---------------------------------------------------------------------------
