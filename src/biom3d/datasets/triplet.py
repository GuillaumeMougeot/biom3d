
#---------------------------------------------------------------------------
# triplet dataloader
#---------------------------------------------------------------------------

import os
import numpy as np 
import torchio as tio
from torch.utils.data import Dataset
# from monai.data import CacheDataset
import pandas as pd 
from skimage.io import imread

from biom3d.utils import RandomCropResize, get_folds_train_test_df, centered_pad


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

def random_crop_pad(img, final_size):
    """
    random crop and pad if needed.
    """
    
    img = random_crop(img, final_size)

    # pad if needed
    if np.any(np.array(img.shape)[1:]-final_size)!=0:
        img = centered_pad(img, final_size)
    return img


#---------------------------------------------------------------------------

class Triplet(Dataset):
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
        # load_data = False,
        initial_overlap = 0.8,
        initial_global_crop_scale = 1.0,
        initial_global_crop_min_shape_scale = 1.0,
        ):

        self.img_dir = img_dir

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.aug_patch_size = aug_patch_size

        self.nbof_steps = nbof_steps

        # self.load_data = load_data
        
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
        
        # if self.load_data:
        #     print("Loading the whole dataset into computer memory...")
        #     def load_data(imgs_fnames):
        #         imgs_data = []
        #         for idx in range(len(imgs_fnames)):
        #             # file names
        #             img_path = os.path.join(self.img_dir, imgs_fnames[idx])
                    
        #             # image
        #             img = imread(img_path)
                    
        #             # expand dim if needed
        #             if len(img.shape)==3:
        #                 img = np.expand_dims(img, 0)

        #             # load img and msks
        #             imgs_data += [img]
        #         return imgs_data

        #     self.train_imgs_data = load_data(self.train_imgs)
        #     self.val_imgs_data = load_data(self.val_imgs)
        #     print("Done!")
            
        self.train = train
        print("current fold: {}\n \
        length of the training set: {}\n \
        length of the validation set: {}\n \
        length of the testing set: {}\n \
        is training mode active?: {}".format(fold, len(self.train_imgs), len(self.val_imgs), len(testset), self.train))

        self.use_aug = use_aug

        # data augmentation
        if self.use_aug:
            ps = np.array(self.patch_size)

            # [aug] flipping probabilities
            flip_prop=ps.min()/ps
            flip_prop/=flip_prop.sum()

            # [aug] 'axes' for tio.RandomAnisotropy
            anisotropy_axes=tuple(np.arange(3)[ps/ps.min()>3].tolist())

            # [aug] 'degrees' for tio.RandomAffine
            if np.any(ps/ps.min()>3): # else use dummy_2d
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
        
        self.crt_overlap = initial_overlap
        self.crt_global_crop_scale = initial_global_crop_scale
        self.crt_global_crop_min_shape_scale = initial_global_crop_min_shape_scale
        self.set_min_overlap(self.crt_overlap)
        self.set_global_crop(self.crt_global_crop_scale, self.crt_global_crop_min_shape_scale)
    
    def set_min_overlap(self, value):
        self.crop_resize = RandomCropResize(
            local_crop_shape=self.aug_patch_size if self.use_aug else self.patch_size,
            global_crop_shape=self.aug_patch_size if self.use_aug else self.patch_size,
            global_crop_scale=self.crt_global_crop_scale,
            global_crop_min_shape_scale=self.crt_global_crop_min_shape_scale,
            min_overlap=value,
        )
        self.crt_overlap = value
    
    def set_global_crop(self, scale=None, min_shape_scale=None):
        if scale is None and min_shape_scale is None:
            print("[Warning] Triplet.set_global_crop : one of 'scale' or 'min_shape_scale' must defined.")

        self.crop_resize = RandomCropResize(
            local_crop_shape=self.aug_patch_size if self.use_aug else self.patch_size,
            global_crop_shape=self.aug_patch_size if self.use_aug else self.patch_size,
            global_crop_scale=self.crt_global_crop_scale if scale is None else scale,
            global_crop_min_shape_scale=self.crt_global_crop_min_shape_scale if min_shape_scale is None else min_shape_scale,
            min_overlap=self.crt_overlap,
        )
        self.crt_global_crop_scale=self.crt_global_crop_scale if scale is None else scale
        self.crt_global_crop_min_shape_scale=self.crt_global_crop_min_shape_scale if min_shape_scale is None else min_shape_scale
    
    def __len__(self):
        # return len(self.train_imgs) if self.training else len(self.val_imgs)
        return self.nbof_steps*self.batch_size
    
    def __getitem__(self, idx):
        
        length = len(self.train_imgs) if self.train else len(self.val_imgs)
        idx_neg = np.random.randint(0, length)
        while (idx_neg%length)==(idx%length): idx_neg = np.random.randint(0, length)

        # if self.load_data:
        #     if self.train:
        #         img = self.train_imgs_data[idx%length]
        #         neg = self.train_imgs_data[idx_neg%length]
        #     else: 
        #         img = self.val_imgs_data[idx%length]
        #         neg = self.val_imgs_data[idx_neg%length]
        # else:
        fnames = self.train_imgs if self.train else self.val_imgs
        # img_fname = np.random.choice(fnames)
        img_fname = fnames[idx%length]
        neg_fname = fnames[idx_neg%length]

        # file names
        img_path = os.path.join(self.img_dir, img_fname)
        neg_path = os.path.join(self.img_dir, neg_fname)

        # read the images
        img = imread(img_path)
        neg = imread(neg_path)
        
        # expand dim if needed
        if len(img.shape)==3:
            img = np.expand_dims(img, 0)
            neg = np.expand_dims(neg, 0)
            

        # random crop and pad
        final_size = self.aug_patch_size if self.use_aug else self.patch_size
        anc = self.crop_resize.global_crop_resize(img)
        # pos = self.crop_resize.local_crop_pad(img)
        pos = self.crop_resize.global_crop_resize(img)
#         anc = random_crop_resize(img, scale=(0.7,1.0), final_size=final_size)
#         pos = random_crop_resize(img, scale=(0.7,1.0), final_size=final_size)
#         neg = random_crop_resize(neg, scale=(0.7,1.0), final_size=final_size)
        # anc = random_crop_pad(img,final_size=final_size)
        # pos = random_crop_pad(img,final_size=final_size)
        # neg = random_crop_pad(neg,final_size=final_size)
        neg = self.crop_resize.global_crop_resize(neg)

        # data augmentation
        if self.use_aug:
            anc = self.transform(anc)
            pos = self.transform(pos)
            neg = self.transform(neg)

        return anc, pos, neg

#---------------------------------------------------------------------------