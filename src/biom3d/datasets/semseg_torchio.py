#---------------------------------------------------------------------------
# Dataset primitives for 3D segmentation dataset
# solution: patch approach with the whole dataset into memory 
#           based on Torchio, fastest dataloading method so far.
#---------------------------------------------------------------------------

import os
import numpy as np 
import torchio as tio
import random 
import pickle
# from monai.data import CacheDataset
import pandas as pd 
from tifffile import imread
import torch

from torchio import SubjectsDataset
from torchio import Subject
import copy

from biom3d.utils import get_folds_train_test_df, adaptive_imread

#---------------------------------------------------------------------------
# utilities to random crops

from torchio.transforms.augmentation import RandomTransform
from torchio.transforms import SpatialTransform
from torchio import Subject, LOCATION
from torchio.typing import TypeTripletInt, TypeSpatialShape
from torchio.utils import to_tuple

class RandomCropOrPad(RandomTransform, SpatialTransform):
    """Randomly crop a subject, and pad it if needed
    """

    def __init__(
        self,
        patch_shape,
        fg_rate = 0,
        label_name = None,
        use_softmax = True,
        **kwargs,
    ):
        """
        __init__ adapted from tio.data.sampler.PatchSampler
        
        Parameters
        ----------
        fg_rate: int, default=0
            Foreground rate, if > 0, force the use of foreground. Label name must be specified.
        label_name: str, default=None
            Used with the foreground rate. Name of the label image in the tio.Subject.
        use_softmax: boolean, default=True
            Used with the foreground rate to know if the background should be removed.
        """
        super().__init__(**kwargs)
        patch_size_array = np.array(to_tuple(patch_shape, length=3))
        for n in patch_size_array:
            if n < 1 or not isinstance(n, (int, np.integer)):
                message = (
                    'Patch dimensions must be positive integers,'
                    f' not {patch_size_array}'
                )
                raise ValueError(message)
        self.patch_size = patch_size_array.astype(np.uint16)
        
        self.fg_rate = fg_rate
        self.label_name = label_name
        self.start_fg_idx = int(use_softmax)

    def extract_patch(
        self,
        subject: Subject,
        index_ini: TypeTripletInt,
    ) -> Subject:
        cropped_subject = self.crop(subject, index_ini, self.patch_size)  # type: ignore[arg-type]  # noqa: B950
        return cropped_subject
    
    def crop(
        self,
        subject: Subject,
        index_ini: TypeTripletInt,
        patch_size: TypeTripletInt,
    ) -> Subject:
        """
        copied from tio.data.sampler.PatchSampler
        """
        transform = self._get_crop_transform(subject, index_ini, patch_size)
        cropped_subject = transform(subject)
        index_ini_array = np.asarray(index_ini)
        patch_size_array = np.asarray(patch_size)
        index_fin = index_ini_array + patch_size_array
        location = index_ini_array.tolist() + index_fin.tolist()
        cropped_subject[LOCATION] = torch.as_tensor(location)
        cropped_subject.update_attributes()
        return cropped_subject
    
    @staticmethod
    def _get_crop_transform(
        subject,
        index_ini: TypeTripletInt,
        patch_size: TypeSpatialShape,
    ):
        """
        adapted from tio.data.sampler.PatchSampler
        """

        shape = np.array(subject.spatial_shape, dtype=np.uint16)
        index_ini_array = np.array(index_ini, dtype=np.uint16)
        patch_size_array = np.array(patch_size, dtype=np.uint16)
        assert len(index_ini_array) == 3
        assert len(patch_size_array) == 3
        index_fin = np.minimum(index_ini_array + patch_size_array,shape)
        crop_ini = index_ini_array.tolist()
        crop_fin = (shape - index_fin).tolist()
        start = ()
        cropping = sum(zip(crop_ini, crop_fin), start)
        return tio.Crop(cropping)  # type: ignore[arg-type]
        
    def apply_transform(self, subject: Subject) -> Subject:
        """
        adapted from tio.data.sampler.UniformSampler
        """
        valid_range = np.maximum(subject.spatial_shape - self.patch_size,1)
        
        force_fg = random.random()
        if self.fg_rate>0 and force_fg<self.fg_rate:
            if 'fg' in subject[self.label_name].keys() and subject[self.label_name]['fg'] is not None:
                fg = subject[self.label_name]['fg']
                locations = fg[random.choice(list(fg.keys()))]
            else:
                label = subject[self.label_name].data
                if tuple(label.shape)[0]==1:
                    # then we consider that we don't have a one hot encoded label
                    rnd_label = random.randint(1,label.max()+1)
                    locations = torch.argwhere(label[0] == rnd_label)
                else:
                    # then we have a one hot encoded label
                    rnd_label = random.randint(self.start_fg_idx,tuple(label.shape)[0]-1)
                    locations = torch.argwhere(label[rnd_label] == 1)
            
            if len(locations)==0: # bug fix when having empty arrays 
                index_ini = tuple(int(torch.randint(np.maximum(x,0) + 1, (1,)).item()) for x in valid_range)
            else:
                # crop the img and msk so that the foreground voxel is located somewhere (random) in the cropped patch
                # the margin argument adds a margin to force the foreground voxel to be located nearer the middle of the patch 
                # center=random.choice(locations.numpy()) # choose a random voxel of this label
                center=locations[torch.randint(locations.size(0), (1,)).item()]
                # margin=np.zeros(3) # TODO: make this a parameter
                # lower_bound = np.clip(center-np.array(self.patch_size)+margin, 0., valid_range)
                # higher_bound = np.clip(center-margin, lower_bound+1, valid_range)
                # index_ini = tuple(np.random.randint(low=lower_bound, high=higher_bound))
                # index_ini = tuple(np.maximum(center-np.array(self.patch_size)//2, 0).astype(int))
                index_ini = tuple(int(np.maximum(x,0)) for x in (center.numpy()-self.patch_size//2))
        else:
            index_ini = tuple(int(torch.randint(np.maximum(x,0) + 1, (1,)).item()) for x in valid_range)
        transformed = self.extract_patch(subject, index_ini)
        
        # centered pad if needed
        if np.any(transformed.spatial_shape-self.patch_size)!=0:
            start = -valid_range//2
            start = start * (start > 0)
            end = self.patch_size-(transformed.spatial_shape+start)
            end = end * (end > 0)

            padding = np.stack((start,end),axis=1).flatten()
            transformed = tio.Pad(padding)(transformed)

        assert isinstance(transformed, Subject)
        return transformed

#---------------------------------------------------------------------------
# utilities to change variable type in label/mask

class LabelToFloat:
    def __init__(self, label_name):
        self.label_name = label_name
        
    def __call__(self, subject):
        if self.label_name in subject.keys():
            subject[self.label_name].set_data(subject[self.label_name].data.float())
        return subject

class LabelToLong:
    def __init__(self, label_name):
        self.label_name = label_name
        
    def __call__(self, subject):
        if self.label_name in subject.keys():
            subject[self.label_name].set_data(subject[self.label_name].data.long())
        return subject

class LabelToBool:
    def __init__(self, label_name):
        self.label_name = label_name
        
    def __call__(self, subject):
        if self.label_name in subject.keys():
            subject[self.label_name].set_data(subject[self.label_name].data.bool())
        return subject

#---------------------------------------------------------------------------

def reader(x):
    return adaptive_imread(str(x))[0], None

#---------------------------------------------------------------------------
# Based on torchio.SubjectsDataset

class TorchioDataset(SubjectsDataset):
    """
    Similar as torchio.SubjectsDataset but can be use with an unlimited amount of steps.
    """
    
    def __init__(
        self,
        img_dir,
        msk_dir,
        batch_size, 
        patch_size,
        nbof_steps,
        fg_dir     = None,
        folds_csv  = None, 
        fold       = 0, 
        val_split  = 0.25,
        train      = True,
        use_aug    = True,
        aug_patch_size = None,
        fg_rate = 0.33, # if > 0, force the use of foreground, needs to run some pre-computations (note: better use the foreground scheduler)
        # crop_scale = 1.0, # if > 1, then use random_crop_resize instead of random_crop_pad
        load_data = False,
        use_softmax = True,
    ):
        """
        Parameters
        ----------
        load_data : boolean, default=False
            if True, loads the all dataset into computer memory (faster but more memory expensive). ONLY COMPATIBLE WITH .npy PREPROCESSED IMAGES
        """
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.fg_dir = fg_dir

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

        else: 
            all_set = os.listdir(img_dir)
            val_split = np.round(val_split * len(all_set)).astype(int)

            # force validation to contain at least one image
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

        if len(self.fnames)==1: self.load_data=True # we force dataloading for single images.

        # print train and validation image names
        print("{} images: {}".format("Training" if self.train else "Validation", self.fnames))
        
        if self.load_data:
            print("Loading the whole dataset into computer memory...")
            
        def load_subjects(fnames):
            subjects_list = []
            for idx in range(len(fnames)):
                img_path = os.path.join(self.img_dir, fnames[idx])
                msk_path = os.path.join(self.msk_dir, fnames[idx])
                if self.fg_dir is not None:
                    fg_path = os.path.join(self.fg_dir, fnames[idx][:fnames[idx].find(".")]+'.pkl')
                    fg = pickle.load(open(fg_path, 'rb'))
                    fg = {k:torch.tensor(v) for k,v in fg.items()}
                else: 
                    fg = None

                # load img and msks
                if self.load_data:
                    img = torch.from_numpy(adaptive_imread(img_path)[0].astype(np.float32))
                    msk = torch.from_numpy(adaptive_imread(msk_path)[0].astype(np.int8)).long()
                    subjects_list += [
                        tio.Subject(
                            img=tio.ScalarImage(tensor=img),
                            msk=tio.LabelMap(tensor=msk) if fg is None else tio.LabelMap(tensor=msk, fg=fg))]    
                else:
                    subjects_list += [
                        tio.Subject(
                            img=tio.ScalarImage(img_path, reader=reader),
                            msk=tio.LabelMap(msk_path, reader=reader) if fg is None else tio.LabelMap(tensor=msk, reader=reader, fg=fg))] 
            return subjects_list

        self.subjects_list = load_subjects(self.fnames)
        self.use_aug = use_aug
        self.fg_rate = fg_rate
        self.use_softmax = use_softmax
        self.batch_idx = 0
        
        if self.use_aug:
            ps = np.array(self.patch_size)

            anisotropy_axes=tuple(np.arange(3)[ps/ps.min()>3].tolist())
            # if anisotropy is empty, it means that all axes could be use for anisotropy augmentation
            if len(anisotropy_axes)==0: anisotropy_axes=tuple(i for i in range(len(ps)))

            # [aug] 'degrees' for tio.RandomAffine
            if np.any(ps/ps.min()>3): # then use dummy_2d
                # norm = ps*3/ps.min()
                # softmax=np.exp(norm)/sum(np.exp(norm))
                # degrees=softmax.min()*90/softmax
                # degrees = 180*ps.min()/ps
                degrees = []
                for dim in ps/ps.min():
                    if dim < 3:
                        degrees += [-180,180]
                    else:
                        degrees += [0,0]
                degrees = tuple(degrees)
            else:
                degrees = (-30,30)

            # [aug] 'cropping'
            # the affine transform is computed on bigger patches than the other transform
            # that's why we need to crop the patch after potential affine augmentation
            start = (np.array(self.aug_patch_size)-np.array(self.patch_size))//2
            end = self.aug_patch_size-(np.array(self.patch_size)+start)
            cropping = (start[0],end[0],start[1],end[1],start[2],end[2])
            
            self.transform = tio.Compose([
                # pre-cropping to aug_patch_size
                tio.OneOf({
                    tio.Compose([# RandomCropOrPad(self.aug_patch_size, fg_rate=self.fg_rate, label_name='msk', use_softmax=self.use_softmax),
                                #  tio.RandomAffine(scales=(0.7,1.4), degrees=degrees, translation=0),
                                 tio.RandomAffine(scales=0, degrees=degrees, translation=0, default_pad_value=0),
                                 tio.Crop(cropping=cropping),
                                 LabelToLong(label_name='msk')
                                ]): 0.2,
                    tio.Crop(cropping=cropping): 0.8,
#                     RandomCropOrPad(self.patch_size, fg_rate=self.fg_rate, label_name='msk',use_softmax=self.use_softmax): 0.8,
                }),

                tio.Compose([tio.RandomAffine(scales=(0.7,1.4), degrees=0, translation=0),
                             LabelToLong(label_name='msk')
                            ], p=0.2),
                # RandomCropOrPad(AUG_PATCH_SIZE),

                # spatial augmentations
                tio.RandomAnisotropy(p=0.2, axes=anisotropy_axes, downsampling=(1,2)),
                # tio.RandomAffine(p=0.25, scales=(0.7,1.4), degrees=degrees, translation=0),
                # tio.Crop(cropping=cropping),
                tio.RandomFlip(p=1, axes=(0,1,2)),
                # tio.OneOf({
                #     tio.RandomAffine(scales=0.1, degrees=10, translation=0): 0.8,
                #     tio.RandomElasticDeformation(): 0.2,
                # }),

                # intensity augmentations
                # tio.RandomMotion(p=0.2),
                # tio.RandomGhosting(p=0.2),
                # tio.RandomSpike(p=0.15),
                tio.RandomBiasField(p=0.15, coefficients=0.2),
                tio.RandomBlur(p=0.2, std=(0.5,1)),
                tio.RandomNoise(p=0.2, std=(0,0.1)),
                tio.RandomSwap(p=0.2, patch_size=ps//8),
                tio.RandomGamma(p=0.3, log_gamma=(-0.35,0.4)),
                # LabelToFloat(label_name='msk')
            ])

        SubjectsDataset.__init__(self, subjects=self.subjects_list)
    
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

    def __getitem__(self, index: int) -> Subject:
        try:
            index = int(index)%len(self._subjects)
        except (RuntimeError, TypeError):
            message = (
                f'Index "{index}" must be int or compatible dtype,'
                f' but an object of type "{type(index)}" was passed'
            )
            raise ValueError(message)

        subject = self._subjects[index]
        subject = copy.deepcopy(subject)  # cheap since images not loaded yet

        if self.load_getitem:
            subject.load()

        # Apply transform (this is usually the bottleneck)
        patch_size = self.aug_patch_size if self.use_aug else self.patch_size
        subject = RandomCropOrPad(patch_size, fg_rate=int(self._do_fg()), label_name='msk', use_softmax=self.use_softmax)(subject)
        self._update_batch_idx()
        if self.use_aug:
            subject = self.transform(subject)
        return subject['img'][tio.DATA], subject['msk'][tio.DATA]
    
#---------------------------------------------------------------------------
