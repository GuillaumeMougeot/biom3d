"""Dataset primitives for 3D segmentation dataset. Solution: patch approach with the whole dataset into memory."""


from typing import Any, Iterable, Optional
import numpy as np 
import torchio as tio
import random 
from torch.utils.data import Dataset
import pandas as pd 
from biom3d.utils import centered_pad, get_folds_train_test_df, DataHandlerFactory, DataHandler

#---------------------------------------------------------------------------
# utilities to random crops

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
    msk: ndaarray
        Mask data.
    center: iterable of int
        Center voxel location for cropping.
    crop_shape: iterable of int
        Shape of the crop.
    margin: iterable of float, default np.zeros(3)
        Margin around the center location.

    Raises
    ------
    AssertionError
        If center is out of the image

    Returns
    -------
    crop_img : numpy.ndarray
        Cropped image data.
    crop_msk : numpy.ndarray
        Cropped mask data.
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
    """
    Do a crop, forcing the location voxel to be located in the crop.
    
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
    margin :  iterable of float
        Margin around the location.
    Returns
    -------
    crop_img : numpy.ndarray
        Cropped image data.
    crop_msk : numpy.ndarray
        Cropped mask data.
     
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

def centered_pad(img:np.ndarray, 
                 final_size:Iterable[int], 
                 msk:Optional[np.ndarray]=None,
                 )->np.ndarray|tuple[np.ndarray,np.ndarray]:
    """
    Do a centered pad an img and msk to fit the final_size.

    Parameters
    ----------
    img : numpy.ndarray
        Image data.
    final_size : iterable of int
        Final size of the cropped image and mask.
    msk : numpy.ndarray, optional
        Mask data.

    Returns
    -------
    pad_img : numpy.ndarray
        Cropped image data, focused on the foreground region.
    pad_msk : numpy.ndarray, optional
        Cropped mask data, corresponding to the cropped image region.
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

def random_crop(img:np.ndarray, 
                msk:np.ndarray, 
                crop_shape:Iterable[int], 
                force_in:bool=True,
                )->tuple[np.ndarray,np.ndarray]:
    """
    Randomly crop a portion of size prop of the original image size.
    
    Parameters
    ----------
    img : numpy.ndarray
        Image data.
    msk : numpy.ndarray
        Mask data.
    crop_shape : 
        Shape of the crop.
    force_in : bool, optional
        If True, ensures the crop is fully within the image boundaries.

    Raises
    ------
    AssertionError
        If image shape (minus C) is not the same shape as crop_shape

    Returns
    -------
    crop_img : numpy.ndarray
        Cropped image data.
    crop_msk : numpy.ndarray
        Cropped mask data.
    """ 
    img_shape = np.array(img.shape)[1:]
    assert len(img_shape)==len(crop_shape),"[Error] Not the same dimensions! Image shape {}, Crop shape {}".format(img_shape, crop_shape)
    
    if force_in: # force the crop to be located in image shape range
        start = np.random.randint(0, np.maximum(1,img_shape-crop_shape))
        end = start+crop_shape
        
        idx = [slice(0,img.shape[0])]+[slice(s[0], s[1]) for s in zip(start, end)]
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

def random_crop_pad(img:np.ndarray, 
                    msk:np.ndarray, 
                    final_size:Iterable[int], 
                    fg_rate:float=0.33, 
                    fg_margin:Iterable[float]=np.zeros(3), 
                    fg:Optional[dict[int,np.ndarray]]=None, 
                    use_softmax:bool=True,
                    )->tuple[np.ndarray,np.ndarray]:
    """
    Do a random crop and pad if needed.
    
    Parameters
    ----------
    img : numpy.ndarray
        Image data.
    msk : numpy.ndarray
        Mask data.
    final_size : 
        Final size of the image and mask after cropping and padding.
    fg_rate : float, default=0.33
        Probability of focusing the crop on the foreground.
    fg_margin : iterable of float, default=np.zeros(3)
        Margin around the foreground location.
    fg : dict of int to numpy.ndarray, optional
        Foreground information.
    use_softmax : bool, default=True
        If True, assumes softmax activation.

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
        img, msk = random_crop(img, msk, final_size, force_in=False)
        
    # pad if needed
    if np.any(np.array(img.shape)[1:]-final_size)!=0:
        img, msk = centered_pad(img=img, msk=msk, final_size=final_size)
    return img, msk

def random_crop_resize(img:np.ndarray, 
                       msk:np.ndarray, 
                       crop_scale:float, 
                       final_size:Iterable[int], 
                       fg_rate:int=0.33, 
                       fg_margin:Iterable[float]=np.zeros(3),
                       )->tuple[np.ndarray,np.ndarray]:
    """
    Do a random crop and resize if needed.
    
    Parameters
    ----------
    img : numpy.ndarray
        Image data.
    msk : numpy.ndarray
        Mask data.
    crop_scale : float, >=1
        Scale factor for the crop size.
    final_size : iterable of int 
        Final size of the image and mask after cropping and resizing.
    fg_rate : float, default=0.33
        Probability of focusing the crop on the foreground.
    fg_margin : iterable of float, default=np.zeros(3)
        Margin around the foreground location.

    Raises
    ------
    ValueError
        If crop_scale < 1.

    Returns
    -------
    img : numpy.ndarray
        Cropped and resized image data.
    msk : numpy.ndarray
        Cropped and resized mask data.
    """
    final_size = np.array(final_size)

    if crop_scale < 1 :raise ValueError(f"Crop scale must be a float >1, found '{crop_scale}'")
        
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
    """
    Transform to convert label data to long (integer) type.
            
    :ivar str label_name: Name of the label to be transformed.

    """

    label_name:str

    def __init__(self, label_name:str):
        """
        Transform to convert label data to long (integer) type.
                
        Parameters
        ----------
        label_name : str
            Name of the label to be transformed.
            
        Returns
        -------
        subject : dict
            Dictionary with the label data converted to long (integer) type.
        """
        self.label_name = label_name
        
    def __call__(self, subject:dict[str,Any])->dict[str,Any]:
        """
        Transform to convert label data to long (integer) type.
                
        Parameters
        ----------
        subject : dict of string to any
            Dictionary that associate label name to values, should contains self.label_name
            
        Returns
        -------
        subject : dict
            Dictionary with the label data converted to long (integer) type.
        """
        if self.label_name in subject.keys():
            subject[self.label_name].set_data(subject[self.label_name].data.long())
        return subject

#---------------------------------------------------------------------------

class SemSeg3DPatchFast(Dataset):
    """
    Dataset class for semantic segmentation with 3D patches. Supports data augmentation and efficient loading.
    
    :ivar str img_path: Path to collection containing the image files.
    :ivar str msk_path: Path to collection containing the mask files.
    :ivar str | None fg_path: Path to collection containing the foreground files.
    :ivar int batch_size: Size of a batch.
    :ivar numpy.ndarray patch_size: Size of a patch.
    :ivar numpy.ndarray | None aug_patch_size: Size of augmented patch size, may be bigger than patch size.
    :ivar int nbof_steps: Number of steps (batches) per epoch.
    :ivar bool load_data: If True, load the entire dataset into memory. 
    :ivar DataHandler handler: DataHandler used to load data.
    :ivar bool train: If True, use the dataset for training; otherwise, use it for validation.
    :ivar list[str] fnames: List of image paths relative to img_path.
    :ivar bool use_aug: Whether to use data augmentation
    :ivar float fg_rate: Foreground rate, used to force foreground inclusion in patches.
    :ivar float crop_scale: Scale factor for crop size during augmentation.
    :ivar bool use_softmax: If True, use softmax activation.
    :ivar int batch_idx: Current batch index.
    """

    img_path:str
    msk_path:str
    fg_path:str
    batch_size:int
    patch_size:np.ndarray
    aug_patch_size:bool
    nbof_steps:int
    load_data:bool
    handler:DataHandler
    train:bool
    fnames:list[str]
    use_aug:bool
    fg_rate:float
    crop_scale:float
    use_softmax:bool
    batch_idx:int

    def __init__(
        self,
        img_path:str,
        msk_path:str,
        batch_size:int, 
        patch_size:np.ndarray,
        nbof_steps:int,
        folds_csv:Optional[str]  = None, 
        fold:int       = 0, 
        val_split:float  = 0.25,
        train:bool      = True,
        use_aug:bool    = True,
        aug_patch_size:Optional[np.ndarray] = None,
        fg_path:Optional[str]  = None,
        fg_rate:float = 0.33,  
        crop_scale:float = 1.0, 
        load_data:bool = False,
        use_softmax:bool = True,
        ):
        """
        Dataset class for semantic segmentation with 3D patches. Supports data augmentation and efficient loading.
        
        Parameters
        ----------
        img_path : str
            Path to collection containing the image files.
        msk_path : str
            Path to collection containing the mask files.
        batch_size : int
            Batch size for dataset sampling.
        patch_size : numpy.ndarray
            Size of the patches to be used.
        nbof_steps : int
            Number of steps (batches) per epoch.
        folds_csv : str, optional
            CSV file containing fold information for dataset splitting.
        fold : int, default=0
            The current fold number for training/validation splitting.
        val_split : float, default=0.25
            Proportion of data to be used for validation.
        train : bool, default=True
            If True, use the dataset for training; otherwise, use it for validation.
        use_aug : bool, default=True
            If True, apply data augmentation.
        aug_patch_size : numpy.ndarray, optional
            Patch size to use for augmented patches.
        fg_path : str, optional
            Path to collection containing foreground information.
        fg_rate : float, default=0.33
            Foreground rate, used to force foreground inclusion in patches. If > 0, force the use of foreground, needs to run some pre-computations (note: better use the foreground scheduler)
        crop_scale : float, default=1.0
            Scale factor for crop size during augmentation. If > 1, then use random_crop_resize instead of random_crop_pad
        load_data : bool, default=False
            If True, load the entire dataset into memory. 
        use_softmax : bool, default=True
            If True, use softmax activation.

        Raises
        ------
        AssertionError
            If fold_csv is None and not valid path for datas, or empty collections

            If crop_scale < 1
        """
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
            img_inner_paths_list = self.fnames,
            msk_inner_paths_list = self.fnames,
            fg_inner_paths_list = [f[:f.find('.')]+'.pkl' for f in self.fnames],
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
                        fg_data += [self.handler.load(f)[0]]
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
    
    def set_fg_rate(self,value:float)->None:
        """Setter function for the foreground rate class parameter."""
        self.fg_rate = value

    def _do_fg(self)->bool:
        """
        Determine whether to force the foreground depending on the batch idx.
        
        Returns
        -------
        bool
            True if batch_index >= batch_size * (1-fg_rate)
        """
        return self.batch_idx >= round(self.batch_size * (1 - self.fg_rate))
    
    def _update_batch_idx(self)->None:
        """Increment batch index, modulo batch_size."""
        self.batch_idx += 1
        if self.batch_idx >= self.batch_size:
            self.batch_idx = 0
    
    def __len__(self)->int:
        """Return nbof_step*batch_size."""
        return self.nbof_steps*self.batch_size
    
    def __getitem__(self, idx:int)->tuple[np.ndarray,np.ndarray]:
        """
        Return image and mask associated with index, with padding/croping, and data augmentation if use_data_aug.
        
        Parameters
        ----------
        idx: int
            The index of the wanted data.
        """
        if self.load_data:
            img = self.imgs_data[idx%len(self.imgs_data)]
            msk = self.msks_data[idx%len(self.msks_data)]
            if len(self.fg_data)>0: fg = self.fg_data[idx%len(self.fg_data)]
            else: fg = None
        else:
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
