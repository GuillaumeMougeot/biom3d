"""Dataset primitives for 3D segmentation dataset. Solution: patch approach with the whole dataset into memory, based on Torchio, fastest dataloading method so far."""

from typing import Optional
import numpy as np 
import torchio as tio
import random 
import pandas as pd 
import torch

from torchio import SubjectsDataset
from torchio import Subject
import copy

from biom3d.utils import get_folds_train_test_df, DataHandlerFactory, DataHandler

#---------------------------------------------------------------------------
# utilities to random crops

from torchio.transforms.augmentation import RandomTransform
from torchio.transforms import SpatialTransform
from torchio import Subject, LOCATION
from torchio.types import TypeTripletInt, TypeSpatialShape
from torchio.utils import to_tuple

class RandomCropOrPad(RandomTransform, SpatialTransform):
    """
    Randomly crop a subject, and pad it if needed.

    :ivar numpy.ndarray[np.uint16] patch_size:      
    :ivar float fg_rate: Foreground rate, if > 0, force the use of foreground.
    :ivar str label_name: Name of the label image in the tio.Subject.
    :ivar int start_fg_idx: Starting index in foreground. Determined by softmax use.    
    """

    patch_size:np.ndarray[np.uint16]        
    fg_rate:float
    label_name:str
    start_fg_idx:int

    def __init__(
        self,
        patch_shape:np.ndarray,
        fg_rate:float = 0,
        label_name:str = None,
        use_softmax:bool = True,
        **kwargs,
    ):
        """
        Randomly crop a subject, and pad it if needed.

        Adapted from tio.data.sampler.PatchSampler.
        
        Parameters
        ----------
        patch_size: numpy.ndarray
            Size of a patch.
        fg_rate: int, default=0
            Foreground rate, if > 0, force the use of foreground. Label name must be specified.
        label_name: str, default=None
            Used with the foreground rate. Name of the label image in the tio.Subject.
        use_softmax: boolean, default=True
            Used with the foreground rate to know if the background should be removed.
        **kwargs : dict
            Additional keyword arguments.

        Raises
        ------
        ValueError
            If a dimension of patch_size in <1 or not an int (or np.integer)
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
        """
        Extract a patch from the given subject starting at a specified index.

        Args:
            subject: Subject 
                The subject to extract the patch from.
            index_ini: TypeTripletInt 
                The starting index (x, y, z) of the patch.

        Returns:
            cropped_subject: Subject
                The extracted patch as a new subject.
        """
        cropped_subject = self.crop(subject, index_ini, self.patch_size)  # type: ignore[arg-type]  # noqa: B950
        return cropped_subject
    
    # TODO: Maybe use method injection instead of copying for maintability (beware behaviour change)
    def crop(
        self,
        subject: Subject,
        index_ini: TypeTripletInt,
        patch_size: TypeTripletInt,
    ) -> Subject:
        """
        Crop a patch from the subject at a given position and size.

        Copied from ``tio.data.sampler.PatchSampler``.

        Parameters
        ----------
        subject : Subject
            The subject to crop.
        index_ini : TypeTripletInt
            The starting index (x, y, z) of the crop.
        patch_size : TypeTripletInt
            The size of the patch to extract (dx, dy, dz).

        Returns
        -------
        cropped_subject: Subject
            The cropped subject with the patch and an updated LOCATION attribute.
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
    )->tio.Crop:
        """
        Compute a centered crop transform from index and patch size.

        Adapted from ``tio.data.sampler.PatchSampler``.

        Parameters
        ----------
        subject : Subject
            The subject to be cropped.
        index_ini : TypeTripletInt
            The (x, y, z) starting index of the patch.
        patch_size : TypeSpatialShape
            The size of the patch to extract.

        Returns
        -------
        tio.Crop
            A crop transform that extracts the desired patch while remaining within bounds.
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
        Apply patch sampling to the subject, with optional foreground enforcement.

        A patch is randomly sampled from the subject. If `fg_rate` > 0, a random foreground
        voxel may be used to center the patch, based on the label map. Otherwise, a random
        valid location is used. If the patch is smaller than `patch_size`, symmetric padding
        is applied.

        Adapted from tio.data.sampler.UniformSampler

        Parameters
        ----------
        subject : Subject
            The subject to transform.

        Returns
        -------
        transformed: Subject
            The subject containing the sampled and padded patch.
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
            
            # safeguard
            if isinstance(locations,np.ndarray):locations = torch.from_numpy(locations)
            if len(locations)==0: # bug fix when having empty arrays 
                index_ini = tuple(int(torch.randint(np.maximum(x,0) + 1, (1,)).item()) for x in valid_range)
            else:
                # crop the img and msk so that the foreground voxel is located somewhere (random) in the cropped patch
                # the margin argument adds a margin to force the foreground voxel to be located nearer the middle of the patch 
                # center=random.choice(locations.numpy()) # choose a random voxel of this label
                center=locations[torch.randint(locations.size(0), (1,)).item()]
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
    """
    Transform to convert label data to float type.
        
    :ivar str label_name: Name of the label to be transformed.
    """

    label_name:str

    def __init__(self, label_name:str):
        """
        Transform to convert label data to float type.
            
        Parameters
        ----------
        label_name : str
            Name of the label to be transformed.
        """
        self.label_name = label_name
        
    def __call__(self, subject:Subject)->Subject:
        """
        Apply the transform to the given subject.

        Converts the label tensor to float if the label is present.
            
        Parameters
        ----------
        subject : Subject
            A TorchIO subject that may contain the specified label.
        """
        if self.label_name in subject.keys():
            subject[self.label_name].set_data(subject[self.label_name].data.float())
        return subject

class LabelToLong:
    """
    Transform to convert label data to long type.
        
    :ivar str label_name: Name of the label to be transformed.
    """

    label_name:str

    def __init__(self, label_name:str):
        """
        Transform to convert label data to long type.
            
        Parameters
        ----------
        label_name : str
            Name of the label to be transformed.
        """
        self.label_name = label_name
        
    def __call__(self, subject:Subject)->Subject:
        """
        Apply the transform to the given subject.

        Converts the label tensor to long if the label is present.
            
        Parameters
        ----------
        subject : Subject
            A TorchIO subject that may contain the specified label.
        """
        if self.label_name in subject.keys():
            subject[self.label_name].set_data(subject[self.label_name].data.long())
        return subject

class LabelToBool:
    """
    Transform to convert label data to bool type.
        
    :ivar str label_name: Name of the label to be transformed.
    """

    label_name:str

    def __init__(self, label_name:str):
        """
        Transform to convert label data to bool type.
            
        Parameters
        ----------
        label_name : str
            Name of the label to be transformed.
        """
        self.label_name = label_name
        
    def __call__(self, subject:Subject)->Subject:
        """
        Apply the transform to the given subject.

        Converts the label tensor to bool if the label is present.
            
        Parameters
        ----------
        subject : Subject
            A TorchIO subject that may contain the specified label.
        """
        if self.label_name in subject.keys():
            subject[self.label_name].set_data(subject[self.label_name].data.bool())
        return subject

#---------------------------------------------------------------------------

class TorchIOReaderWrapper:
    """
    A wrapper class so TorchIO can use a DataHandler.

    :ivar DataHandler handler: DataHandler used to read data.
    """

    handler:DataHandler

    def __init__(self, handler:DataHandler):
        """
        Initialize the wrapper.
        
        Paramters
        ---------
        handler: DataHandler
            DataHandler used to read data.
        """
        self.handler = handler  

    def __call__(self, path:str)->tuple[torch.Tensor,Optional[dict]]:
        """
        Delegate data reading to DataHandler.

        Parameters
        ----------
        path : str
            Path to the image file. Supposedly a path coming from the DataHandler.

        Returns
        -------
        img: torch.Tensor
            Image data
        meta: dict, optional
            Eventual meta data (can be None).
        """
        img,meta = self.handler.load(path), None
        img = torch.from_numpy(img)
        return img,meta
#---------------------------------------------------------------------------
# Based on torchio.SubjectsDataset

class TorchioDataset(SubjectsDataset):
    """
    Custom dataset similar to `torchio.SubjectsDataset` but supports an unlimited number of steps (batches) per epoch.

    Handles loading of images, masks, and foreground data, train/validation splitting,
    optional in-memory data loading, and specific data augmentations.

    :ivar str img_path: Path to the collection containing image files.
    :ivar str msk_path: Path to the collection containing mask files.
    :ivar Optional[str] fg_path: Path to the collection containing foreground data (optional).
    :ivar int batch_size: Batch size for sampling.
    :ivar numpy.ndarray patch_size: Size of the patches to extract.
    :ivar Optional[numpy.ndarray] aug_patch_size: Size of patches used for augmentation (optional). Can be larger than patch_size
    :ivar int nbof_steps: Number of steps (batches) per epoch.
    :ivar bool load_data: Whether to load all data into memory.
    :ivar DataHandler handler: Data handler for loading images and masks.
    :ivar bool train: Indicates if the dataset is used for training (True) or validation (False).
    :ivar list[str] fnames: List of filenames used depending on training or validation mode.
    :ivar list[Subject] subjects_list: List of TorchIO Subjects created from the files.
    :ivar bool use_aug: Whether data augmentations are enabled.
    :ivar float fg_rate: Foreground inclusion rate to force foreground sampling in patches.
    :ivar bool use_softmax: Whether to use softmax activation; if False, sigmoid is used.
    :ivar int batch_idx: Current batch index for internal tracking.
    """

    img_path:str
    msk_path:str
    fg_path:Optional[str]
    batch_size:int
    patch_size:np.ndarray
    aug_patch_size:Optional[np.ndarray]
    nbof_steps:int
    load_data:bool
    handler:DataHandler
    train:bool
    fnames:list[str]
    subjects_list:list[Subject]
    use_aug:bool
    fg_rate:float
    use_softmax:bool
    batch_idx:int
    
    def __init__(
        self,
        img_path:str,
        msk_path:str,
        batch_size:int, 
        patch_size:np.ndarray,
        nbof_steps:int,
        fg_path:Optional[str]     = None,
        folds_csv:Optional[str]  = None, 
        fold:int       = 0, 
        val_split:float  = 0.25,
        train:bool      = True,
        use_aug:bool    = True,
        aug_patch_size:Optional[np.ndarray] = None,
        fg_rate:float = 0.33, 
        load_data:bool = False,
        use_softmax:bool = True,
    ):
        """
        Similar as torchio.SubjectsDataset but can be use with an unlimited amount of steps.

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
        fg_path : str, optional
            Path to collection containing foreground information.
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
        fg_rate : float, default=0.33
            Foreground rate, used to force foreground inclusion in patches. If > 0, force the use of foreground, needs to run some pre-computations (note: better use the foreground scheduler)
        load_data : bool, default=False
            If True, loads the all dataset into computer memory (faster but more memory expensive). 
        use_softmax : bool, default=True
            If True, use softmax activation; otherwise, sigmoid is used.
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

        self.handler.open(
            img_path = img_path,
            msk_path = msk_path,
            fg_path = fg_path,
            img_inner_paths_list = self.fnames,
            msk_inner_paths_list = self.fnames,
            fg_inner_paths_list = [f[:f.find('.')]+'.pkl' for f in self.fnames],
        )

        if len(self.fnames)==1: self.load_data=True # we force dataloading for single images.

        # print train and validation image names
        print("{} images: {}".format("Training" if self.train else "Validation", self.fnames))
        
        if self.load_data:
            print("Loading the whole dataset into computer memory...")
            
        def load_subjects():
            subjects_list = []
            for i,m,f in self.handler:
                if self.fg_path is not None:
                    fg = self.handler.load(f)[0]
                else: 
                    fg = None

                # load img and msks
                if self.load_data:
                    img = torch.from_numpy(self.handler.load(i)[0].astype(np.float32))
                    msk = torch.from_numpy(self.handler.load(m)[0].astype(np.int8)).long()
                    subjects_list += [
                        tio.Subject(
                            img=tio.ScalarImage(tensor=img),
                            msk=tio.LabelMap(tensor=msk) if fg is None else tio.LabelMap(tensor=msk, fg=fg))]    
                else:
                    reader = TorchIOReaderWrapper(self.handler)
                    subjects_list += [
                        tio.Subject(
                            img=tio.ScalarImage(i, reader=reader),
                            msk=tio.LabelMap(m, reader=reader) if fg is None else tio.LabelMap(tensor=msk, reader=reader, fg=fg))] 
            return subjects_list

        self.subjects_list = load_subjects()
        self.use_aug = use_aug
        self.fg_rate = fg_rate
        self.use_softmax = use_softmax
        self.batch_idx = 0
        
        if self.use_aug:
            ps = np.array(self.patch_size)

            anisotropy_axes=tuple(np.arange(3)[ps/ps.min()>3].tolist())
            # if anisotropy is empty, it means that all axes could be use for anisotropy augmentation
            if len(anisotropy_axes)==0: anisotropy_axes=tuple(range(len(ps)))

            # [aug] 'degrees' for tio.RandomAffine
            if np.any(ps/ps.min()>3): # then use dummy_2d
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
                    tio.Compose([
                                 tio.RandomAffine(scales=0, degrees=degrees, translation=0, default_pad_value=0),
                                 tio.Crop(cropping=cropping),
                                 LabelToLong(label_name='msk')
                                ]): 0.2,
                    tio.Crop(cropping=cropping): 0.8,
                }),

                tio.Compose([tio.RandomAffine(scales=(0.7,1.4), degrees=0, translation=0),
                             LabelToLong(label_name='msk')
                            ], p=0.2),
                # spatial augmentations
                tio.RandomAnisotropy(p=0.2, axes=anisotropy_axes, downsampling=(1,2)),
                tio.RandomFlip(p=1, axes=(0,1,2)),
                tio.RandomBiasField(p=0.15, coefficients=0.2),
                tio.RandomBlur(p=0.2, std=(0.5,1)),
                tio.RandomNoise(p=0.2, std=(0,0.1)),
                tio.RandomSwap(p=0.2, patch_size=ps//8),
                tio.RandomGamma(p=0.3, log_gamma=(-0.35,0.4)),
            ])

        SubjectsDataset.__init__(self, subjects=self.subjects_list)
    
    def _do_fg(self)->bool:
        """
        Determine whether to force the foreground depending on the batch idx.
        
        Returns
        -------
        bool
            True if batch_index >= batch_size * (1-fg_rate).
        """
        return self.batch_idx >= round(self.batch_size * (1 - self.fg_rate))
    
    def _update_batch_idx(self)->None:
        """Increment batch index, modulo batch size."""
        self.batch_idx += 1
        if self.batch_idx >= self.batch_size:
            self.batch_idx = 0

    def __len__(self)->int:
        """Return number of step * batch size."""
        return self.nbof_steps*self.batch_size

    def __getitem__(self, index: int) -> Subject:
        """
        Return Subject corresponding to index in the dataloader.

        Parameters
        ----------
        index: int
            Index of wanted data.
        """
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
