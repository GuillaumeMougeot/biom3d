"""
Model predictors.

Contains 2 versions of predictors and the postprocessing function.
"""
# TODO: re-structure with classes maybe? more like creating a folder with V1 and V2 and separating post pro

import torch 
import torchio as tio
import numpy as np
from skimage.io import imread
from tqdm import tqdm

from biom3d.utils import keep_biggest_volume_centered, adaptive_imread, keep_big_volumes, resize_3d

#---------------------------------------------------------------------------
# model predictor for segmentation

def load_img_seg(fname:str)->torch.Tensor:
    """
    Load and preprocess a single image for segmentation prediction.
    
    *Segmentation Predictor V1*
    
    The image is normalized to [0,1], converted to a float PyTorch tensor,
    and reshaped to a 4D tensor with dimensions (batch=1, channel=1, height, width).
    
    Parameters
    ----------
    fname : str
        Path to the image file to load.
    
    Returns
    -------
    torch.Tensor
        Preprocessed image as a float PyTorch tensor,
        with shape (1, 1, H, W), ready as input for a model.
    """
    img = imread(fname)

    # normalize
    img = (img - img.min()) / (img.max() - img.min())
    
    # to tensor
    img = torch.tensor(img, dtype=torch.float)

    # expand dim
    img = torch.unsqueeze(img, dim=0)
    img = torch.unsqueeze(img, dim=0)

    return img

def seg_predict(
    img_path:str,
    model:torch.nn.Module,
    return_logit:bool=False,
    )->np.ndarray:
    """
    Run a prediction on given image.

    **Segmentation Predictor V1**
    
    Load an image from a given path, run model prediction,
    and return either the binarized segmentation mask or raw logits.
    
    Parameters
    ----------
    img_path : str
        Path to the image file to predict.
    model : torch.nn.Module
        The PyTorch segmentation model.
    return_logit : bool, default=False
        If True, returns the raw model logits without post-processing.
    
    Returns
    -------
    numpy.ndarray
        Segmentation mask (binary values 0 or 255) or raw logits.
    """
    img = load_img_seg(img_path)
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            img = img.cuda()
        elif torch.backends.mps.is_available():
            img = img.to('mps')
        logit = model(img)[0][0]

    if return_logit: return logit.cpu().detach().numpy()

    out = (torch.sigmoid(logit)>0.5).int()*255
    return out.cpu().detach().numpy()

def seg_predict_old(img:torch.Tensor, model:torch.nn.Module, return_logit:bool=False)->np.ndarray:
    """
    For one image path, load the image, compute the model prediction, return the prediction.
    
    **Segmentation Predictor V0**
    
    Parameters
    ----------
    img: torch.Tensor
        Image as a tensor.
    model : torch.nn.Module
        The segmentation model.
    return_logit : bool, optional
        If True, return the raw logit instead of the post-processed output.

    Returns
    -------
    numpy.ndarray
        The predicted segmentation mask or logit.
    """
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            img = img.cuda()
        elif torch.backends.mps.is_available():
            img = img.to('mps')
        img = torch.unsqueeze(img, 0)
        logit = model(img)[0]

    if return_logit: return logit.cpu().detach().numpy()

    out = (torch.sigmoid(logit)>0.5).int()*255
    return out.cpu().detach().numpy()

#---------------------------------------------------------------------------
# model predictor for segmentation with patches

class LoadImgPatch:
    """
    Class to load and preprocess an image for TorchIO-like patch-based segmentation prediction.

    **Segmentation Predictor V1**

    Designed to handle image loading, resampling, clipping, normalization, and patch sampling preparation.

    :ivar str fname: Filename of the image.
    :ivar numpy.ndarray img: Preprocessed image tensor.
    :ivar tuple[int] img_shape: Original shape of the image.
    :ivar numpy.ndarray patch_size: Patch size for sampling.
    :ivar numpy.ndarray median_spacing: Median spacing of the dataset for resampling.
    :ivar list[float] clipping_bounds: Clipping bounds for intensity.
    :ivar list[float] intensity_moments: Intensity moments for normalization.
    :ivar list[float]|tuple[float] spacing: Original image spacing.
    """

    def __init__(
        self,
        fname:str,
        patch_size:tuple[int]=(64,64,32),
        median_spacing:list[float]=[],
        clipping_bounds:list[float]=[],
        intensity_moments:list[float]=[],
        ):
        """
        Initialize the LoadImgPatch class.

        Parameters
        ----------
        fname : str
            Path to the image file.
        patch_size : tuple of int, default=(64,64,32)
            Size of the patches.
        median_spacing : list of float, default=[]
            Median spacing of the dataset to resample the image.
        clipping_bounds : list, default=[]
            Bounds for clipping the intensity values. If provided must be [min,max]
        intensity_moments : list of float, default=[]
            Mean and variance of the intensity of the images voxels in the masks regions as [mean,std]. These values are used to normalize the image. 
        """
        self.fname = fname
        self.patch_size = np.array(patch_size)
        self.median_spacing = np.array(median_spacing)
        self.clipping_bounds = clipping_bounds
        self.intensity_moments = intensity_moments
        
        # prepare image
        # load the image
        img,metadata = adaptive_imread(self.fname) # We don't replace this call since it is in a already depreciated method
        self.spacing = None if 'spacing' not in metadata.keys() else metadata['spacing']

        # store img shape (for post processing)
        self.img_shape = img.shape
        print("image shape: ",self.img_shape)
        
        # expand image dim
        if len(img.shape)==3:
            img = np.expand_dims(img, 0)
        elif len(img.shape)==4:
            # we consider as the channel dimension, the smallest dimension
            # it should be either the first or the last dim
            # if it is the last dim, then we move it to the first
            if np.argmin(img.shape)==3:
                img = np.moveaxis(img, -1, 0)
            elif np.argmin(img.shape)!=0:
                print("[Error] Invalid image shape:", img.shape)
        else:
            print("[Error] Invalid image shape:", img.shape)
    
        # preprocessing: resampling, clipping, z-normalization
        # resampling if needed
        if len(self.median_spacing)>0:
            resample = (self.median_spacing/self.spacing)[::-1] # transpose the dimension
            if resample.sum() > 0.1: # otherwise no need of spacing 
                sub = tio.Subject(img=tio.ScalarImage(tensor=img))
                sub = tio.Resample(resample)(sub)
                img = sub.img.numpy()
                print("Resampling required! From {} to {}".format(self.img_shape, img.shape))

        # clipping if needed
        if len(self.clipping_bounds)>0:
            img = np.clip(img, self.clipping_bounds[0], self.clipping_bounds[1])

        # normalize 
        if len(self.intensity_moments)>0:
            img = (img-self.intensity_moments[0])/self.intensity_moments[1]
        else:
            img = (img-img.mean())/img.std()

        # convert to tensor of float
        self.img = torch.from_numpy(img).float()
        
    def get_gridsampler(self)->tio.data.GridSampler:
        """
        Prepare image for model prediction and return a TorchIO GridSampler.

        Returns
        -------
        tio.data.GridSampler
            TorchIO GridSampler object for patch sampling.
        """
        # define the grid sampler 
        patch_overlap = np.maximum(self.patch_size//2, self.patch_size-np.array(self.img_shape))
        patch_overlap = np.ceil(patch_overlap/2).astype(int)*2
        sub = tio.Subject(img=tio.ScalarImage(tensor=self.img))
        sampler= tio.data.GridSampler(subject=sub, 
                                patch_size=self.patch_size, 
                                patch_overlap=patch_overlap,
                                padding_mode='constant')
        return sampler
    
    def post_process(self, logit:torch.Tensor)->torch.Tensor:
        """
        Resample the model output back to the original image space after prediction.

        Parameters
        ----------
        logit : torch.Tensor
            Model output tensor to be resampled.

        Returns
        -------
        torch.Tensor
            Resampled model output tensor.
        """
        if len(self.median_spacing)==0:
            return logit 
        
        resample = (self.spacing/self.median_spacing)[::-1] # transpose the dimension
        if resample.sum() > 0.1:
            sub = tio.Subject(img=tio.ScalarImage(tensor=logit))
            sub = tio.Resample(resample)(sub)
            
            # if the image has not the right size still, then try to crop it first
            if not np.array_equal(sub.img.shape[1:], self.img_shape):
                img = sub.img.tensor
                x,y,z = self.img_shape
                img = img[:,:x,:y,:z]
                
            # if the image has still not the right size still, then resize it
            if not np.array_equal(sub.img.shape[1:], self.img_shape):
                sub = tio.Resize(self.img_shape)(sub)
                img = sub.img.tensor
            return img
        else:
            return logit

def seg_predict_patch(
    img_path:str,
    model:torch.nn.Module,
    return_logit:bool=False,
    patch_size:tuple[int]=None,
    tta:bool=False,          # test time augmentation 
    median_spacing:list[float]=[],
    clipping_bounds:list[float]=[],
    intensity_moments:list[float]=[],
    use_softmax:bool=False,
    force_softmax:bool=False,
    num_workers:int=4,
    enable_autocast:bool=True, 
    keep_biggest_only:bool=False,
    )->np.ndarray:
    """
    For one image path, load the image, compute the model prediction, return the prediction.
    
    **Segmentation Predictor V1-TorchIO**
    
    Parameters
    ----------
    img_path : str
        Path to the image file.
    model : torch.nn.Module
        The segmentation model.
    return_logit : bool, default=False
        If True, return the raw logit instead of the post-processed output.
    patch_size : tuple of int, default=None
        Size of the patches for processing.
    tta : bool, default=False
        If True, apply test time augmentation.
    median_spacing : list of float, default=[]
        Median spacing of the dataset to resample the image.
    clipping_bounds : list, default=[]
        Bounds for clipping the intensity values. If provided must be [min,max]
    intensity_moments : list of float, default=[]
        Mean and variance of the intensity of the images voxels in the masks regions as [mean,std]. These values are used to normalize the image. 
    use_softmax : bool, default=False
        Flag for softmax processing.
    force_softmax : bool, default=False
        Flag to output a softmax-like output even if the model output is sigmoid-like.
    num_workers : int, default=4
        Number of workers for data loading.
    enable_autocast : bool, default=True
        Enable mixed precision.
    keep_biggest_only : bool, default=False
        If True, keep only the biggest volume in the output.

    Returns
    -------
    numpy.ndarray
        The predicted segmentation mask or logit.
    """
    if torch.cuda.is_available(): device='cuda'
    elif torch.backends.mps.is_available(): device='mps'
    else: device = 'cpu'
    enable_autocast = torch.cuda.is_available() and enable_autocast # tmp, autocast seems to work only with gpu for now... 
    print('AMP {}'.format('enabled' if enable_autocast else 'disabled'))

    img_loader = LoadImgPatch(
        img_path,
        patch_size,
        median_spacing,
        clipping_bounds,
        intensity_moments,
    )

    img = img_loader.get_gridsampler()

    model.eval()
    with torch.no_grad():
        pred_aggr = tio.inference.GridAggregator(img, overlap_mode='hann')
        patch_loader = torch.utils.data.DataLoader(
            img, 
            batch_size=2, 
            drop_last=False, 
            shuffle  =False, 
            num_workers=num_workers, 
            pin_memory =True)

        for patch in tqdm(patch_loader):
            X = patch['img'][tio.DATA]
            if torch.cuda.is_available():
                X = X.cuda()
            elif torch.backends.mps.is_available():
                X = X.to('mps')
            
            if tta: # test time augmentation: flip around each axis
                with torch.autocast(device, enabled=enable_autocast):
                    pred=model(X).cpu()
                
                # flipping tta
                dims = [[2],[3],[4]]
                for i in range(len(dims)):
                    X_flip = torch.flip(X,dims=dims[i])

                    with torch.autocast(device, enabled=enable_autocast):
                        pred_flip = model(X_flip)
                        pred += torch.flip(pred_flip, dims=dims[i]).cpu()
                    
                    del X_flip, pred_flip
                
                pred = pred/(len(dims)+1)
                del X
            else:
                with torch.autocast(device, enabled=enable_autocast):
                    pred=model(X).cpu()
                    del X

            pred_aggr.add_batch(pred, patch[tio.LOCATION])
        
        print("Prediction done!")

        print("Aggregation...")
        logit = pred_aggr.get_output_tensor().float()
        print("Aggregation done!")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # post-processing:
    print("Post-processing...")
    logit = img_loader.post_process(logit)

    if return_logit: 
        print("Post-processing done!")
        return logit

    if use_softmax:
        out = (logit.softmax(dim=0).argmax(dim=0)).int()
    elif force_softmax:
        # if the training has been done with a sigmoid activation and we want to export a softmax
        # it is possible to use `force_softmax` argument
        sigmoid = (logit.sigmoid()>0.5).int()
        softmax = (logit.softmax(dim=0).argmax(dim=0)).int()+1
        cond = sigmoid.max(dim=0).values
        out = torch.where(cond>0, softmax, 0)
    else:
        out = (logit.sigmoid()>0.5).int()
    out = out.numpy()

    # TODO: the function below is too slow
    if keep_biggest_only:
        if len(out.shape)==3:
            out = keep_biggest_volume_centered(out)
        elif len(out.shape)==4:
            tmp = []
            for i in range(out.shape[0]):
                tmp += [keep_biggest_volume_centered(out[i])]
            out = np.array(tmp)

    out = out.astype(np.byte) 
    print("Post-processing done!")
    print("Output shape:",out.shape)
    return out

#---------------------------------------------------------------------------
# new predictor

def seg_predict_patch_2(
    img:np.ndarray,
    original_shape:tuple[int],
    model:torch.nn.Module,
    patch_size:tuple[int],
    conserve_size:bool=False,
    tta:bool=False,          # test time augmentation 
    num_workers:int=4,
    enable_autocast:bool=True, 
    use_softmax:bool=True,   # DEPRECATED!
    keep_biggest_only:bool=False, # DEPRECATED!
    **kwargs, # just for handling other image metadata
    )->np.ndarray:
    """
    For one image, compute the model prediction, return the predicted logit.

    **Segmentation Predictor V2**

    Image are supposed to be preprocessed already, which is doable using biom3d.preprocess.seg_preprocessor.

    Parameters
    ----------
    img : numpy.ndarray
        The preprocessed image to predict.
    original_shape : tuple of int
        Original shape of the image.
    model : torch.nn.Module
        The segmentation model.
    patch_size : tuple of int
        Size of the patch used during training.
    conserve_size : bool, default=False
        Force the logit to be the same size as the input. May be used if intended to not use post-processing.
    tta : bool, default=False
        Test time augmentation.
    num_workers : int, default=4
        Number of workers.
    enable_autocast : bool, default=True
        Whether to use half-precision.
    use_softmax : bool, default=True
        [DEPRECATED!] Whether softmax activation has been used for training.
    keep_biggest_only : bool, default=True
        [DEPRECATED!] When true keeps the biggest object only in the output image.
    **kwargs: dict from str to any
        Just here for compatibility.

    Returns
    -------
    numpy.ndarray
        The predicted segmentation mask or logit.
    """
    if torch.cuda.is_available(): device='cuda'
    elif torch.backends.mps.is_available(): device='mps'
    else: device = 'cpu'
    enable_autocast = torch.cuda.is_available() and enable_autocast # tmp, autocast seems to work only with gpu for now... 
    print('AMP {}'.format('enabled' if enable_autocast else 'disabled'))

    # make original_shape 3D
    original_shape = original_shape[-3:]

    # get grid sampler
    overlap = 0.5
    patch_size = np.array(patch_size)

    # check that overlap is smaller or equal to image shape
    patch_overlap = np.maximum(patch_size*overlap, patch_size-np.array(img.shape[-3:]))

    # round patch_overlap
    patch_overlap = (np.ceil(patch_overlap*overlap)/overlap).astype(int)

    # if patch_overlap is equal to one of the patch_size dimension then torchio throw an error
    patch_overlap = np.minimum(patch_overlap, patch_size-1)

    sub = tio.Subject(img=tio.ScalarImage(tensor=img))
    sampler= tio.data.GridSampler(subject=sub, 
                            patch_size=patch_size, 
                            patch_overlap=patch_overlap,
                            padding_mode='constant')

    model.to(device).eval()

    with torch.no_grad():
        pred_aggr = tio.inference.GridAggregator(sampler, overlap_mode='hann')
        patch_loader = tio.SubjectsLoader(
            sampler, 
            batch_size=2, 
            drop_last=False, 
            shuffle  =False, 
            num_workers=num_workers, 
            pin_memory =True)

        for patch in tqdm(patch_loader):
            X = patch['img'][tio.DATA]
            if torch.cuda.is_available():
                X = X.cuda()
            elif torch.backends.mps.is_available():
                X = X.to('mps')
            
            if tta: # test time augmentation: flip around each axis
                with torch.autocast(device, enabled=enable_autocast):
                    pred=model(X).cpu()
                
                # flipping tta
                dims = [[2],[3],[4],[3,2],[4,2],[4,3],[4,3,2]]
                for i in range(len(dims)):
                    X_flip = torch.flip(X,dims=dims[i])

                    with torch.autocast(device, enabled=enable_autocast):
                        pred_flip = model(X_flip)
                        pred += torch.flip(pred_flip, dims=dims[i]).cpu()
                    
                    del X_flip, pred_flip
                
                pred = pred/(len(dims)+1)
                del X
            else:
                with torch.autocast(device, enabled=enable_autocast):
                    pred=model(X).cpu()
                    del X

            pred_aggr.add_batch(pred, patch[tio.LOCATION])
        
        print("Prediction done!")

        print("Aggregation...")
        logit = pred_aggr.get_output_tensor().float()
        print("Aggregation done!")
    
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # reshape the logit so it has the same size as the original image
    if conserve_size:
        return resize_3d(logit, original_shape, order=3)

    return logit

def seg_postprocessing(
        logit:np.ndarray|torch.Tensor,
        original_shape:tuple[int],
        use_softmax:bool=True,
        force_softmax:bool=False,
        keep_big_only:bool=False,
        keep_biggest_only:bool=False,
        return_logit:bool=False,
        is_2d:bool=False,
        **kwargs, # just for handling other image metadata
    ):
    """
    Post-process the logit (model output) to obtain the final segmentation mask. Can optionally remove some noise.

    Recommended to be used after biom3d.predictors.seg_predict_patch_2.

    Parameters
    ----------
    logit : torch.Tensor or numpy.ndarray
        The raw model output.
    original_shape : tuple of int
        Shape to resize the output to.
    use_softmax : bool, default=True
        Whether softmax was used for training.
    force_softmax : bool, default=False
        Whether sigmoid was used for training and intended to convert to softmax-like output.
    keep_big_only : bool, default=False
        Whether to keep the big objects only in the output. An Otsu threshold is used on the object volume distribution.
    keep_biggest_only : bool, default=False
        When true keeps the biggest **centered** object only in the output.
    return_logit : bool, optional
        Whether to return the logit. Resampling will be applied before.
    is_2d: bool, default=False
        Whether the image is in 2D, only affect resizing.
    **kwargs: dict from str to any
        Just here for compatibility.

    Raises
    ------
    AssertionError
        If logit is not a numpy.ndarray or torch.Tensor.

    Returns
    -------
    numpy.ndarray
        The post-processed segmentation mask or logit.
    """
    # make original_shape only spatial
    if is_2d: original_shape= (1,original_shape[-2],original_shape[-1])
    else: original_shape=original_shape[-3:] 
    num_classes = logit.shape[0]

    # post-processing:
    print("Post-processing...")

    if return_logit: 
        if original_shape is not None:
            if type(logit)==torch.Tensor:
                logit = logit.numpy()
            assert type(logit)==np.ndarray, "[Error] Logit must be numpy.ndarray but found {}.".format(type(logit))
            logit = resize_3d(logit, original_shape, order=3)
        print("Returning logit...")
        print("Post-processing done!")
        return logit

    if use_softmax:
        out = (logit.softmax(dim=0).argmax(dim=0)).int().numpy()        
    elif force_softmax:
        # if the training has been done with a sigmoid activation and we want to export a softmax
        # it is possible to use `force_softmax` argument
        sigmoid = (logit.sigmoid()>0.5).int()
        softmax = (logit.softmax(dim=0).argmax(dim=0)).int()+1
        cond = sigmoid.max(dim=0).values
        out = torch.where(cond>0, softmax, 0).numpy()        
    else:
        out = (logit.sigmoid()>0.5).int().numpy()
        
    # resampling
    if original_shape is not None:
        if use_softmax or force_softmax:
            out = resize_3d(np.expand_dims(out,0), original_shape, order=1, is_msk=True).squeeze()
        else: 
            out = resize_3d(out, original_shape, order=1, is_msk=True)
    
    if keep_big_only and keep_biggest_only:
        print("[Warning] Incompatible options 'keep_big_only' and 'keep_biggest_only' have both been set to True. Please deactivate one! We consider here only 'keep_biggest_only'.")
    # TODO: the function below is too slow
    if keep_biggest_only or keep_big_only:
        fct = keep_biggest_volume_centered if keep_biggest_only else keep_big_volumes
        if use_softmax: # then one-hot encode the net output
            out = (np.arange(num_classes)==out[...,None]).astype(int)
            out = np.rollaxis(out, -1)

        if len(out.shape)==3:
            out = fct(out)
        elif len(out.shape)==4:
            tmp = []
            for i in range(out.shape[0]):
                tmp += [fct(out[i])]
            out = np.array(tmp)
            
        if use_softmax: # set back to non-one-hot encoded
            out = out.argmax(0)
            
    if is_2d:
        logit = logit.squeeze(1) # Remove Z dim

    out = out.astype(np.uint8)    
    
    print("Post-processing done!")
    print("Output shape:",out.shape)
    return out

#---------------------------------------------------------------------------
