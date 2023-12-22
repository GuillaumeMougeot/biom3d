#---------------------------------------------------------------------------
# model predictors
# these functions returns the model predictions for a single input 
# TODO: re-structure with classes maybe?
#---------------------------------------------------------------------------

import torch 
import torchio as tio
import numpy as np
from skimage.io import imread
from tqdm import tqdm
# from scipy.ndimage.filters import gaussian_filter

from biom3d.utils import keep_biggest_volume_centered, adaptive_imread, resize_3d, keep_big_volumes

#---------------------------------------------------------------------------
# model predictor for segmentation

def load_img_seg(fname):
    """
    *Segmentation Predictor V1*
    
    Load and preprocess a single image for segmentation prediction.

    Parameters
    ----------
    fname : str
        Path to the image file.

    Returns
    -------
    torch.Tensor
        Preprocessed image as a PyTorch tensor.
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
    img_path,
    model,
    return_logit=False,
    ):
    """
    *Segmentation Predictor V1*

    For one image path, load the image, compute the model prediction, return the prediction.
    
    Parameters
    ----------
    img_path : str
        Path to the image file.
    model : torch.nn.Module
        The segmentation model.
    return_logit : bool, optional
        If True, return the raw logit instead of the post-processed output.

    Returns
    -------
    numpy.ndarray
        The predicted segmentation mask or logit.
    """
    img = load_img_seg(img_path)
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            img = img.cuda()
        logit = model(img)[0][0]

    if return_logit: return logit.cpu().detach().numpy()

    out = (torch.sigmoid(logit)>0.5).int()*255
    return out.cpu().detach().numpy()

def seg_predict_old(img, model, return_logit=False):
    """
    *Segmentation Predictor V0*

    For one image path, load the image, compute the model prediction, return the prediction.
    
    Parameters
    ----------
    img_path : str
        Path to the image file.
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

    Parameters
    ----------
    fname : str
        Path to the image file.
    patch_size : tuple of int
        Size of the patches.
    median_spacing : list, optional
        Spacing to resample the image.
    clipping_bounds : list, optional
        Bounds for clipping the intensity values.
    intensity_moments : list, optional
        Moments for intensity normalization.
    """
    def __init__(
        self,
        fname,
        patch_size=(64,64,32),
        median_spacing=[],
        clipping_bounds=[],
        intensity_moments=[],
        ):
        
        self.fname = fname
        self.patch_size = np.array(patch_size)
        self.median_spacing = np.array(median_spacing)
        self.clipping_bounds = clipping_bounds
        self.intensity_moments = intensity_moments
        
        # prepare image
        # load the image
        img,metadata = adaptive_imread(self.fname)
        self.spacing = None if not 'spacing' in metadata.keys() else metadata['spacing']

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
        
    def get_gridsampler(self):
        """
        Prepare image for model prediction and return a tio.data.GridSampler.

        Returns
        -------
        tio.data.GridSampler
            TorchIO GridSampler.
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
    
    def post_process(self, logit):
        """
        Resampling back the image after model prediction.

        Parameters
        ----------
        logit : torch.Tensor
            Model output.
        
        Returns
        -------
        torch.Tensor
            Resampled logit.
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
    img_path,
    model,
    return_logit=False,
    patch_size=None,
    tta=False,          # test time augmentation 
    median_spacing=[],
    clipping_bounds=[],
    intensity_moments=[],
    use_softmax=False,
    force_softmax=False,
    num_workers=4,
    enable_autocast=True, 
    keep_biggest_only=False,
    ):
    """
    *Segmentation Predictor V1-TorchIO*

    For one image path, load the image, compute the model prediction, return the prediction.
    
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
    median_spacing : list, default=[]
        Spacing to resample the image.
    clipping_bounds : list, default=[]
        Bounds for clipping the intensity values.
    intensity_moments : list, default=[]
        Moments for intensity normalization.
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    img,
    original_shape,
    model,
    patch_size,
    conserve_size=False,
    tta=False,          # test time augmentation 
    num_workers=4,
    enable_autocast=True, 
    use_softmax=True,   # DEPRECATED!
    keep_biggest_only=False, # DEPRECATED!
    **kwargs, # just for handling other image metadata
    ):
    """
    *Segmentation Predictor V2*

    For one image, compute the model prediction, return the predicted logit.

    Image are supposed to be preprocessed already, which is doable using biom3d.preprocess.seg_preprocessor.
    
    Parameters
    ----------
    img : numpy.ndarray
        The preprocessed image to predict.
    original_shape : tuple
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
        
      
    Returns
    -------
    numpy.ndarray
        The predicted segmentation mask or logit.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        patch_loader = torch.utils.data.DataLoader(
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

    # reshape the logit so it has the same size as the original image
    if conserve_size:
        return resize_3d(logit, original_shape, order=3)

    return logit

def seg_postprocessing(
        logit,
        original_shape,
        use_softmax=True,
        force_softmax=False,
        keep_big_only=False,
        keep_biggest_only=False,
        return_logit=False,
        **kwargs, # just for handling other image metadata
    ):
    """
    Post-process the logit (model output) to obtain the final segmentation mask. Can optionally remove some noise. 

    Recommanded to be used after biom3d.predictors.seg_predict_patch_2.
  
    Parameters
    ----------
    logit : torch.Tensor
        The raw model output.
    original_shape : tuple
        Shape to resize the output to.
    use_softmax : bool, default=True
        Whether softmax was used for training.
    force_softmax : bool, default=False
        Whether sigmoid was used for training and intended to convert to softmax-like output.
    keep_big_only : bool, default=False
        Whether to keep the big objects only in the output. An Otsu threshold is used on the object volume distribution.
    keep_biggest_only : bool, default=False
        When true keeps the biggest object only in the output.
    return_logit : bool, optional
        Whether to return the logit. Resampling will be applied before.

    Returns
    -------
    numpy.ndarray
        The post-processed segmentation mask or logit.
    """
    # make original_shape 3D
    original_shape = original_shape[-3:]
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

    out = out.astype(np.uint8)    
    
    print("Post-processing done!")
    print("Output shape:",out.shape)
    return out

#---------------------------------------------------------------------------
