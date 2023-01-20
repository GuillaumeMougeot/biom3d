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

from biom3d.utils import keep_biggest_volume_centered, adaptive_imread

#---------------------------------------------------------------------------
# model predictor for segmentation

def load_img_seg(fname):
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
    for one image path, load the image, compute the model prediction, return the prediction
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
        img,self.spacing = adaptive_imread(self.fname)

        # store img shape (for post processing)
        self.img_shape = img.shape
        print("image shape: ",self.img_shape)
        
        # expand dims
        img = np.expand_dims(img, 0)
    
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
        Prepare image for model prediction and return a tio.data.GridSampler
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
        resampling back the image after model prediction
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
    num_workers=4,
    enable_autocast=True, 
    keep_biggest_only=False,
    ):
    """
    for one image path, load the image, compute the model prediction, return the prediction
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enable_autocast = torch.cuda.is_available() and enable_autocast # tmp, autocast seems to work only with gpu for now... 
    print('AMP {}'.format('enable' if enable_autocast else 'disable'))

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
        return logit.numpy()

    if use_softmax:
        out = (logit.softmax(dim=0).argmax(dim=0)).int()
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
