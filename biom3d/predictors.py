#---------------------------------------------------------------------------
# model predictors
# these functions returns the model predictions for a single input 
# TODO: re-structure with classes maybe?
#---------------------------------------------------------------------------

import torch 
import torchio as tio
import numpy as np
from tqdm import tqdm

import utils

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
        # img = imread(fname)
        img,self.spacing = utils.sitk_imread(self.fname)

        # store img shape (for post processing)
        self.img_shape = img.shape
        
        # expand dims
        img = np.expand_dims(img, 0)
    
        # preprocessing: resampling, clipping, z-normalization
        # resampling if needed
        if median_spacing and len(self.median_spacing)>0:
            resample = (self.median_spacing/self.spacing)[::-1] # transpose the dimension
            if resample.sum() > 0.1: # otherwise no need of spacing 
                sub = tio.Subject(img=tio.ScalarImage(tensor=img))
                sub = tio.Resample(resample)(sub)
                img = sub.img.numpy()
                print("Resampling required! From {} to {}".format(self.img_shape, img.shape))

        # clipping if needed
        if clipping_bounds and len(self.clipping_bounds)>0:
            img = np.clip(img, self.clipping_bounds[0], self.clipping_bounds[1])

        # normalize 
        if intensity_moments and len(self.intensity_moments)>0:
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
        if self.median_spacing and len(self.median_spacing)==0:
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
    ):
    """
    for one image path, load the image, compute the model prediction, return the prediction
    """
    # img = load_img_seg_patch(img_path, patch_size)
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
            num_workers=4, 
            pin_memory =True)

        for patch in tqdm(patch_loader):
            X = patch['img'][tio.DATA]
            if torch.cuda.is_available():
                X = X.cuda()
            
            if tta: # test time augmentation: flip around each axis
                # Xs = [X] + [torch.flip(X,dims=[i]) for i in range(2,5)]
                # with torch.cuda.amp.autocast():
                #     preds = [model(x).cpu() for x in Xs]
                # preds = [preds[0]]+[torch.flip(preds[i], dims=[i+1]) for i in range(1,4)]
                # pred = torch.mean(torch.stack(preds), dim=0)

                with torch.cuda.amp.autocast():
                    pred=model(X).cpu()
                    # pred+=torch.flip(pred, dims=[1]).cpu()
                
                # flipping tta
                # dims = [[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]]
                dims = [[2],[3],[4]]
                # dims = [[2]]
                for i in range(len(dims)):
                    X_flip = torch.flip(X,dims=dims[i])
                    with torch.cuda.amp.autocast():
                        pred_flip = model(X_flip)
                        pred += torch.flip(pred_flip, dims=dims[i]).cpu()
                    
                    del X_flip, pred_flip
                    torch.cuda.empty_cache()
                
                pred = pred/(len(dims)+1)
                # pred = pred/3
                del X; torch.cuda.empty_cache()
            else:
                with torch.cuda.amp.autocast():
                    pred=model(X).cpu()
                    del X; torch.cuda.empty_cache()
            pred_aggr.add_batch(pred, patch[tio.LOCATION])
        
        logit = pred_aggr.get_output_tensor().float()
    
    torch.cuda.empty_cache()

    # post-processing:
    logit = img_loader.post_process(logit)

    if return_logit: return logit.numpy()

    if use_softmax:
        # out = (logit.softmax(dim=0)>0.5).int()[1:]
        out = (logit.softmax(dim=0).argmax(dim=0)).int()
    else:
        out = (logit.sigmoid()>0.5).int()
    out = out.numpy()
    out = out.astype(np.byte)
    # out = utils.keep_center_only(out)
    print("output shape",out.shape)
    return out

#---------------------------------------------------------------------------
