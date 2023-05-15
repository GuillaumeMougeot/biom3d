#---------------------------------------------------------------------------
# Dataset preparation to fasten the training
#   -normalization
#   -expand dims and one_hot encoding
#   -saving to tif file
#---------------------------------------------------------------------------

import numpy as np
import os 
import pickle # for foreground storage
from tqdm import tqdm
import argparse
import tifffile

from biom3d.auto_config import auto_config, data_fingerprint
from biom3d.utils import adaptive_imread, one_hot_fast, resize_3d, save_python_config

np.random.seed(42)

#---------------------------------------------------------------------------
# 3D segmentation preprocessing
# Nifti convertion (Medical segmentation decathlon)
# normalization: z-score
# resampling
# intensity normalization
# one_hot encoding

def resize_img_msk(img, output_shape, msk=None):
    new_img = resize_3d(img, output_shape, order=3)
    if msk is not None:
        new_msk = resize_3d(msk, output_shape, is_msk=True, order=1)
        return new_img, new_msk
    else: 
        return new_img

def get_resample_shape(input_shape, spacing, median_spacing):
    if len(input_shape)==4:
        input_shape=input_shape[1:]
    return np.round(((spacing/median_spacing)[::-1]*input_shape)).astype(int)


def sanity_check(msk, num_classes=None):
    """Check if the mask is correctly annotated.
    """
    uni = np.sort(np.unique(msk))
    if num_classes is None:
        num_classes = len(uni)
        
    assert type(num_classes)==int
    assert num_classes >= 2
    
    if len(msk.shape)==4:
        # if we have 4 dimensions in the mask, we consider it one-hot encoded
        # and thus we perform a sanity check for each channel
        for i in range(msk.shape[0]):
            sanity_check(msk[i], num_classes=2)
            
    cls = np.arange(num_classes)
    if np.all(uni==cls):
        # the mask is correctly annotated
        return msk
    else:
        # there is something wrong with the annotations
        # depending on the case we make automatic adjustments
        # or we through an error message
        print("[Warning] There is something abnormal with the annotations. Each voxel value must be in range {} but is in range {}.".format(cls, uni))
        if num_classes==2:
            print("[Warning] Applying a thresholding.")
            # then we apply a threshold to the data
            # for instance: unique [2,127,232] becomes [0,1], 0 being 2 and 1 being 127 and 232
            return (msk > msk.min()).astype(np.uint8)
        elif np.all(np.isin(uni, cls)):
            # then one label is missing in the current mask... but it should work
            print("[Warning] One or more labels are missing.")
            return msk
        elif len(uni)==num_classes:
            # then we re-annotate the unique values in the mask
            # for instance: unique [2,127,232] becomes [0,1,2]
            print("[Warning] Annotation are wrong in the mask, we will re-annotate the mask.")
            new_msk = np.zeros(msk.shape, dtype=msk.dtype)
            for i,c in enumerate(uni):
                new_msk[msk == c] = i
            return new_msk
        else:
            # case like [2,18,128,254] where the number of classes should be 3 are impossible to decide...
            print("[Error] There is an error in the labels that could not be solved automatically.")
            raise RuntimeError
        

class Preprocessing:
    """A helper class to transform nifti (.nii.gz) and Tiff (.tif or .tiff) images to .tif format and to normalize them.

    Parameters
    ----------
    img_dir : str
        Path to the input image folder
    img_outdir : str
        Path to the output image folder
    msk_dir : str, optional
        Path to the input mask folder
    msk_outdir : str, optional
        Path to the output mask folder
    fg_outdir : str, optional
        Foreground location, eventually later used by the dataloader.
    num_classes : int, optional
        Number of classes (channel) in the masks. Required by the 
    remove_bg : bool, default=True
        Whether to remove the background in the one-hot encoded mask. Remove the background is done when training with sigmoid activations instead of softmax.
    median_spacing : list, optional
        A list of length 3 containing the median spacing of the input images. Median_spacing must not be transposed: for example, median_spacing might be [0.8, 0.8, 2.5] if median shape of the training image is [40,224,224].
    clipping_bounds : list, optional
        A list of length 2 containing the intensity clipping boundary. In nnUNet implementation it corresponds to the 0.5 an 99.5 percentile of the intensities of the voxels of the training images located inside the masks regions.
    intensity_moments : list, optional
        Mean and variance of the intensity of the images voxels in the masks regions. These values are used to normalize the image. 
    use_tif : bool, default=True
        Use tif format to save the preprocessed images instead of npy format.
    split_rate_for_single_img : float, default=0.2
        If a single image is present in image/mask folders, then the image/mask are split in 2 portions of size split_rate_for_single_img*largest_dimension for validation and split_rate_for_single_img*(1-largest_dimension) for training.
    """
    def __init__(
        self,
        img_dir,
        img_outdir = None,
        msk_dir = None, # if None, only images are preprocesses not the masks
        msk_outdir = None,
        fg_outdir = None, # foreground location, eventually used by the dataloader
        num_classes = None, # just for debug when empty masks are provided
        use_one_hot = False,
        remove_bg = False, # keep the background in labels 
        median_spacing=[],
        clipping_bounds=[],
        intensity_moments=[],
        use_tif=False, # use tif instead of npy 
        split_rate_for_single_img=0.25,
        ):
        assert img_dir!='', "[Error] img_dir must not be empty."

        # fix bug path/folder/ to path/folder
        if os.path.basename(img_dir)=='':
            img_dir = os.path.dirname(img_dir)
        if msk_dir is not None and os.path.basename(msk_dir)=='':
            msk_dir = os.path.dirname(msk_dir)
        
        self.img_dir=img_dir
        self.msk_dir=msk_dir
        self.img_fnames=os.listdir(self.img_dir)

        if img_outdir is None: # name the out dir the same way as the input and add the _out suffix
            img_outdir = img_dir+'_out'
            print("Image output path:", img_outdir)
        if msk_dir is not None and msk_outdir is None:
            msk_outdir = msk_dir+'_out'
            print("Mask output path:", msk_outdir)
            if fg_outdir is None:
                # get parent directory of mask dir
                fg_outdir = os.path.join(os.path.dirname(msk_dir), 'fg_out')
                print("Foreground output path:", fg_outdir)

        self.img_outdir=img_outdir 
        self.msk_outdir=msk_outdir
        self.fg_outdir =fg_outdir

        # create output directory if needed
        if not os.path.exists(self.img_outdir):
            os.makedirs(self.img_outdir, exist_ok=True)
        if msk_dir is not None and not os.path.exists(self.msk_outdir):
            os.makedirs(self.msk_outdir, exist_ok=True)
        if msk_dir is not None and not os.path.exists(self.fg_outdir):
            os.makedirs(self.fg_outdir, exist_ok=True)

        self.num_classes = num_classes
        self.num_channels = 1

        self.remove_bg = remove_bg

        self.median_spacing = np.array(median_spacing)
        self.clipping_bounds = np.array(clipping_bounds)
        self.intensity_moments = intensity_moments
        self.use_tif = use_tif

        self.split_rate_for_single_img = split_rate_for_single_img

        self.use_one_hot = use_one_hot
        
    def _split_single(self):
        """
        if there is only a single image/mask in each folder, then split them both in two portions with self.split_rate_for_single_img
        """
        # set image and mask name
        img_fname = self.img_fnames[0]
        img_path = os.path.join(self.img_dir, img_fname)
        msk_path = os.path.join(self.msk_dir, img_fname) # mask must be present

        # read image and mask
        img,_ = adaptive_imread(img_path)
        msk,_ = adaptive_imread(msk_path)

        # determine the slicing indices to crop an image along its maximum dimension
        idx = lambda start,end,shape: tuple(slice(s) if s!=max(shape) else slice(start,end) for s in shape)

        # slicing indices of the image
        # validation is cropped along its largest dimension in the interval [0, self.split_rate_for_single_img*largest_dim]
        # training is cropped along its largest dimension in the interval [self.split_rate_for_single_img*largest_dim, largest_dim]
        s = max(img.shape)
        val_img_idx = idx(start=0, end=int(np.floor(self.split_rate_for_single_img*s)), shape=img.shape)
        train_img_idx = idx(start=int(np.floor(self.split_rate_for_single_img*s)), end=s, shape=img.shape)

        # idem for the mask indices
        s = max(msk.shape)
        val_msk_idx = idx(start=0, end=int(np.floor(self.split_rate_for_single_img*s)), shape=msk.shape)
        train_msk_idx = idx(start=int(np.floor(self.split_rate_for_single_img*s)), end=s, shape=msk.shape)

        # crop the images and masks
        val_img = img[val_img_idx]
        train_img = img[train_img_idx]
        val_msk = msk[val_msk_idx]
        train_msk = msk[train_msk_idx]

        # save the images and masks 
        # validation names start with a 0
        # training names start with a 1

        # validation
        val_img_name = "0_"+os.path.basename(img_path).split('.')[0]
        val_img_name += '.tif' if self.use_tif else '.npy'

        val_img_path = os.path.join(self.img_outdir, val_img_name)
        if self.use_tif:
            tifffile.imwrite(val_img_path, val_img, compression=('zlib', 1))
        else:
            np.save(val_img_path, val_img)

        val_msk_path = os.path.join(self.msk_outdir, val_img_name)
        if self.use_tif:
            tifffile.imwrite(val_msk_path, val_msk, compression=('zlib', 1))
        else:
            np.save(val_msk_path, val_msk)

        # training save
        train_img_name = "1_"+os.path.basename(img_path).split('.')[0]
        train_img_name += '.tif' if self.use_tif else '.npy'

        train_img_path = os.path.join(self.img_outdir, train_img_name)
        if self.use_tif:
            tifffile.imwrite(train_img_path, train_img, compression=('zlib', 1))
        else:
            np.save(train_img_path, train_img)

        train_msk_path = os.path.join(self.msk_outdir, train_img_name)
        if self.use_tif:
            tifffile.imwrite(train_msk_path, train_msk, compression=('zlib', 1))
        else:
            np.save(train_msk_path, train_msk)

        # replace self.img_fnames and self.img_dir/self.msk_dir
        self.img_fnames = os.listdir(self.img_outdir)
        self.img_dir = self.img_outdir
        self.msk_dir = self.msk_outdir

    @staticmethod
    def run_single(
        img_path, 
        msk_path=None,
        num_classes=None,
        use_one_hot = False,
        remove_bg = False, 
        median_spacing=[],
        clipping_bounds=[],
        intensity_moments=[],
        ):

        do_msk = msk_path is not None

        # read image and mask
        img,spacing = adaptive_imread(img_path)
        if do_msk: 
            msk,_ = adaptive_imread(msk_path)
            # sanity check
            msk = sanity_check(msk, num_classes)

        # keep the input shape, used for preprocessing before prediction
        original_shape = img.shape
        
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

        # one hot encoding for the mask if needed
        if do_msk and len(msk.shape)!=4: 
            if use_one_hot:
                msk = one_hot_fast(msk, num_classes)
                if remove_bg:
                    msk = msk[1:]
            else:
                msk = np.expand_dims(msk, 0)
        elif do_msk and len(msk.shape)==4:
            # normalize each channel
            msk = (msk > msk.min()).astype(np.uint8)

        assert len(img.shape)==4
        if do_msk: assert len(msk.shape)==4

        # clip img
        if len(clipping_bounds)>0:
            img = np.clip(img, clipping_bounds[0], clipping_bounds[1])

        # normalize the image and msk
        # z-score normalization for the image
        if len(intensity_moments)>0:
            img = (img-intensity_moments[0])/intensity_moments[1]
        else:
            img = (img-img.mean())/img.std()
        
        # enhance contrast
        # img = exposure.equalize_hist(img)

        # range image in [-1, 1]
        # img = (img - img.min())/(img.max()-img.min()) * 2 - 1

        # resample the image and mask if needed
        if len(median_spacing)>0:
            output_shape = get_resample_shape(img.shape, spacing, median_spacing)
            if do_msk:
                # img, msk = resample_img_msk(img, msk, spacing, median_spacing)
                img, msk = resize_img_msk(img, msk=msk, output_shape=output_shape)
            else:
                # img = resample_with_spacing(img, spacing, median_spacing, order=3)
                img = resize_3d(img, output_shape)

        # set image type
        img = img.astype(np.float32)
        if do_msk: msk = msk.astype(np.uint16)
        
        # foreground computation
        if do_msk:
            fg={}
            if use_one_hot: start = 0 if remove_bg else 1
            else: start = 1
            for i in range(start,len(msk) if use_one_hot else msk.max()+1):
                fgi = np.argwhere(msk[i] == 1) if use_one_hot else np.argwhere(msk[0] == i)
                if len(fgi)>0:
                    num_samples = min(len(fgi), 10000)
                    fgi_idx = np.random.choice(np.arange(len(fgi)), size=num_samples, replace=False)
                    fgi = fgi[fgi_idx,:]
                else:
                    fgi = []
                fg[i] = fgi

            if len(fg)==0:
                print("[Warning] Empty foreground!")

        # return
        if do_msk:
            return img, msk, fg 
        else:
            return img, {'original_shape': original_shape}
    
    def run(self):
        """Start the preprocessing.
        """
        print("Preprocessing...")
        # if there is only a single image/mask, then split them both in two portions
        if len(self.img_fnames)==1 and self.msk_dir is not None:
            print("Single image found per folder. Split the images...")
            self._split_single()
            
        for i in tqdm(range(len(self.img_fnames))):
            # set image and mask name
            img_fname = self.img_fnames[i]
            img_path = os.path.join(self.img_dir, img_fname)
            if self.msk_dir is not None: msk_path = os.path.join(self.msk_dir, img_fname)

            if self.msk_dir is not None:
                img, msk, fg = self.run_single(
                    img_path            =img_path, 
                    msk_path            =msk_path,
                    num_classes         =self.num_classes,
                    use_one_hot         =self.use_one_hot,
                    remove_bg           =self.remove_bg, 
                    median_spacing      =self.median_spacing,
                    clipping_bounds     =self.clipping_bounds,
                    intensity_moments   =self.intensity_moments,)
            else:
                img, _ = self.run_single(
                    img_path            =img_path, 
                    msk_path            =None,
                    median_spacing      =self.median_spacing,
                    clipping_bounds     =self.clipping_bounds,
                    intensity_moments   =self.intensity_moments,)

            # sanity check to be sure that all images have the save number of channel
            s = img.shape
            if len(s)==4: # only for images with 4 dimensionalities
                if i==0: self.num_channels = s[0]
                else: assert len(s)==4 and self.num_channels==s[0], "[Error] Not all images have {} channels. Problematic image: {}".format(self.num_channels, img_path)

            # save the image and the mask as tif
            img_fname = os.path.basename(img_path).split('.')[0]
            # save image

            # save image as tif
            if self.use_tif:
                img_out_path = os.path.join(self.img_outdir, img_fname+'.tif')
                tifffile.imwrite(img_out_path, img, compression=('zlib', 1))
                # imsave(img_out_path, img)
                # tifffile.imwrite(img_out_path, img) # no compression --> increased training speed!
            # save image as npy
            else:
                img_out_path = os.path.join(self.img_outdir, img_fname+'.npy')
                np.save(img_out_path, img)

            # save mask
            if self.msk_outdir is not None: 
                # imsave(msk_out_path, msk)
                # save image as tif
                if self.use_tif:
                    msk_out_path = os.path.join(self.msk_outdir, img_fname+'.tif')
                    tifffile.imwrite(msk_out_path, msk, compression=('zlib', 1))
                    # imsave(msk_out_path, msk)
                    # tifffile.imwrite(msk_out_path, msk) # no compression --> increased training speed!
                # save image as npy
                else:
                    msk_out_path = os.path.join(self.msk_outdir, img_fname+'.npy')
                    np.save(msk_out_path, msk)

                # store it in a pickle format
                fg_file = os.path.join(self.fg_outdir, img_fname+'.pkl')
                with open(fg_file, 'wb') as handle:
                    pickle.dump(fg, handle, protocol=pickle.HIGHEST_PROTOCOL)

                
        print("Done preprocessing!")

#---------------------------------------------------------------------------

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Dataset preprocessing for training purpose.")
    parser.add_argument("--img_dir", type=str,
        help="Path of the images directory")
    parser.add_argument("--msk_dir", type=str, default=None,
        help="(default=None) Path to the masks/labels directory")
    parser.add_argument("--img_outdir", type=str, default=None,
        help="(default=None) Path to the directory of the preprocessed images")
    parser.add_argument("--msk_outdir", type=str, default=None,
        help="(default=None) Path to the directory of the preprocessed masks/labels")
    parser.add_argument("--num_classes", type=int, default=1,
        help="(default=1) Number of classes (types of objects) in the dataset. The background is not included.")
    parser.add_argument("--config_dir", type=str, default='configs/',
        help="(default=\'configs/\') Configuration folder to save the auto-configuration.")
    parser.add_argument("--base_config", type=str, default=None,
        help="(default=None) Optional. Path to an existing configuration file which will be updated with the preprocessed values.")
    parser.add_argument("--desc", type=str, default='unet_default',
        help="(default=unet_default) Optional. A name used to describe the model.")
    parser.add_argument("--use_tif", default=False,  action='store_true', dest='use_tif',
        help="(default=False) Whether to use tif format to save the preprocessed images instead of npy format. Tif files are easily readable with viewers such as Napari and takes fewer disk space but are slower to load and may slow down the training process.") 
    parser.add_argument("--use_one_hot", default=False,  action='store_true', dest='use_one_hot',
        help="(default=False) Whether to use one hot encoding of the mask. Can slow down the training.") 
    parser.add_argument("--remove_bg", default=False,  action='store_true', dest='remove_bg',
        help="(default=False) If use one hot, remove the background in masks. Remove the bg to use with sigmoid activation maps (not softmax).") 
    parser.add_argument("--no_auto_config", default=False,  action='store_true', dest='no_auto_config',
        help="(default=False) For debugging, deactivate auto-configuration.") 
    parser.add_argument("--ct_norm", default=False,  action='store_true', dest='ct_norm',
        help="(default=False) Whether to use CT-Scan normalization routine (cf. nnUNet).") 
    parser.add_argument("--skip_preprocessing", default=False,  action='store_true', dest='skip_preprocessing',
        help="(default=False) Whether to skip the preprocessing. Only for debugging.") 
    args = parser.parse_args()

    if args.ct_norm:
        print("Computing data fingerprint for CT normalization...")
        median_size, median_spacing, mean, std, perc_005, perc_995 = data_fingerprint(args.img_dir, args.msk_dir)
        clipping_bounds = [perc_005, perc_995]
        intensity_moments = [mean, std]
        print("Done!")
    else:
        median_size = None
        median_spacing = []
        clipping_bounds = []
        intensity_moments = []

    p=Preprocessing(
        img_dir=args.img_dir,
        msk_dir=args.msk_dir,
        img_outdir=args.img_outdir,
        msk_outdir=args.msk_outdir,
        num_classes=args.num_classes+1,
        use_one_hot=args.use_one_hot,
        remove_bg=args.remove_bg,
        use_tif=args.use_tif,
        median_spacing=median_spacing,
        clipping_bounds=clipping_bounds,
        intensity_moments=intensity_moments,
    )

    if not args.skip_preprocessing:
        p.run()

    if not args.no_auto_config:
        print("Start auto-configuration")
        

        batch, aug_patch, patch, pool = auto_config(median=median_size, img_dir=args.img_dir if median_size is None else None)

        config_path = save_python_config(
            config_dir=args.config_dir,
            base_config=args.base_config,

            # store hyper-parameters in the config file:
            IMG_DIR=p.img_outdir,
            MSK_DIR=p.msk_outdir,
            FG_DIR=p.fg_outdir,
            NUM_CLASSES=args.num_classes,
            NUM_CHANNELS=p.num_channels,
            BATCH_SIZE=batch,
            AUG_PATCH_SIZE=aug_patch,
            PATCH_SIZE=patch,
            NUM_POOLS=pool,
            MEDIAN_SPACING=median_spacing,
            CLIPPING_BOUNDS=clipping_bounds,
            INTENSITY_MOMENTS=intensity_moments,
            DESC=args.desc,
        )

        print("Auto-config done! Configuration saved in: ", config_path)

#---------------------------------------------------------------------------

