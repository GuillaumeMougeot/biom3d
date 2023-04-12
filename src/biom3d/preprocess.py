#---------------------------------------------------------------------------
# Dataset preparation to fasten the training
#   -normalization
#   -expand dims and one_hot encoding
#   -saving to tif file
#---------------------------------------------------------------------------

import numpy as np
import os 
# import torchio as tio
from tqdm import tqdm
import argparse
from skimage.io import imread
import SimpleITK as sitk
import tifffile
from numba import njit

#---------------------------------------------------------------------------
# Nifti imread

def sitk_imread(img_path):
    """
    image reader for nii.gz files
    """
    img = sitk.ReadImage(img_path)
    img_np = sitk.GetArrayFromImage(img)
    return img_np, np.array(img.GetSpacing())

def adaptive_imread(img_path):
    """
    use skimage imread or sitk imread depending on the file extension:
    .tif --> skimage.io.imread
    .nii.gz --> SimpleITK.imread
    """
    extension = img_path[img_path.rfind('.'):]
    if extension == ".tif":
        return imread(img_path), []
    elif extension == ".npy":
        return np.load(img_path), []
    else:
        return sitk_imread(img_path)

#---------------------------------------------------------------------------
# Nifti convertion (Medical segmentation decathlon)
# normalization: z-score
# one_hot encoding
# no resampling yet

def one_hot(values, num_classes=None):
    """
    transform the values np.array into a one_hot encoded
    """
    if num_classes==None: n_values = np.max(values) + 1
    else: n_values = num_classes
        
    # WARNING! potential bug if we have 255 label
    # this function normalize the values to 0,1 if it founds that the maximum of the values if 255
    if values.max()==255: values = (values / 255).astype(np.int64) 
    
    # re-order values if needed
    # for examples if unique values are [2,124,178,250] then they will be changed to [0,1,2,3]
    uni, inv = np.unique(values, return_inverse=True)
    if np.array_equal(uni, np.arange(len(uni))):
        values = np.arange(len(uni))[inv].reshape(values.shape)
        
    out = np.eye(n_values)[values]
    return np.moveaxis(out, -1, 0).astype(np.int64)

@njit
def one_hot_fast(values, num_classes=None):
    """
    transform the 'values' array into a one_hot encoded one

    Warning ! If the number of unique values in the input array is lower than the number of classes, then it will consider that the array values are all between zero and `num_classes`. If one value is greater than `num_classes`, then it will add missing values systematically after the maximum value, which could not be the expected behavior. 
    """
    # get unique values
    uni = np.sort(np.unique(values)).astype(np.uint8)

    if num_classes==None: 
        n_values = len(uni)
    else: 
        n_values = num_classes
    
        # if the expected number of class is two then apply a threshold
        if n_values==2 and (len(uni)>2 or uni.max()>1):
            print("[Warning] The number of expected values is 2 but the maximum value is higher than 1. Threshold will be applied.")
            values = (values>uni[0]).astype(np.uint8)
            uni = np.array([0,1]).astype(np.uint8)
        
        # add values if uni is incomplete
        if len(uni)<n_values: 
            # if the maximum value of the array is greater than n_value, it might be an error but still, we add values in the end.
            if values.max() >= n_values:
                print("[Warning] The maximum values in the array is greater than the provided number of classes, this might be unexpected and might cause issues.")
                while len(uni)<n_values:
                    uni = np.append(uni, np.uint8(uni[-1]+1))
            # add missing values in the array by considering that each values are in 0 and n_value
            else:
                uni = np.arange(0,n_values).astype(np.uint8)
        
    # create the one-hot encoded matrix
    out = np.zeros((n_values, *values.shape), dtype=np.uint8)
    for i in range(n_values):
        out[i] = (values==uni[i]).astype(np.uint8)
    return out

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
        num_classes = None, # just for debug when empty masks are provided
        remove_bg = False, # keep the background in labels 
        median_spacing=[],
        clipping_bounds=[],
        intensity_moments=[],
        use_tif=True, # use tif instead of npy 
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
        if msk_dir is not None and msk_outdir is None:
            msk_outdir = msk_dir+'_out'

        self.img_outdir=img_outdir 
        self.msk_outdir=msk_outdir

        # create output directory if needed
        if not os.path.exists(self.img_outdir):
            os.makedirs(self.img_outdir, exist_ok=True)
        if msk_dir is not None and  not os.path.exists(self.msk_outdir):
            os.makedirs(self.msk_outdir, exist_ok=True)

        self.num_classes = num_classes

        self.remove_bg = remove_bg

        self.median_spacing = np.array(median_spacing)
        self.clipping_bounds = np.array(clipping_bounds)
        self.intensity_moments = intensity_moments
        self.use_tif = use_tif

        # self.one_hot = tio.OneHot()
        self.split_rate_for_single_img = split_rate_for_single_img
        
    def _split_single(self):
        """
        if there is only a single image/mask in each folder, then split them both in two portions with self.split_rate_for_single_img
        """
        # set image and mask name
        img_fname = self.img_fnames[0]
        img_path = os.path.join(self.img_dir, img_fname)
        if self.msk_dir: msk_path = os.path.join(self.msk_dir, img_fname)

        # read image and mask
        img,_ = adaptive_imread(img_path)
        if self.msk_dir: msk,_ = adaptive_imread(msk_path)

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

    
    def run(self):
        """Start the preprocessing.
        """
        print("Preprocessing...")
        # if there is only a single image/mask, then split them both in two portions
        if len(self.img_fnames)==1:
            print("Single image found per folder. Split the images...")
            self._split_single()
            
        for i in tqdm(range(len(self.img_fnames))):
            # set image and mask name
            img_fname = self.img_fnames[i]
            img_path = os.path.join(self.img_dir, img_fname)
            if self.msk_dir: msk_path = os.path.join(self.msk_dir, img_fname)
            # read image and mask
            img,spacing = adaptive_imread(img_path)
            if self.msk_dir: msk,_ = adaptive_imread(msk_path)

            # extend dim
            img = np.expand_dims(img, 0)

            # one hot encoding for the mask if needed
            if self.msk_dir and len(msk.shape)!=4: 
                msk = one_hot_fast(msk, self.num_classes)
                if self.remove_bg:
                    msk = msk[1:]
            elif self.msk_dir and len(msk.shape)==4:
                # normalize each channel
                msk = (msk > 0).astype(np.uint8)

            assert len(img.shape)==len(msk.shape)==4
            # TODO: remove torchio dependency! 
            # resample the image if needed
            # if len(self.median_spacing)>0:
            #     resample = (self.median_spacing/spacing)[::-1] # transpose the dimension
            #     if resample.sum() > 0.1: # otherwise no need of spacing 
            #         if self.msk_dir: 
            #             sub = tio.Subject(img=tio.ScalarImage(tensor=img), msk=tio.LabelMap(tensor=msk))
            #             sub = tio.Resample(resample)(sub)
            #             img, msk = sub.img.numpy(), sub.msk.numpy()
            #         else:
            #             img = tio.Resample(resample)(img)

            # clip img
            if len(self.clipping_bounds)>0:
                img = np.clip(img, self.clipping_bounds[0], self.clipping_bounds[1])

            # normalize the image and msk
            # z-score normalization for the image
            if len(self.intensity_moments)>0:
                img = (img-self.intensity_moments[0])/self.intensity_moments[1]
            else:
                img = (img-img.mean())/img.std()
            
            # enhance contrast
            # img = exposure.equalize_hist(img)

            # range image in [-1, 1]
            # img = (img - img.min())/(img.max()-img.min()) * 2 - 1

            # set image type
            img = img.astype(np.float32)
            if self.msk_dir: msk = msk.astype(np.byte)

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
            if self.msk_outdir: 
                # imsave(msk_out_path, msk)
                # save image as tif
                if self.use_tif:
                    msk_out_path = os.path.join(self.msk_outdir, img_fname+'.tif')
                    tifffile.imwrite(msk_out_path, msk, compression=('zlib', 1))
                    # imsave(msk_out_path, msk)
                    # tifffile.imwrite(msk_out_path, msk) # no compression --> increased training spee!
                # save image as npy
                else:
                    msk_out_path = os.path.join(self.msk_outdir, img_fname+'.npy')
                    np.save(msk_out_path, msk)
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
    parser.add_argument("--config_folder", type=str, default='configs/',
        help="(default=\'configs/\') Configuration folder to save the auto-configuration.")
    parser.add_argument("--use_tif", default=False,  action='store_true', dest='use_tif',
        help="(default=False) Whether to use tif format to save the preprocessed images instead of npy format. Tif files are easily readable with viewers such as Napari and takes fewer disk space but are slower to load and may slow down the training process.") 
    parser.add_argument("--remove_bg", default=False,  action='store_true', dest='remove_bg',
        help="(default=False) Remove the background in masks. Remove the bg to use with sigmoid activation maps (not softmax).") 
    parser.add_argument("--no_auto_config", default=False,  action='store_true', dest='no_auto_config',
        help="(default=False) Stop showing the information to copy and paste inside the configuration file (patch_size, batch_size and num_pools).") 
    args = parser.parse_args()

    p=Preprocessing(
        img_dir=args.img_dir,
        msk_dir=args.msk_dir,
        img_outdir=args.img_outdir,
        msk_outdir=args.msk_outdir,
        num_classes=args.num_classes+1,
        remove_bg=args.remove_bg,
        use_tif=args.use_tif,
        # clipping_bounds=[-928.0,296.0]
    )

    p.run()

    if not args.no_auto_config:
        print("Start auto-configuration")
        from biom3d import auto_config

        batch, aug_patch, patch, pool = auto_config.auto_config(img_dir=p.img_dir)

        config_path = auto_config.save_auto_config(
            config_dir=args.config_dir,
            img_dir=p.img_outdir,
            msk_dir=p.msk_outdir,
            num_classes=args.num_classes,
            batch_size=batch,
            aug_patch_size=aug_patch,
            patch_size=patch,
            num_pools=pool
        )

        print("Auto-config done! Configuration saved in: ", config_path)

        # median = auto_config.compute_median(path=p.img_dir)
        # patch, pool, batch = auto_config.find_patch_pool_batch(dims=median, max_dims=(128,128,128))
        # auto_config.display_info(patch, pool, batch)

#---------------------------------------------------------------------------

