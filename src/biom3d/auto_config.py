#---------------------------------------------------------------------------
# Auto-configuration 
# This script can be used to compute and display:
# - the batch size
# - the patch size
# - the augmentation patch size
# - the number of poolings in the 3D U-Net
#---------------------------------------------------------------------------

import numpy as np
import itertools
import argparse

from biom3d.utils import adaptive_imread, abs_listdir

# ----------------------------------------------------------------------------
# Median computation

def compute_median(path, return_spacing=False):
    """Compute the median shape of a folder of images. If `return_spacing` is True, 
    then also return the median spacing.

    Parameters
    ----------
    path: str
        Folder path.
    return_spacing: bool
        Whether to return the mean image spacing. Works only for Nifti format.

    Returns
    -------
    numpy.ndarray
        Median shape of the images in the folder. 
    """
    path_imgs = abs_listdir(path)
    sizes = []
    if return_spacing: spacings = []
    for i in range(len(path_imgs)):

        img,metadata = adaptive_imread(path_imgs[i])
        spacing = None if not 'spacing' in metadata.keys() else metadata['spacing']

        assert len(img.shape)>0, "[Error] Wrong image image."
        sizes += [list(img.shape)]
        if return_spacing and (spacing is not None): spacings+=[spacing]
    assert len(sizes)>0, "[Error] List of sizes for median computation is empty. It is probably due to an empty image folder."
    sizes = np.array(sizes)
    median = np.median(sizes, axis=0).astype(int)
    
    if return_spacing: 
        spacings = np.array(spacings)
        median_spacing = np.median(spacings, axis=0)
        return median, median_spacing

    return median 

def data_fingerprint(img_dir, msk_dir=None, num_samples=10000):
    """Compute the data fingerprint. 

    Parameters 
    ----------
    img_dir : str
        Path to the directory of images.
    msk_dir : str, default=None
        (Optional) Path to the corresponding directory of masks. If provided the function will compute the mean, the standard deviation, the 0.5% percentile and the 99.5% percentile of the intensity values of the images located inside the masks. If not provide, the function returns zeros for each of these values.
    num_samples : int, default=10000
        We compute the intensity characteristic on only a sample of the candidate voxels.
    
    Returns
    -------
    median_size : numpy.ndarray
        Median size of the images in the image folder.
    median_spacing : numpy.ndarray
        Median spacing of the images in the image folder.
    mean : float
        Mean of the intensities.
    std : float
        Standard deviation of the intensities.
    perc_005 : float
        0.5% percentile of the intensities.
    perc_995 : float
        99.5% percentile of the intensities.
    """ 
    path_imgs = abs_listdir(img_dir)
    if msk_dir is not None:
        path_msks = abs_listdir(msk_dir)
    
    sizes = []
    spacings = []
    samples = []
        
    for i in range(len(path_imgs)):
        img,metadata = adaptive_imread(path_imgs[i])
        spacing = None if not 'spacing' in metadata.keys() else metadata['spacing']

        # store the size
        sizes += [list(img.shape)]

        # store the spacing
        if spacing is not None or spacing!=[]: 
            spacings+=[spacing]

        if msk_dir is not None:
            # read msk
            msk,_ = adaptive_imread(path_msks[i])
            
            # extract only useful voxels
            img = img[msk > 0]
    
            # to get a global sample of all the images, 
            # we use random sampling on the image voxels inside the mask
            samples.append(np.random.choice(img, num_samples, replace=True) if len(img)>0 else [])

    # median computation
    median_size = np.median(np.array(sizes), axis=0).astype(int)
    median_spacing = np.median(np.array(spacings), axis=0)
    
    # compute fingerprints
    mean = float(np.mean(samples)) if samples!=[] else 0
    std = float(np.std(samples)) if samples!=[] else 0
    perc_005 = float(np.percentile(samples, 0.5)) if samples!=[] else 0
    perc_995 = float(np.percentile(samples, 99.5)) if samples!=[] else 0

    return median_size, median_spacing, mean, std, perc_005, perc_995 

# ----------------------------------------------------------------------------
# Patch pool batch computation

def find_patch_pool_batch(dims, max_dims=(128,128,128), max_pool=5, epsilon=1e-3):
    """Given the median image size, compute the patch size, the number of pooling and the batch size.
    The generated patch size is repecting the input dimension proportions.
    The product of the patch dimensions is lower than the product of the `max_dims` dimensions.

    Parameters
    ----------
    dims: tuple of int or list of int
        A median size of an image dataset.
    max_dims: tuple, default=(128,128,128)
        Maximum patch size. The product of `max_dims` is used to determine the maximum patch size
    max_pool: int, default=5
        Maximum pooling size.
    epsilon: float, default=1e-3
        Used to reduce the input dimensions if they are too big. Input dimensions will be divided by (1+epsilon) an sufficient number times so they resulting dimensions respect the max_dims limit.

    Returns
    -------
    patch: numpy.ndarray
        Patch size.
    pool: numpy.ndarray
        Number of pooling.
    batch: numpy.ndarray
        Batch size.
    """
    # transform tuples into arrays
    assert len(dims)==3 or len(dims)==4, print("Dims has not the correct number of dimensions: len(dims)=", len(dims))
    if len(dims)==4:
        dims=dims[1:]
    dims = np.array(dims)
    ori_dim = dims.copy()
    max_dims = np.array(max_dims)
    
    assert np.all(dims>0), "[Error] One dimension is non-positve {}".format(dims)
    
    # minimum feature maps size
    # is determined using min of max_dims tuple divided by 2**max_pool
    min_fmaps = min(max_dims)//(2**max_pool)
    assert min_fmaps >= 1, "[Error] The minimum of max_dims {} is too small regarding max_pool {}. Increase max_dims or reduce max_pool.".format(min(max_dims), max_pool)
    
    # if the input dimensions are too big, they will be reduced.
    # the reduced dimensions will respect input dimension proportions.
    # they will be divided by a 1+epsilon until reaching a sufficiently small resolution
    if dims.prod() > max_dims.prod():
        # grouped lower bound to the reduction level,
        # lower values will cause the final dimension to be bigger than the max_dim
        # the goal of the reduction is that the product of the input dimensions is
        # smaller than the product of max_dims
        # here, logarithms are used, dealing with sums instead of multiplications
        lb = (np.log(dims).sum()-np.log(max_dims).sum())/np.log(1+epsilon)
        
        # reduction order is set, by default, to be equal for every dimension
        reduction = lb/len(dims)
        
        # there are individual upper bounds to the reduction order for each dimension, 
        # higher values will cause final dimension to be smaller than one
        ub = np.log(dims)/np.log(1+epsilon) 
        
        # values higher than the upper bound are thus clipped
        too_high_values = (reduction > ub)
        
        # other values, respecting the lower bound, are decreased
        while np.any(too_high_values):
            # new decreased reduction value for value respecting the lower bound
            reduction_small = (lb-(ub*too_high_values).sum())/(~too_high_values).sum()
            
            # clipping
            reduction = np.where(too_high_values, ub, reduction_small)
            
            # update too high values
            too_high_values = (reduction > ub)
            
        # reduce the dimensions
        dims = dims / (1+epsilon)**reduction
    
    dims = np.floor(dims).astype(int)
    dims = np.maximum(dims, 1)
    
    # find in which interval dims/min_fmaps are:
    # is it: [1,2], [2,4], [4,8], [8,16], [16,32], or [32, ...]?
    pool = np.floor(np.log2(dims/min_fmaps)).astype(int)
    pool = np.clip(pool, 0, max_pool)
    
    # patch size is determined by the closest multiple of 2**pool from dims
    pool_pow = (2**pool).astype(int)
    # patch = (dims//pool_pow + np.round((dims%pool_pow)/pool_pow))*pool_pow
    patch = (dims//pool_pow)*pool_pow
    patch = patch.astype(int)
    
    # [gpu memory optimization]
    # dimensions of the patch are eventually increased one last time to be 
    # below but as close as possible from max_dims
    unique_patch = sorted(np.unique(patch), reverse=True)
    
    # generate all possible combination of unique values of elements of patch
    # to simplify the explanation: 
    # first, all values in patch are attempted to be increase simultaneously, 
    # then only the biggest ones, and finally, only the smallest ones
    indices_unique_patch = list(itertools.product([True, False], repeat=len(unique_patch)))[:-1]
    
    for i in range(len(indices_unique_patch)): # from biggest to smallest
        unique_patch = sorted(np.unique(patch), reverse=True)
        crt_patch = np.isin(patch, unique_patch*np.array(indices_unique_patch[i]))
        
        # first condition: check if adding something will not exceed max_dims
        # second condition: check if patch size is not already bigger than input image
        while (patch + pool_pow*crt_patch).prod() <= max_dims.prod() and \
              np.any((patch < ori_dim)*crt_patch):
            patch = patch + pool_pow*crt_patch

    # update pool size 
    pool = np.floor(np.log2(patch/min_fmaps)).astype(int)
    pool = np.clip(pool, 0, max_pool)
    
    # batch_size is set to 2 and increased if possible
    batch = 2*np.floor(max_dims.prod()/patch.prod()).astype(int)
    batch = np.maximum(batch, 2)
    
    return patch, pool, batch

def get_aug_patch(patch_size):
    """Return augmentation patch size.
    The current solution is to use the diagonal of the rectagular cuboid of the patch size, for isotripic images and, for anisotropic images, the diagonal of the rectangle spaned by the non-anisotropic dimensions. 

    Parameters
    ----------
    patch_size : tuple, list or numpy.ndarray
        Patch size.

    Returns
    -------
    aug_patch : numpy.ndarray
        Augmentation patch size.
    """
    ps = np.array(patch_size)
    dummy_2d = ps/ps.min()

    if np.any(dummy_2d>3): # then use dummy_2d
        axis = np.argmin(dummy_2d)
        # aug_patch = np.round(1.17*ps).astype(int)
        diag = np.sqrt(np.array(list(s**2 if i!=axis else 0 for i,s in enumerate(ps))).sum())
        diag = np.round(diag).astype(int)
        aug_patch = list(diag for _ in range(len(patch_size)))
        aug_patch[axis] = patch_size[axis]
    else:
        # aug_patch = np.round(1.37*ps).astype(int)
        diag = np.round(np.sqrt((ps**2).sum())).astype(int)
        aug_patch = list(diag for _ in range(len(patch_size)))
    return aug_patch
        

# ----------------------------------------------------------------------------
# Display 

def display_info(patch, pool, batch):
    """Print in terminal the patch size, the number of pooling and the batch size.
    """
    print("*"*20,"YOU CAN COPY AND PASTE THE FOLLOWING LINES INSIDE THE CONFIG FILE", "*"*20)
    print("BATCH_SIZE =", batch)
    print("PATCH_SIZE =", list(patch))
    aug_patch = get_aug_patch(patch)
    print("AUG_PATCH_SIZE =",list(aug_patch))
    print("NUM_POOLS =", list(pool))

def auto_config(img_dir=None, median=None, max_dims=(128,128,128), max_batch=16, min_batch=2):
    """Given an image folder, return the batch size, the patch size and the number of pooling.
    Provide either an image directory or a median shape. If a median shape is provided it will not be recomputed and the auto-configuration will be much faster.

    Parameters
    ----------
    img_dir : str
        Image folder path.
    median : list or tuple
        Median size of the images in the image directory.
    max_dims: tuple, default=(128,128,128)
        Maximum patch size. The product of `max_dims` is used to determine the maximum patch size

    Returns
    -------
    batch: numpy.ndarray
        Batch size.
    aug_patch: numpy.ndarray
        Augmentation patch size.
    patch: numpy.ndarray
        Patch size.
    pool: numpy.ndarray
        Number of pooling.
    """
    assert not(img_dir is None and median is None), "[Error] Please provide either an image directory or a median shape."
    if median is None: median = compute_median(path=img_dir) 
    patch, pool, batch = find_patch_pool_batch(dims=median, max_dims=max_dims) 
    aug_patch = get_aug_patch(patch)
    if batch > max_batch: batch = max_batch
    if batch < min_batch: batch = min_batch
    return batch, aug_patch, patch, pool

# ----------------------------------------------------------------------------
# Main
# Note 2023/04/28, Guillaume: I think that the main is now a bit outdated... still works tho

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Auto-configuration of the hyper-parameter for training.")
    parser.add_argument("--img_dir", type=str,
        help="Path of the images directory")
    parser.add_argument("--max_dim", type=int, default=128,
        help="Maximum size of one dimension of the patch (default: 128)")  
    parser.add_argument("--spacing", default=False,  action='store_true', dest='spacing',
        help="Print median spacing if set.")
    parser.add_argument("--median", default=False,  action='store_true', dest='median',
        help="Print the median.")
    parser.add_argument("--save_config", default=False,  action='store_true', dest='save_config',
        help="(default=False) Whether to save the configuration.")
    parser.add_argument("--config_dir", type=str, default='configs/',
        help="(default=\'configs/\') Configuration folder to save the auto-configuration.")
    parser.add_argument("--base_config", type=str, default=None,
        help="(default=None) Optional. Path to an existing configuration file which will be updated with the preprocessed values.")
    args = parser.parse_args()

    median = compute_median(path=args.img_dir, return_spacing=args.spacing)
    
    if args.spacing: 
        median_spacing = median[1]
        median = median[0]

    patch, pool, batch = find_patch_pool_batch(dims=median, max_dims=(args.max_dim, args.max_dim, args.max_dim))
    aug_patch = np.array(patch)+2**(np.array(pool)+1)

    display_info(patch, pool, batch)
    
    if args.spacing:print("MEDIAN_SPACING =",list(median_spacing))
    if args.median:print("MEDIAN =", list(median))

    if args.save_config:
        try: 
            from biom3d.utils import save_python_config
            config_path = save_python_config(
                config_dir=args.config_dir,
                base_config=args.base_config,
                
                BATCH_SIZE=batch,
                AUG_PATCH_SIZE=aug_patch,
                PATCH_SIZE=patch,
                NUM_POOLS=pool,
                MEDIAN_SPACING=median_spacing,
            )
        except:
            print("[Error] Import error. Biom3d must be installed if you want to save your configuration. Another solution is to config the function function in biom3d.utils here...")
            raise ImportError

# ----------------------------------------------------------------------------