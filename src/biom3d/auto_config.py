"""
Auto-configuration.

This script can be used to compute and display:
- the batch size
- the patch size
- the augmentation patch size
- the number of poolings in the 3D U-Net
"""

from typing import Optional
import numpy as np
import itertools
import argparse

from biom3d.utils import DataHandlerFactory, save_python_config

# ----------------------------------------------------------------------------
# Median computation

def compute_median(path:str, return_spacing:bool=False)->np.ndarray | tuple[np.ndarray,np.ndarray]:
    """
    Compute the median shape of a folder of images. If `return_spacing` is True, then also return the median spacing.

    Parameters
    ----------
    path: str
        Folder path.
    return_spacing: bool
        Whether to return the mean image spacing. Works only for Nifti format.

    Raises
    ------
    AssertionError
        If no image is found at `path`, or size couldn't be retrieved.
    AssertionError
        If images has inconsistents dimensions

    Returns
    -------
    median: numpy.ndarray
        Median shape of the images in the folder. 
    spacing: numpy.ndarray
        Median spacing of the images in the folder. 
    """
    handler = DataHandlerFactory.get(
        path,
        read_only=True,
        output=None,
    )

    sizes = []
    num_dims=None
    if return_spacing: spacings = []
    for img_path,_,_ in handler:
        img,metadata = handler.load(img_path)
        spacing = None if 'spacing' not in metadata.keys() else metadata['spacing']

        assert len(img.shape)>0, f"[Error] Wrong image {img_path}."
        img_shape = img.shape 
        # Check if the number of dimension is consistent across the dataset
        if num_dims is None: num_dims = len(img_shape)
        else: assert num_dims == len(img_shape), f"[Error] Inconsistency in the number of dimensions across the dataset: {num_dims} and {len(img_shape)}."
        # Check if the image is 2D (has two dimensions)
        if len(img_shape) == 2:
            # Add a third dimension with size 1 to make it 3D
            img_shape = (1,) + img_shape
        sizes += [list(img_shape)]
        if return_spacing and (spacing is not None): spacings+=[spacing]
    handler.close()
    assert len(sizes)>0, "[Error] List of sizes for median computation is empty. It is probably due to an empty image folder."
    num_dims = [len(s) for s in sizes]
    assert min(num_dims)==max(num_dims), f"Inconsistent number of dimensions in images: {num_dims}"
    sizes = np.array(sizes)
    median = np.median(sizes, axis=0).astype(int)
    if return_spacing: 
        if len(spacings)==0: 
            print("Warning: `return_spacing` was set to True but no spacing was found in the image metadata, will return an empty `median_spacing` array.")
            return median, spacings
        spacings = np.array(spacings)
        median_spacing = np.median(spacings, axis=0)
        return median, median_spacing

    return median 

def data_fingerprint(img_path:str, 
                     msk_path:Optional[str]=None, 
                     num_samples:int=10000,
                     seed:int=42,
                     )->tuple[np.ndarray,np.ndarray,float,float,float,float]:
    """
    Compute the data fingerprint.

    The fingerprint consist of:
    - Median size of the images.
    - Median spacing of the images.
    - Mean value of the images' voxels (or pixels).
    - Standard deviation of the images' voxels (or pixels).
    - Percentil 0.5% of the images' voxels (or pixels) intensity.
    - Percentil 99.5% of the images' voxels (or pixels) intensity.
    

    Parameters 
    ----------
    img_path : str
        Path to the images collection.
    msk_path : str, optional
        (Optional) Path to the corresponding collection of masks. If provided the function will compute the mean, the standard deviation, the 0.5% percentile and the 99.5% percentile of the intensity values of the images located inside the masks. If not provide, the function returns zeros for each of these values.
    num_samples : int, default=10000
        We compute the intensity characteristic on only a sample of the candidate voxels.
    seed : int, default=42
        (Optional) Random generator seed, is used if msk_path isn't None.

    Raises
    ------
    AssertionError | ValueError
        If inconsistent number of dimension across dataset.
    
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
    handler = DataHandlerFactory.get(
        img_path,
        read_only=True,
        output=None,
        msk_path = msk_path,
    )
    
    sizes = []
    spacings = []
    samples = []

        
    for img_path,msk_path,_ in handler:
        img,metadata = handler.load(img_path)

        num_dims = None

        spacing = None if 'spacing' not in metadata.keys() else metadata['spacing']

        # store the size
        img_shape = img.shape 
        # Check if the number of dimension is consistent across the dataset
        if num_dims is None: num_dims = len(img_shape)
        else: assert num_dims == len(img_shape), f"[Error] Inconsistency in the number of dimensions across the dataset: {num_dims} and {len(img_shape)}."
        # Check if the image is 2D (has two dimensions)
        if len(img_shape) == 2:
            # Add a third dimension with size 1 to make it 3D
            img_shape = (1,) + img_shape
        sizes += [list(img_shape)]

        # store the spacing
        if spacing is not None and spacing!=[]: 
            spacings+=[spacing]

        if msk_path is not None:
            # read msk
            msk,_ = handler.load(msk_path)
            
            # extract only useful voxels
            img = img[msk > 0]

            # to get a global sample of all the images, 
            # we use random sampling on the image voxels inside the mask
            rng = np.random.default_rng(seed)

            if len(img) > 0:
                samples.append(rng.choice(img, num_samples, replace=True) if len(img)>0 else [])
    handler.close()

    # median computation
    try:
        median_size = np.median(np.array(sizes), axis=0).astype(int)
    except ValueError:
        raise ValueError( "Images don't have the same number of dimensions" ) # Already checked in the loop ?
    
    for i in range(len(spacings)):
        if spacings[i] is None : spacings[i] = []


    median_spacing = np.median(np.array(spacings), axis=0) if len(spacings) > 0 else 0

    
    if not samples: 
        return median_size, median_spacing, 0, 0, 0, 0

    # compute fingerprints
    mean = float(np.mean(samples)) 
    std = float(np.std(samples)) 
    perc_005 = float(np.percentile(samples, 0.5)) 
    perc_995 = float(np.percentile(samples, 99.5)) 

    return median_size, median_spacing, mean, std, perc_005, perc_995 

# ----------------------------------------------------------------------------
# Patch pool batch computation

def find_patch_pool_batch(dims:tuple[int]|list[int], 
                          max_dims:tuple[int]=(128,128,128), 
                          max_pool:int=5, 
                          epsilon:float=1e-3,
                          )->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Given the median image size, compute the patch size, the number of pooling and the batch size.
    
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

    Raises
    ------
    AssertionError
        If dims not in 3D or 4D.
    AssertionError
        If a dimension is negative.
    AssertionError
        If max_pool has at least 1 element that is less than 2 times bigger than an element of max_dims

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
    patch = (dims//pool_pow)*pool_pow
    patch = patch.astype(int)
    
    # [gpu memory optimization]
    # dimensions of the patch are eventually increased one last time to be 
    # below but as close as possible from max_dims
    unique_patch = sorted(np.unique(patch), reverse=True)
    
    # generate all possible combination of unique values of elements of patch
    # to simplify the explanation: 
    # first, all values in patch dimensions are attempted to be increase simultaneously, 
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

def get_aug_patch(patch_size:tuple[int]|list[int]|np.ndarray)->np.ndarray:
    """
    Return augmentation patch size.
    
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
        diag = np.sqrt(np.array([s**2 if i!=axis else 0 for i,s in enumerate(ps)]).sum())
        diag = np.round(diag).astype(int)
        aug_patch = [int(diag) for _ in range(len(patch_size))]
        aug_patch[axis] = int(patch_size[axis])
    else:
        diag = np.round(np.sqrt((ps**2).sum())).astype(int)
        aug_patch = [int(diag) for _ in range(len(patch_size))]
    return aug_patch
        

# ----------------------------------------------------------------------------
# Display 

def parameters_return(patch:np.ndarray, 
                      pool:np.ndarray, 
                      batch:np.ndarray, 
                      config_path:str,
                      median_spacing:Optional[np.ndarray]=None,
                      )->None:
    """
    Display the provided parameters.
    
    Parameters
    ----------
    patch: numpy.ndarray
        Patch size.
    pool: numpy.ndarray
        Pool size.
    batch: numpy.ndarray
        batch size.
    config_path: str
        Path to configuration file
    median_spacing: numpy.ndarray, optional
        Median spacing over the dataset.

    Returns
    -------
    None
    """
    print(batch)
    print(patch)
    print(get_aug_patch(patch))
    print(pool)
    print(median_spacing)
    print(config_path)

def display_info(patch:np.ndarray, pool:np.ndarray, batch:np.ndarray)->None:
    """
    Print in terminal the patch size, the number of pooling, augmented patch size and the batch size.

    The output follow the config file syntaxe and can copied into.

    Parameters
    ----------
    patch: numpy.ndarray
        Patch size.
    pool: numpy.ndarray
        Pool size.
    batch: numpy.ndarray
        batch size.

    Returns
    -------
    None
    """
    print("*"*20,"YOU CAN COPY AND PASTE THE FOLLOWING LINES INSIDE THE CONFIG FILE", "*"*20)
    print("BATCH_SIZE =", batch)
    print("PATCH_SIZE =", list(patch))
    aug_patch = get_aug_patch(patch)
    print("AUG_PATCH_SIZE =",list(aug_patch))  
    print("NUM_POOLS =", list(pool))

def auto_config(img_path:Optional[str]=None,
                median:Optional[list[int]|tuple[int]]=None,
                max_dims:tuple[int]=(128,128,128), 
                max_batch:int=16, 
                min_batch:int=2,
                )->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Given an image collection, return the batch size, the patch size and the number of pooling.

    Provide either an image collection path or a median shape. If a median shape is provided it will not be recomputed and the auto-configuration will be much faster.

    Parameters
    ----------
    img_path : str, optional
        Image collection path. If not provided, must give a median shape.
    median : list or tuple, optional
        Median size of the images in the image collection. If not provided, must give an image path.
    max_dims: tuple, default=(128,128,128)
        Maximum patch size. The product of `max_dims` is used to determine the maximum patch size
    max_batch: int, default=16
        Maximum batch size. Clamp computed batch size if needed.
    min_batch: int, default=16
        Minimum batch size. Clamp computed batch size if needed.

    Raises
    ------
    AssertionError
        If img_path and median are both None.

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
    assert not(img_path is None and median is None), "[Error] Please provide either an image collection path or a median shape."
    if median is None: median = compute_median(path=img_path) 
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
    parser.add_argument("--img_path", type=str,
        help="Path of the images collection")
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
    parser.add_argument("--remote", default=False, dest='remote',
        help="Use this argument when using remote autoconfig only.")
    args = parser.parse_args()

    median = compute_median(path=args.img_path, return_spacing=args.spacing)
    
    if args.spacing: 
        median_spacing = median[1]
        median = median[0]
    else:
        median_spacing = None

    patch, pool, batch = find_patch_pool_batch(dims=median, max_dims=(args.max_dim, args.max_dim, args.max_dim))
    aug_patch = np.array(patch)+2**(np.array(pool)+1)

    if args.remote or args.save_config:
        config_path = save_python_config(
            config_dir=args.config_dir,
            base_config=args.base_config,
            
            BATCH_SIZE=batch,
            AUG_PATCH_SIZE=aug_patch,
            PATCH_SIZE=patch,
            NUM_POOLS=pool,
            MEDIAN_SPACING=median_spacing,
        )
    if args.remote:
        parameters_return(patch, pool, batch, config_path, median_spacing)  
    else:
        display_info(patch, pool, batch)

    if args.spacing:print("MEDIAN_SPACING =",list(median_spacing))
    if args.median:print("MEDIAN =", list(median))

#----------------------------------------------------------------------------
