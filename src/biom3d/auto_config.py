#---------------------------------------------------------------------------
# Auto-configuration 
# This script can be used to compute and display:
# - the batch size
# - the patch size
# - the augmentation patch size
# - the number of poolings in the 3D U-Net
#---------------------------------------------------------------------------

from skimage.io import imread
import SimpleITK as sitk
import os
import numpy as np
import argparse

# ----------------------------------------------------------------------------
# path utils

def abs_path(root, listdir_):
    """Add the root path to each element in a list of path.

    Parameters
    ----------
    root: str
        Path to root folder.
    listdir_: list of str
        List of file names.
    
    Returns
    -------
    list of str
        List of the absolute paths.
    """
    listdir = listdir_.copy()
    for i in range(len(listdir)):
        listdir[i] = root + '/' + listdir[i]
    return listdir

def abs_listdir(path):
    """Return a list of absolute paths from a folder. 
    Equivalent to os.listdir but with absolute path.

    Parameters
    ----------
    path: str
        Path to the folder.

    Returns
    -------
    list of str
        List of the absolute paths.
    """
    return abs_path(path, os.listdir(path))

# ----------------------------------------------------------------------------
# Imread utils

def sitk_imread(img_path):
    """SimpleITK image reader. Used for nii.gz files.

    Parameters
    ----------
    img_path: str
        Image path.

    Returns
    -------
    numpy.ndarray
        Images.
    tuple 
        Image spacing. 
    """
    img = sitk.ReadImage(img_path)
    img_np = sitk.GetArrayFromImage(img)
    return img_np, np.array(img.GetSpacing())

def adaptive_imread(img_path):
    """Use skimage imread or sitk imread depending on the file extension:
    .tif --> skimage.io.imread
    .nii.gz --> SimpleITK.imread

    Parameters
    ----------
    img_path: str
        Image path.

    Returns
    -------
    numpy.ndarray
        Images.
    tuple 
        Image spacing. Can be None (for non-nifti files).
    """
    extension = img_path[img_path.rfind('.'):]
    if extension == ".gz":
        return sitk_imread(img_path)
    else:
        return imread(img_path), None

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
        img,spacing = adaptive_imread(path_imgs[i])
        sizes += [list(img.shape)]
        if return_spacing and (spacing is not None): spacings+=[spacing]
    sizes = np.array(sizes)
    median = np.median(sizes, axis=0).astype(int)
    
    if return_spacing: 
        spacings = np.array(spacings)
        median_spacing = np.median(spacings, axis=0)
        return median, median_spacing

    return median 

# ----------------------------------------------------------------------------
# Patch pool batch computation

def single_patch_pool(dim, size_limit=7):
    """Return the patch size with the heuristic proposed by nnUNet.
    Divide by two the `dim` number until obtaining a number lower than 7.
    Then np.ceil this number. Then multiply this number multiple times by two to obtain the patch size.

    Parameters
    ----------
    dim: int
        A single dimension.

    Returns
    -------
    numpy.ndarray
        Median shape of the images in the folder.
    """
    pool = 0
    while dim > size_limit:
        dim /= 2
        pool += 1
    patch = np.round(dim)
    patch = patch.astype(int)*(2**pool)
    return patch, pool

def find_patch_pool_batch(dims, max_dims=(128,128,128), max_pool=5, epsilon=1e-3):
    """
    take the median size as input, determine the patch size and the number of pool
    with "single_patch_pool" function for each dimension and 
    assert that the final dimension size is smaller than max_dims.prod().
    """
    # transform tuples into arrays
    assert len(dims)==3 or len(dims)==4, print("Dims has not the correct number of dimensions: len(dims)=", len(dims))
    if len(dims)==4:
        dims=dims[1:]
    dims = np.array(dims)
    max_dims = np.array(max_dims)
    
    # divides by a 1+epsilon until reaching a sufficiently small resolution
    while dims.prod() > max_dims.prod():
        dims = dims / (1+epsilon)
    dims = dims.astype(int)
    
    # compute patch and pool for all dims
    patch_pool = np.array([single_patch_pool(m) for m in dims])
    patch = patch_pool[:,0]
    pool = patch_pool[:,1]
    
    # assert the final size is smaller than max_dims
    while patch.prod()>max_dims.prod():
        patch = patch - np.array([32,32,32])*(patch>max_dims) # removing multiples of 32
    pool = np.where(pool > max_pool, max_pool, pool)
    
    # batch_size
    batch = 2
    while batch*patch.prod() <= 2*max_dims.prod():
        batch += 1
    if batch*patch.prod() > 2*max_dims.prod():
        batch -= 1
    return patch, pool, batch

# ----------------------------------------------------------------------------
# Display 

def display_info(patch, pool, batch):
    print("*"*20,"YOU CAN COPY AND PASTE THE FOLLOWING LINES INSIDE THE CONFIG FILE", "*"*20)
    print("BATCH_SIZE =", batch)
    print("PATCH_SIZE =", list(patch))
    aug_patch = np.array(patch)+2**(np.array(pool)+1)
    print("AUG_PATCH_SIZE =",list(aug_patch))
    print("NUM_POOLS =", list(pool))

def auto_config(img_dir, max_dims=(128,128,128)):
    median = compute_median(path=img_dir)
    patch, pool, batch = find_patch_pool_batch(dims=median, max_dims=max_dims) 
    aug_patch = np.array(patch)+2**(np.array(pool)+1)
    return batch, aug_patch, patch, pool

def minimal_display(img_dir, max_dims=(128,128,128)):
    out = auto_config(img_dir, max_dims=max_dims)
    for element in out:
        print(element)

# ----------------------------------------------------------------------------
# Main

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Dataset preprocessing for training purpose.")
    parser.add_argument("--img_dir", type=str,
        help="Path of the images directory")
    parser.add_argument("--max_dim", type=int, default=128,
        help="Maximum size of one dimension of the patch (default: 128)")  
    # parser.add_argument("--min_dis", default=False,  action='store_true', dest='min_dis',
    #     help="Minimal display. Display only the raw batch, aug_patch, patch and pool")
    parser.add_argument("--spacing", default=False,  action='store_true', dest='spacing',
        help="Print median spacing if set.")
    parser.add_argument("--median", default=False,  action='store_true', dest='median',
        help="Print the median.")
    args = parser.parse_args()


    # if args.min_dis:
    #     minimal_display(img_dir=args.img_dir, max_dims=(args.max_dim, args.max_dim, args.max_dim))
    # else: 
    median = compute_median(path=args.img_dir, return_spacing=args.spacing)
    
    if args.spacing: 
        median_spacing = median[1]
        median = median[0]
    patch, pool, batch = find_patch_pool_batch(dims=median, max_dims=(args.max_dim, args.max_dim, args.max_dim))

    display_info(patch, pool, batch)
    
    if args.spacing:print("MEDIAN_SPACING =",list(median_spacing))
    if args.median:print("MEDIAN =", list(median))

    # median=compute_median(path='/home/gumougeot/all/codes/python/3dnucleus/data/pancreas/tif_imagesTr_small')
    # median=compute_median(path='/home/gumougeot/all/codes/python/3dnucleus/data/lung/tif_imagesTr')
    # median=compute_median(path='/home/gumougeot/all/codes/python/3dnucleus/data/remi/tif_img')
    # print("patch, pool, batch:", find_patch_pool_batch(dims=median, max_dims=(160,160,160)))
    # print("patch, pool, batch:", find_patch_pool_batch(dims=median, max_dims=(192,192,192)))
    # print("patch, pool, batch:", find_patch_pool_batch(dims=median, max_dims=(128,128,128)))

