#---------------------------------------------------------------------------
# Auto-configuration 
# This script can be used to compute and display:
# - the batch size
# - the patch size
# - the augmentation patch size
# - the number of poolings in the 3D U-Net
#---------------------------------------------------------------------------

import shutil
import fileinput
from datetime import datetime
from skimage.io import imread
import SimpleITK as sitk
import os
import numpy as np
import argparse

from biom3d import config_default

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
    if extension == ".tif":
        return imread(img_path), []
    elif extension == ".npy":
        return np.load(img_path), []
    else:
        return sitk_imread(img_path)

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
    """Given the median image size, compute the patch size, the number of pooling and the batch size.
    Take the median size of an image dataset as input, 
    determine the patch size and the number of pool with "single_patch_pool" function for each dimension and 
    assert that the final dimension size is smaller than max_dims.prod().

    Parameters
    ----------
    dims: tuple of int or list of int
        A median size of an image dataset.
    max_dims: tuple, default=(128,128,128)
        Maximum patch size. The product of `max_dims` is used to determine the maximum patch size
    max_pool: int, default=5
        Maximum pooling size.
    epsilon: float, default=1e-3
        Used to have a positive value in the dimension computation.

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
    """Print in terminal the patch size, the number of pooling and the batch size.
    """
    print("*"*20,"YOU CAN COPY AND PASTE THE FOLLOWING LINES INSIDE THE CONFIG FILE", "*"*20)
    print("BATCH_SIZE =", batch)
    print("PATCH_SIZE =", list(patch))
    aug_patch = np.array(patch)+2**(np.array(pool)+1)
    print("AUG_PATCH_SIZE =",list(aug_patch))
    print("NUM_POOLS =", list(pool))

def auto_config(img_dir, max_dims=(128,128,128)):
    """Given an image folder, return the batch size, the patch size and the number of pooling.

    Parameters
    ----------
    img_dir: str
        Image folder path.
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
    median = compute_median(path=img_dir)
    patch, pool, batch = find_patch_pool_batch(dims=median, max_dims=max_dims) 
    aug_patch = np.array(patch)+2**(np.array(pool)+1)
    return batch, aug_patch, patch, pool

# ----------------------------------------------------------------------------
# Save the auto-config values in a config file.

def replace_line_single(line, key, value):
    """Given a line, replace the value if the key is in the line. This function follows the following format:
    \'key = value\'. The line must follow this format and the output will respect this format. 
    
    Parameters
    ----------
    line : str
        The input line that follows the format: \'key = value\'.
    key : str
        The key to look for in the line.
    value : str
        The new value that will replace the previous one.
    
    Returns
    -------
    line : str
        The modified line.
    
    Examples
    --------
    >>> line = "IMG_DIR = None"
    >>> key = "IMG_DIR"
    >>> value = "path/img"
    >>> replace_line_single(line, key, value)
    IMG_DIR = 'path/img'
    """
    if key==line[:len(key)]:
        assert line[len(key):len(key)+3]==" = ", "[Error] Invalid line. A valid line must contains \' = \'. Line:"+line
        line = line[:len(key)]
        
        # if value is string then we add brackets
        line += " = "
        if type(value)==str: 
            line += "\'" + value + "\'"
        elif type(value)==np.ndarray:
            line += str(value.tolist())
        else:
            line += str(value)
    return line

def replace_line_multiple(line, dic):
    """Similar to replace_line_single but with a dictionary of keys and values.
    """
    for key, value in dic.items():
        line = replace_line_single(line, key, value)
    return line

def save_auto_config(
    config_dir,
    img_dir,
    msk_dir,
    num_classes,
    batch_size,
    aug_patch_size,
    patch_size,
    num_pools):

    # copy default config file
    config_path = shutil.copy(config_default.__file__, config_dir) 

    # rename it with date included
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_config_name = os.path.join(config_dir, current_time+"-"+os.path.basename(config_path))
    os.rename(config_path, new_config_name)

    dic = {
        'IMG_DIR':img_dir,
        'MSK_DIR':msk_dir,
        'NUM_CLASSES':num_classes,
        'BATCH_SIZE':batch_size,
        'AUG_PATCH_SIZE':aug_patch_size,
        'PATCH_SIZE':patch_size,
        'NUM_POOLS':num_pools,
    }

    # edit the new config file with the auto-config values
    with fileinput.input(files=(new_config_name), inplace=True) as f:
        for line in f:
            # edit the line
            line = replace_line_multiple(line, dic)
            # write back in the input file
            print(line, end='') 
    return new_config_name

# ----------------------------------------------------------------------------
# Main

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
    args = parser.parse_args()


    median = compute_median(path=args.img_dir, return_spacing=args.spacing)
    
    if args.spacing: 
        median_spacing = median[1]
        median = median[0]
    patch, pool, batch = find_patch_pool_batch(dims=median, max_dims=(args.max_dim, args.max_dim, args.max_dim))

    display_info(patch, pool, batch)
    
    if args.spacing:print("MEDIAN_SPACING =",list(median_spacing))
    if args.median:print("MEDIAN =", list(median))

# ----------------------------------------------------------------------------