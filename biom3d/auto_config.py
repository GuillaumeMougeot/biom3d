from skimage.io import imread
import os
import numpy as np
import argparse

def abs_path(root, listdir_):
    listdir = listdir_.copy()
    for i in range(len(listdir)):
        listdir[i] = root + '/' + listdir[i]
    return listdir

def abs_listdir(path):
    return abs_path(path, os.listdir(path))

def compute_median(path):
    path_imgs = abs_listdir(path)
    sizes = []
    for i in range(len(path_imgs)):
        sizes += [list(imread(path_imgs[i]).shape)]
    sizes = np.array(sizes)
    median = np.median(sizes, axis=0).astype(int)
    # print("median:",median)
    return median 

def single_patch_pool(dim, size_limit=7):
    """
    divide by two the dim number until obtaining a number lower than 7
    then np.ceil this number
    then multiply multiple times by two this number to obtain the patch size
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
    assert len(dims)==3 or len(dims)==4
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

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Dataset preprocessing for training purpose.")
    parser.add_argument("--img_dir", type=str,
        help="Path of the images directory")
    parser.add_argument("--max_dim", type=int, default=128,
        help="Maximum size of one dimension of the patch (default: 128)")  
    parser.add_argument("--min_dis", default=False,  action='store_true', dest='min_dis',
        help="Minimal display. Display only the raw batch, aug_patch, patch and pool")
    args = parser.parse_args()


    if args.min_dis:
        minimal_display(img_dir=args.img_dir, max_dims=(args.max_dim, args.max_dim, args.max_dim))
    else:
        median = compute_median(path=args.img_dir)
        patch, pool, batch = find_patch_pool_batch(dims=median, max_dims=(args.max_dim, args.max_dim, args.max_dim))
        display_info(patch, pool, batch)

    # median=compute_median(path='/home/gumougeot/all/codes/python/3dnucleus/data/pancreas/tif_imagesTr_small')
    # median=compute_median(path='/home/gumougeot/all/codes/python/3dnucleus/data/lung/tif_imagesTr')
    # median=compute_median(path='/home/gumougeot/all/codes/python/3dnucleus/data/remi/tif_img')
    # print("patch, pool, batch:", find_patch_pool_batch(dims=median, max_dims=(160,160,160)))
    # print("patch, pool, batch:", find_patch_pool_batch(dims=median, max_dims=(192,192,192)))
    # print("patch, pool, batch:", find_patch_pool_batch(dims=median, max_dims=(128,128,128)))

