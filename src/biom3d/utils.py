# ----------------------------------------------------------------------------
# a set of utility functions 
# content:
#  - base class for config file 
#  - read folds from a csv file
#  - create logs and models directories
#  - tif metadata reader and writer
# ----------------------------------------------------------------------------

import numpy as np
from datetime import datetime
from time import time 
import os 
import importlib.util
import sys
import shutil
import fileinput
import tifffile as tiff
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # bug fix: change matplotlib backend 
import yaml # pip install pyyaml
from skimage import io
from skimage.transform import resize
from skimage import measure
import SimpleITK as sitk
import torchio as tio
from numba import njit

try: import napari
except: pass

# ----------------------------------------------------------------------------
# read folds from a csv file

def get_train_test_df(df, verbose=True):
    """
    Return the train set and the test set
    """
    train_set = np.array(df[df['hold_out']==0].iloc[:,0])
    test_set = np.array(df[df['hold_out']==1].iloc[:,0])
    return train_set, test_set

def get_folds_df(df, verbose=True):
    """
    Return of folds in a list of list
    """
    folds = []
    if df.empty:
        print("[Warning] one of the data DataFrame is empty!")
        return []
    nbof_folds = df['fold'].max()+1
    if verbose:
        print("Number of folds in df: {}".format(nbof_folds))
    
    size_folds = []
    for i in range(nbof_folds):
        folds += [list(df[df['fold']==i].iloc[:,0])]
        size_folds += [len(folds[-1])]
    if verbose:
        print("Size of folds: {}".format(size_folds))
    return folds

def get_folds_train_test_df(df, verbose=True, merge_test=True):
    """
    Return folds from the train set and the test set in a list of list.
    Output: (train_folds, test_folds)
    If merge_test==True then the test folds are merged in one list.
    """
    if verbose:
        print("Training set:")
    train_folds = get_folds_df(df[df['hold_out']==0], verbose)
    
    if verbose:
        print("Testing set:")
    test_folds = get_folds_df(df[df['hold_out']==1], verbose)
    
    if merge_test:
        test_folds_merged = []
        for i in range(len(test_folds)):
            test_folds_merged += test_folds[i]
        test_folds = test_folds_merged
    return train_folds, test_folds

def get_splits_train_val_test(df):
    """
    the splits contains [100%,50%,25%,10%,5%,2%,the rest] of the dataset
    return the train set as a list of list,
    the val and test set as lists
    """
    nbof_splits = df['split'].max()+1
    valset = list(df[(df['split']==-1)*(df['fold']==0)*(df['hold_out']==0)]['filename'])
    testset = list(df[(df['hold_out']==1)]['filename'])
    train_splits = []
    for i in range(nbof_splits):
        train_splits += [list(df[(df['split']==i)*(df['fold']!=0)*(df['hold_out']==0)].iloc[:,0])]
    # adds the whole dataset in the begging of the train_splits list
    train_splits = [list(df[(df['fold']!=0)*(df['hold_out']==0)].iloc[:,0])] + train_splits
    return train_splits, valset, testset

def get_splits_train_val_test_overlapping(df):
    """
    CAREFUL: works only if the splits contains [1/(2**0), 1/(2**1), ..., 1/(2**n), 1/(2**n)] of the training dataset 
    the splits contains of the dataset.
    "overlapping" indicates that every smaller set is contained into all bigger sets.
    return the train set as a list of list,
    the val and test set as lists
    """
    nbof_splits = df['split'].max()+1
    valset = list(df[(df['split']==-1)*(df['fold']==0)*(df['hold_out']==0)]['filename'])
    testset = list(df[(df['hold_out']==1)]['filename'])
    train_splits = []
    for i in range(nbof_splits):
        train_splits += [list(df[(df['split']>=i)*(df['fold']!=0)*(df['hold_out']==0)].iloc[:,0])]
        
    # adds the last set 
    train_splits += [list(df[(df['split']==(nbof_splits-1))*(df['fold']!=0)*(df['hold_out']==0)].iloc[:,0])]
    return train_splits, valset, testset

# ----------------------------------------------------------------------------
# create logs and models directories

def create_save_dirs(log_dir, desc, dir_names=['model', 'logs', 'images'], return_base_dir=False):
    """
    Creates saving folders. 

    Arguments:
        dir_names: a list of name of the desired folders.
                   e.g.: ['images','cpkt','summary']
    
    Returns:
        list_dirs: a list of path of the corresponding folders.
    """
    list_dirs = []
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = current_time + '-' + desc
    base_dir = os.path.join(log_dir, base_dir)
    for name in dir_names:
        list_dirs += [os.path.join(base_dir, name)]
        if not os.path.exists(list_dirs[-1]):
            os.makedirs(list_dirs[-1])
    if return_base_dir:
        return [base_dir] + list_dirs
    else:
        return list_dirs

# ----------------------------------------------------------------------------
# image readers and savers

def sitk_imread(img_path, return_spacing=True, return_origin=False, return_direction=False):
    """
    image reader for nii.gz files
    """
    img = sitk.ReadImage(img_path)
    img_np = sitk.GetArrayFromImage(img)
    dim = img.GetDimension()

    spacing = np.array(img.GetSpacing())
    origin = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection())
    if dim==4: # if dim==4 then turn it into 3...
        spacing = spacing[:-1]
        origin = origin[:-1]
        direction = direction.reshape(4,4)[:-1, :-1].reshape(-1)
    elif dim != 4 and dim != 3: 
        raise RuntimeError("Unexpected dimensionality: %d of file %s, cannot split" % (dim, img_path))
    return img_np, {"spacing": spacing, "origin": origin, "direction": direction}

def adaptive_imread(img_path):
    """
    use skimage imread or sitk imread depending on the file extension:
    .tif --> skimage.io.imread
    .nii.gz --> SimpleITK.imread
    """
    extension = img_path[img_path.rfind('.'):].lower()
    if extension == ".tif" or extension == ".tiff":
        try: 
            img, img_meta = tif_read_imagej(img_path)  # try loading ImageJ metadata for tif files
            return img, img_meta
        except:   
            img_meta = {}    
            try: img_meta["spacing"] = tif_get_spacing(img_path)
            except: img_meta["spacing"] = []
    
            return io.imread(img_path), img_meta 
    elif extension == ".npy":
        return np.load(img_path), {}
    else:
        return sitk_imread(img_path)

def sitk_imsave(img_path, img, metadata={}):
    """
    image saver for nii gz files
    """
    if not 'spacing' in metadata.keys():
        metadata['spacing']=(1,1,1)
    if not 'origin' in metadata.keys():
        metadata['origin']=(0,0,0)
    if not 'direction' in metadata.keys():
        metadata['direction']=(1., 0., 0., 0., 1., 0., 0., 0., 1.)
    img_out = sitk.GetImageFromArray(img)
    img_out.SetSpacing(metadata['spacing'])
    img_out.SetOrigin(metadata['origin'])
    img_out.SetDirection(metadata['direction'])
    sitk.WriteImage(img_out, img_path)

def adaptive_imsave(img_path, img, img_meta={}):
    """Adaptive image saving. Use tifffile for `.tif`, use numpy for `.npy` and use SimpleITK for other format. 

    Parameters
    ----------
        img_path : str
            Path to the output file.
        img : numpy.ndarray
            Image array.
        spacing : tuple, default=(1,1,1)
            Optional spacing of the image. Only used with the SimpleITK library.
    """
    extension = img_path[img_path.rfind('.'):].lower()
    if extension == ".tif" or extension == ".tiff":
        # if not np.all(spacing==(1.,1.,1.)):
        #     res = int(1e6) # default resolution is MICROMETERS
        #     tiff.imwrite(
        #         img_path,
        #         img,
        #         compression=('zlib', 1),

        #         # the lines below might have to be commented in certain cases, depending on the unit of your images... 
        #         resolution=((int(1/spacing[0]),res), (int(1/spacing[1]), res)), # TODO: unit is set to micrometer by default but this could be a problem... 
        #         metadata={
        #             'spacing':float(spacing[-1]*res),
        #             'unit':'MICROMETER', # TODO: unit is set to micrometer by default but this could be a problem... 
        #             'axes':'ZYX',
        #             },
        #         imagej=True,
        #         )
        # else:

        # Current solution for tif files 
        try:
            tif_write_imagej(
                img_path,
                img,
                img_meta)
        except:
            tiff.imwrite(
                img_path,
                img,
                compression=('zlib', 1))
    elif extension == ".npy":
        np.save(img_path, img)
    else:
        sitk_imsave(img_path, img, img_meta)

# ----------------------------------------------------------------------------
# tif metadata reader and writer

def tif_read_imagej(img_path, axes_order='CZYX'):
    """Read tif file metadata stored in a ImageJ format.
    adapted from: https://forum.image.sc/t/python-copy-all-metadata-from-one-multipage-tif-to-another/26597/8

    Parameters
    ----------
    img_path : str
        Path to the input image.
    axes_order : str, default='CZYX'
        Order of the axes of the output image.

    Returns
    -------
    img : numpy.ndarray
        Image.
    img_meta : dict
        Image metadata. 
    """

    with tiff.TiffFile(img_path) as tif:
        assert tif.is_imagej

        # store img_meta
        img_meta = {}

        # get image resolution from TIFF tags
        tags = tif.pages[0].tags
        x_resolution = tags['XResolution'].value
        y_resolution = tags['YResolution'].value
        resolution_unit = tags['ResolutionUnit'].value
        
        img_meta["resolution"] = (x_resolution, y_resolution, resolution_unit)

        # parse ImageJ metadata from the ImageDescription tag
        ij_description = tags['ImageDescription'].value
        ij_description_metadata = tiff.tifffile.imagej_description_metadata(ij_description)
        # remove conflicting entries from the ImageJ metadata
        ij_description_metadata = {k: v for k, v in ij_description_metadata.items()
                                   if k not in 'ImageJ images channels slices frames'}

        img_meta["description"] = ij_description_metadata
        
        # compute spacing
        xres = (x_resolution[1]/x_resolution[0])
        yres = (y_resolution[1]/y_resolution[0])
        zres = float(ij_description_metadata["spacing"])
        
        img_meta["spacing"] = (xres, yres, zres)

        # read the whole image stack and get the axes order
        series = tif.series[0]
        img = series.asarray()

        img = tiff.tifffile.transpose_axes(img, series.axes, axes_order)
        
        img_meta["axes"] = axes_order
    
    return img, img_meta

def tif_write_imagej(img_path, img, img_meta):
    """Write tif file using metadata in ImageJ format.
    adapted from: https://forum.image.sc/t/python-copy-all-metadata-from-one-multipage-tif-to-another/26597/8
    """
    # saving ImageJ hyperstack requires a 6 dimensional array in axes order TZCYXS
    img = tiff.tifffile.transpose_axes(img, img_meta["axes"], 'TZCYXS')

    # write image and metadata to an ImageJ hyperstack compatible file
    tiff.imwrite(img_path, img,
            resolution=img_meta["resolution"],
            imagej=True, 
            metadata=img_meta["description"],
            compression=('zlib', 1)
            )

def tif_read_meta(tif_path, display=False):
    """
    read the metadata of a tif file and stores them in a python dict.
    if there is a 'ImageDescription' tag, it transforms it as a dictionary
    """
    meta = {}
    with tiff.TiffFile(tif_path) as tif:
        for page in tif.pages:
            for tag in page.tags:
                tag_name, tag_value = tag.name, tag.value
                if display: print(tag.name, tag.code, tag.dtype, tag.count, tag.value)

                # below; fix storage problem for ImageDescription tag
                if tag_name == 'ImageDescription': 
                    list_desc = tag_value.split('\n')
                    dict_desc = {}
                    for idx, elm in enumerate(list_desc):
                        split = elm.split('=')
                        dict_desc[split[0]] = split[1]
                    meta[tag_name] = dict_desc
                else:
                    meta[tag_name] = tag_value
            break # just check the first image
    return meta

def tif_write_meta(data,meta,out_path):
    """
    write data and metadata in 'out_path'
    """
    out_meta = {
        'spacing':float(meta['ImageDescription']['spacing']),
        'unit':meta['ImageDescription']['unit'],
        'axes':'ZYX',
    }
    
    extratags = []
    
    tiff.imwrite(
        out_path,
        data=data,
        resolution=(meta['XResolution'],meta['YResolution']),
        metadata=out_meta,
        extratags=extratags,
        imagej=True,
    )

def tif_copy_meta(in_path1, in_path2, out_path):
    """
    store (metadata of in_path1 + data of in_path2) in out_path
    """
    in_meta = tif_read_meta(in_path1)
    data = tiff.imread(in_path2)
    tif_write_meta(data, in_meta, out_path)

def tif_get_spacing(path, res=1e-6):
    """
    get the image spacing stored in the metadata file.
    """
    img_meta = tif_read_meta(path)

    xres = (img_meta["XResolution"][1]/img_meta["XResolution"][0])*res
    yres = (img_meta["YResolution"][1]/img_meta["YResolution"][0])*res
    zres = float(img_meta["ImageDescription"]["spacing"])*res
    # max_dim = min([xres,yres,zres])
    # xres = max_dim / xres
    # yres = max_dim / yres
    # zres = max_dim / zres
    return (xres, yres, zres)

# ----------------------------------------------------------------------------
# 3d viewer

def display_voxels(image, xlim, ylim, zlim, save=False):
    """
    plot using matplotlib a 3d volume from a 3d image
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(image)
    
    ax.set_xlim(xlim[0], xlim[1])  
    ax.set_ylim(ylim[0], ylim[1])  
    ax.set_zlim(zlim[0], zlim[1])
    
    plt.tight_layout()
    plt.savefig('voxel.png') if save else plt.show() 

def display_mesh(mesh, xlim, ylim, zlim, save=False):
    """
    plot using matplotlib a 3d volume from a 3d mesh
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.add_collection3d(mesh)
    
    ax.set_xlim(xlim[0], xlim[1])  
    ax.set_ylim(ylim[0], ylim[1])  
    ax.set_zlim(zlim[0], zlim[1])
    
    plt.tight_layout()
    plt.savefig('mesh.png') if save else plt.show() 

def napari_viewer(img, pred):
    viewer = napari.view_image(img, name='original')
    viewer.add_image(pred, name='pred')
    viewer.layers['pred'].opacity=0.5
    viewer.layers['pred'].colormap='red'
    napari.run()

# ----------------------------------------------------------------------------
# os utils

def abs_path(root, listdir_):
    listdir = listdir_.copy()
    for i in range(len(listdir)):
        listdir[i] = os.path.join(root, listdir[i])
    return listdir

def abs_listdir(path):
    return abs_path(path, sorted(os.listdir(path)))

# ----------------------------------------------------------------------------
# preprocess utils
# from the median image shape predict the size of the patch, the pool, the batch 


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

def resize_segmentation(segmentation, new_shape, order=3):
    '''
    Copied from batch_generator library. Copyleft Fabian Insensee.
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

def resize_3d(img, output_shape, order=3, is_msk=False, monitor_anisotropy=True, anisotropy_threshold=3):
    """
    Resize a 3D image given an output shape.
    
    Parameters
    ----------
    img : numpy.ndarray
        3D image to resample.
    output_shape : tuple, list or numpy.ndarray
        The output shape. Must have an exact length of 3.
    order : int
        The order of the spline interpolation. For images use 3, for mask/label use 0.

    Returns
    -------
    new_img : numpy.ndarray
        Resized image.
    """
    assert len(img.shape)==4, '[Error] Please provided a 3D image with "CWHD" format'
    assert len(output_shape)==3 or len(output_shape)==4, '[Error] Output shape must be "CWHD" or "WHD"'
    
    # convert shape to array
    input_shape = np.array(img.shape)
    output_shape = np.array(output_shape)
    if len(output_shape)==3:
        output_shape = np.append(input_shape[0],output_shape)
    if np.all(input_shape==output_shape): # return image if no reshaping is needed
        return img 
        
    # resize function definition
    resize_fct = resize_segmentation if is_msk else resize
    resize_kwargs = {} if is_msk else {'mode': 'edge', 'anti_aliasing': False}
        
    # separate axis --> [Guillaume] I am not sure about the interest of that... 
    # we only consider the following case: [147,512,513] where the anisotropic axis is undersampled
    # and not: [147,151,512] where the anisotropic axis is oversampled
    anistropy_axes = np.array(input_shape[1:]) / input_shape[1:].min()
    do_anisotropy = monitor_anisotropy and len(anistropy_axes[anistropy_axes>anisotropy_threshold])==2
    if not do_anisotropy:
        anistropy_axes = np.array(output_shape[1:]) / output_shape[1:].min()
        do_anisotropy = monitor_anisotropy and len(anistropy_axes[anistropy_axes>anisotropy_threshold])==2
        
    do_additional_resize = False
    if do_anisotropy: 
        axis = np.argmin(anistropy_axes)
        print("[resize] Anisotropy monitor triggered! Anisotropic axis:", axis)
        
        # as the output_shape and the input_shape might have different dimension
        # along the selected axis, we must use a temporary image.
        tmp_shape = output_shape.copy()
        tmp_shape[axis+1] = input_shape[axis+1]
        
        tmp_img = np.empty(tmp_shape)
        
        length = tmp_shape[axis+1]
        tmp_shape = np.delete(tmp_shape,axis+1)
        
        for c in range(input_shape[0]):
            coord  = [c]+[slice(None)]*len(input_shape[1:])

            for i in range(length):
                coord[axis+1] = i
                tmp_img[tuple(coord)] = resize_fct(img[tuple(coord)], tmp_shape[1:], order=order, **resize_kwargs)
            
        # if output_shape[axis] is different from input_shape[axis]
        # we must resize it again. We do it with order = 0
        if np.any(output_shape!=tmp_img.shape):
            do_additional_resize = True
            order = 0
            img = tmp_img
        else:
            new_img = tmp_img
    
    # normal resizing
    if not do_anisotropy or do_additional_resize:
        new_img = np.empty(output_shape)
        for c in range(input_shape[0]):
            new_img[c] = resize_fct(img[c], output_shape[1:], order=order, **resize_kwargs)
            
    return new_img

# ----------------------------------------------------------------------------
# determine network dynamic architecture

def convert_num_pools(num_pools):
    """
    Set adaptive number of pools
        for example: convert [3,5,5] into [[1 2 2],[2 2 2],[2 2 2],[2 2 2],[1 2 2]]
    """
    max_pool = max(num_pools)
    strides = []
    for i in range(len(num_pools)):
        st = np.ones(max_pool)
        num_zeros = max_pool-num_pools[i]
        for j in range(num_zeros):
            st[j]=0
        st=np.roll(st,-num_zeros//2)
        strides += [st]
    strides = np.array(strides).astype(int).T+1
    # kernels = (strides*3//2).tolist()
    strides = strides.tolist()
    return strides

# ----------------------------------------------------------------------------
# data augmentation utils
# not used yet...

def centered_pad(img, final_size, msk=None):
    """
    centered pad an img and msk to fit the final_size
    """
    final_size = np.array(final_size)
    img_shape = np.array(img.shape[1:])
    
    start = (final_size-np.array(img_shape))//2
    start = start * (start > 0)
    end = final_size-(img_shape+start)
    end = end * (end > 0)
    
    pad = np.append([[0,0]], np.stack((start,end),axis=1), axis=0)
    pad_img = np.pad(img, pad, 'constant', constant_values=0)
    if msk is not None: pad_msk = np.pad(msk, pad, 'constant', constant_values=0)
    
    if msk is not None:
        return pad_img, pad_msk
    else: 
        return pad_img

class SmartPatch:
    """
    Randomly crop and resize the images to a certain crop_shape.
    The global_crop_resize method performs a random crop and resize.
    The local_crop_resize method performs a random crop and resize making sure that the crop 
    is overlapping (to a certain extent, defined by the min_overlap parameter) with the global
    crop previously performed. 
    """
    def __init__(
        self,
        local_crop_shape,
        global_crop_shape,
        min_overlap,
        global_crop_scale=1.0,
        global_crop_min_shape_scale=1.0,
        ):
        """
        Parameters
        ----------
        global_crop_shape : list or tuple of size == 3
            Minimal crop size
        global_crop_scale : float, default=1.0
            Value between 0 and 1. Factor multiplying (img_shape - global_crop_min_shape) and added to the global_crop_min_shape. A value of 1 means that the maximum shape of the global crop will be the image shape. A value of 0 means that the maximum value will be the global_crop_min_shape. 
        global_crop_min_shape_factor : float, default=1.0
            (DEPRECATED?) Factor multiplying the minimal global_crop_shape, 1.0 is a good default
        
        """
        
        self.local_crop_shape = np.array(local_crop_shape)
        self.global_crop_shape = np.array(global_crop_shape)
        self.global_crop_scale = np.array(global_crop_scale)
        self.global_crop_min_shape_scale = np.array(global_crop_min_shape_scale)
        self.alpha = 1  - min_overlap
        
        # internal arguments
        self.global_crop_center = None
        
    def global_crop_resize(self, img, msk=None):
        img_shape = np.array(img.shape)[1:]
        
        # determine crop shape
        min_crop_shape = np.round(self.global_crop_shape * self.global_crop_min_shape_scale).astype(int)
        min_crop_shape = np.minimum(min_crop_shape, img_shape)
        crop_shape = np.random.randint(min_crop_shape, (img_shape-min_crop_shape)*self.global_crop_scale+min_crop_shape+1)
        
        # determine crop coordinates
        rand_start = np.random.randint(0, np.maximum(1,img_shape-crop_shape))
        rand_end = crop_shape+rand_start
        
        self.global_crop_center = (rand_end-rand_start)//2 + rand_start
        
        # crop
        crop_img = img[:,
                        rand_start[0]:rand_end[0], 
                        rand_start[1]:rand_end[1], 
                        rand_start[2]:rand_end[2]]
        
        if msk is not None:
            crop_msk = msk[:,
                            rand_start[0]:rand_end[0], 
                            rand_start[1]:rand_end[1], 
                            rand_start[2]:rand_end[2]]
    
        # temp: resize must be done!
        if not np.array_equal(crop_img.shape[1:], self.global_crop_shape):
            if msk is not None:
                sub = tio.Subject(img=tio.ScalarImage(tensor=crop_img), msk=tio.LabelMap(tensor=crop_msk))
                sub = tio.Resize(self.global_crop_shape)(sub)
                crop_img, crop_msk = sub.img.tensor, sub.msk.tensor
            else:
                crop_img = tio.Resize(self.global_crop_shape)(crop_img)
        
        # returns
        if msk is not None:
            return crop_img, crop_msk
        else:
            return crop_img

    def local_crop_pad(self, img, msk=None):
        """
        global_crop_resize must be called at least once before calling local_crop_pad
        """
        assert self.global_crop_center is not None, "Error! self.global_crop_resize must be called once before self.local_crop_pad."
        
        img_shape = np.array(img.shape)[1:]
        crop_shape = self.local_crop_shape
        
        # determine crop coordinates
        # we make sure that the crop shape overlap with the global crop shape by at least min_overlap
        centers_max_dist = np.round(crop_shape * self.alpha).astype(np.uint8) + (self.global_crop_shape-crop_shape)//2
        local_center_low = np.maximum(crop_shape//2, self.global_crop_center-centers_max_dist)
        local_center_high = np.minimum(img_shape - crop_shape//2, self.global_crop_center+centers_max_dist)
        local_center_high = np.maximum(local_center_high, local_center_low+1)

        local_crop_center = np.random.randint(low=local_center_low, high=local_center_high)
        
        # local
        start = local_crop_center - (self.local_crop_shape//2)
        start = np.maximum(0,start)
        end = start + self.local_crop_shape
        
        crop_img = img[:,
                    start[0]:end[0], 
                    start[1]:end[1], 
                    start[2]:end[2]]
        
        if msk is not None:
            crop_msk = msk[:,
                        start[0]:end[0], 
                        start[1]:end[1], 
                        start[2]:end[2]]
        
        # pad if needed
        if not np.array_equal(crop_img.shape[1:], self.local_crop_shape):
            if msk is not None:
                crop_img, crop_msk = centered_pad(img=crop_img, final_size=self.local_crop_shape, msk=crop_msk)
            else:
                crop_img = centered_pad(img=crop_img, final_size=self.local_crop_shape)
        
        # returns
        if msk is not None:
            return crop_img, crop_msk
        else:
            return crop_img

    def local_crop_resize(self, img, msk=None):
        """
        global_crop_resize must be called at least once before calling local_crop_resize
        """
        assert self.global_crop_center is not None, "Error! self.global_crop_resize must be called once before self.local_crop_resize."

        img_shape = np.array(img.shape)[1:]

        # determine crop shape
        crop_shape = np.random.randint(self.local_crop_scale[0] * img_shape, self.local_crop_scale[1] * img_shape+1)
        
        # determine crop coordinates
        # we make sure that the crop shape overlap with the global crop shape by at least min_overlap
        centers_max_dist = np.round(crop_shape * self.alpha).astype(np.uint8) + (self.global_crop_shape-crop_shape)//2
        local_center_low = np.maximum(crop_shape//2, self.global_crop_center-centers_max_dist)
        local_center_high = np.minimum(img_shape - crop_shape//2, self.global_crop_center+centers_max_dist)
        local_center_high = np.maximum(local_center_high, local_center_low+1)

        local_crop_center = np.random.randint(low=local_center_low, high=local_center_high)
        
        start = local_crop_center - (self.local_crop_shape//2)
        start = np.maximum(0,start)
        end = start + self.local_crop_shape
        
        crop_img = img[:,
                    start[0]:end[0], 
                    start[1]:end[1], 
                    start[2]:end[2]]
        
        if msk is not None:
            crop_msk = msk[:,
                        start[0]:end[0], 
                        start[1]:end[1], 
                        start[2]:end[2]]
        
        # resize if needed
        if not np.array_equal(crop_img.shape[1:], self.local_crop_shape):
            if msk is not None:
                sub = tio.Subject(img=tio.ScalarImage(tensor=crop_img), msk=tio.LabelMap(tensor=crop_msk))
                sub = tio.Resize(self.global_crop_shape)(sub)
                crop_img, crop_msk = sub.img.tensor, sub.msk.tensor
            else:
                crop_img = tio.Resize(self.global_crop_shape)(crop_img)
        
        # returns
        if msk is not None:
            return crop_img, crop_msk
        else:
            return crop_img

# ----------------------------------------------------------------------------
# config utils
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".
# Author: Terro Keras (progressive_growing_of_gans)

class Dict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

def config_to_type(cfg, new_type):
    """Change config type to a new type. This function is recursive and can be use to change the type of nested dictionaries. 
    """
    old_type = type(cfg)
    cfg = new_type(cfg)
    for k,i in cfg.items():
        if type(i)==old_type:
            cfg[k] = config_to_type(cfg[k], new_type)
    return cfg

def save_yaml_config(path, cfg):
    """
    save a configuration in a yaml file.
    path must thus contains a yaml extension.
    example: path='logs/test.yaml'
    """
    cfg = config_to_type(cfg, dict)
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    
def load_yaml_config(path):
    """
    load a yaml stored with the self.save method.
    """
    return config_to_type(yaml.load(open(path),Loader=yaml.FullLoader), Dict)

def nested_dict_pairs_iterator(dic):
    ''' This function accepts a nested dictionary as argument
        and iterate over all values of nested dictionaries
        get from: https://thispointer.com/python-how-to-iterate-over-nested-dictionary-dict-of-dicts/ 
    '''
    # Iterate over all key-value pairs of dict argument
    for key, value in dic.items():
        # Check if value is of dict type
        if isinstance(value, dict) or isinstance(value, Dict):
            # If value is dict then iterate over all its values
            for pair in  nested_dict_pairs_iterator(value):
                yield [key, *pair]
        else:
            # If value is not dict type then yield the value
            yield [key, value]

def nested_dict_change_value(dic, key, value):
    """
    Change all value with a given key from a nested dictionary.
    """
    # Loop through all key-value pairs of a nested dictionary and change the value 
    for pairs in nested_dict_pairs_iterator(dic):
        if key in pairs:
            save = dic[pairs[0]]; i=1
            while i < len(pairs) and pairs[i]!=key:
                save = save[pairs[i]]; i+=1
            save[key] = value
    return dic

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
        line += "\n"
    return line

def replace_line_multiple(line, dic):
    """Similar to replace_line_single but with a dictionary of keys and values.
    """
    for key, value in dic.items():
        line = replace_line_single(line, key, value)
    return line

def save_python_config(
    config_dir,
    base_config = None,
    **kwargs,
    ):
    """
    Save the configuration in a config file. If the path to a base configuration is provided, then update this file with the new auto-configured parameters else use biom3d.config_default file.

    Parameters
    ----------
    config_dir : str
        Path to the configuration folder. If the folder does not exist, then create it.
    base_config : str, default=None
        Path to an existing configuration file which will be updated with the auto-config values.
    **kwargs
        Keyword arguments of the configuration file.

    Returns
    -------
    config_path : str
        Path to the new configuration file.
    
    Examples
    --------
    >>> config_path = save_config_python(\\
        config_dir="configs/",\\
        base_config="configs/pancreas_unet.py",\\
        IMG_DIR="/pancreas/imagesTs_tiny_out",\\
        MSK_DIR="pancreas/labelsTs_tiny_out",\\
        NUM_CLASSES=2,\\
        BATCH_SIZE=2,\\
        AUG_PATCH_SIZE=[56, 288, 288],\\
        PATCH_SIZE=[40, 224, 224],\\
        NUM_POOLS=[3, 5, 5])
    """

    # create the config dir if needed
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)

    # name config path with the current date 
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # copy default config file or use the one given by the user
    if base_config == None:
        try:
            from biom3d import config_default
            config_path = shutil.copy(config_default.__file__, config_dir) 
        except:
            print("[Error] Please provide a base config file or install biom3d.")
            raise RuntimeError
    else: 
        # config_path = shutil.copy(base_config, config_dir)
        config_path = base_config # WARNING: overwriting!

    # if DESC is in kwargs, then it will be used to rename the config file
    basename = os.path.basename(config_path) if "DESC" not in kwargs.keys() else kwargs['DESC']+'.py'
    new_config_name = os.path.join(config_dir, current_time+"-"+basename)
    os.rename(config_path, new_config_name)

    if base_config is not None:
        # keep a copy of the old file
        shutil.copy(new_config_name, config_path)

    # edit the new config file with the auto-config values
    with fileinput.input(files=(new_config_name), inplace=True) as f:
        for line in f:
            # edit the line
            line = replace_line_multiple(line, kwargs)
            # write back in the input file
            print(line, end='') 
    return new_config_name

def load_python_config(config_path):
    """Return the configuration dictionary given the path of the configuration file.
    The configuration file is in Python format.
    
    Adapted from: https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path 
    
    Parameters
    ----------
    config_path : str
        Path of the configuration file. Should have the '.py' extension.
    
    Returns
    -------
    cfg : biom3d.utils.Dict
        Dictionary of the config.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    return config_to_type(config.CONFIG, Dict) # change type from config.Dict to Dict

def adaptive_load_config(config_path):
    """Return the configuration dictionary given the path of the configuration file.
    The configuration file is in Python or YAML format.

    Parameters
    ----------
    config_path : str
        Path of the configuration file. Should have the '.py' or '.yaml' extension.
    
    Returns
    -------
    cfg : biom3d.utils.Dict
        Dictionary of the config.
    """
    extension = config_path[config_path.rfind('.'):]
    if extension=='.py':
        return load_python_config(config_path=config_path)
    elif extension=='.yaml':
        return load_yaml_config(config_path=config_path)
    else:
        print("[Error] Unknow format for config file.")

# ----------------------------------------------------------------------------
# postprocessing utils

def compute_otsu_criteria(im, th):
    """Otsu's method to compute criteria.
    Found here: https://en.wikipedia.org/wiki/Otsu%27s_method
    """
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one of the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

    return weight0 * var0 + weight1 * var1

def otsu_thresholding(im):
    """Otsu's thresholding.
    """
    threshold_range = np.linspace(im.min(), im.max()+1, num=255)
    criterias = [compute_otsu_criteria(im, th) for th in threshold_range]
    best_th = threshold_range[np.argmin(criterias)]
    return best_th

def dist_vec(v1,v2):
    """
    euclidean distance between two vectors (np.array)
    """
    v = v2-v1
    return np.sqrt(np.sum(v*v))

def center(labels, idx):
    """
    return the barycenter of the pixels of label = idx
    """
    return np.mean(np.argwhere(labels == idx), axis=0)

def closest(labels, num):
    """
    return the index of the object the closest to the center of the image.
    num: number of label in the image (background does not count)
    """
    labels_center = np.array(labels.shape)/2
    centers = [center(labels,idx+1) for idx in range(num)]
    dist = [dist_vec(labels_center,c) for c in centers]
    # bug fix, return 1 if dist is empty:
    if len(dist)==0:
        return 1
    else:
        return np.argmin(dist)+1

def keep_center_only(msk):
    """
    return mask (msk) with only the connected component that is the closest 
    to the center of the image.
    """
    labels, num = measure.label(msk, background=0, return_num=True)
    close_idx = closest(labels,num)
    return (labels==close_idx).astype(msk.dtype)*255

def volumes(labels):
    """
    returns the volumes of all the labels in the image
    """
    # return [((labels==idx).astype(int)).sum() for idx in np.unique(labels)]
    return np.unique(labels, return_counts=True)[1]

def keep_big_volumes(msk, thres_rate=0.3):
    """
    Return the mask (msk) with less labels/volumes. Select only the biggest volumes with
    the following strategy: minimum_volume = thres_rate * np.sum(np.square(vol))/np.sum(vol)
    This computation could be seen as the expected volume if the variable volume follows the 
    probability distribution: p(vol) = vol/np.sum(vol) 
    """
    # transform image to label
    labels, num = measure.label(msk, background=0, return_num=True)

    # if empty or single volume, return msk
    if num <= 1:
        return msk

    # compute the volume
    unq_labels,vol = np.unique(labels, return_counts=True)

    # remove bg
    unq_labels = unq_labels[1:]
    vol = vol[1:]

    # compute the expected volume
    # expected_vol = np.sum(np.square(vol))/np.sum(vol)
    # min_vol = expected_vol * thres_rate
    min_vol = thres_rate*otsu_thresholding(vol)

    # keep only the labels for which the volume is big enough
    unq_labels = unq_labels[vol > min_vol]

    # compile the selected volumes into 1 image
    s = (labels==unq_labels[0])
    for i in range(1,len(unq_labels)):
        s += (labels==unq_labels[i])

    return s

def keep_biggest_volume_centered(msk):
    """
    return mask (msk) with only the connected component that is the closest 
    to the center of the image if its volumes is not too small ohterwise returns
    the biggest object (different from the background).
    (too small meaning that its volumes shouldn't smaller than half of the biggest one)
    the final mask intensities are either 0 or msk.max()
    """
    labels, num = measure.label(msk, background=0, return_num=True)
    if num <= 1: # if only one volume, no need to remove something
        return msk
    close_idx = closest(labels,num)
    vol = volumes(labels)
    relative_vol = [vol[close_idx]/vol[idx] for idx in range(1,len(vol))]
    # bug fix, empty prediction (it should not happen)
    if len(relative_vol)==0:
        return msk
    min_rel_vol = np.min(relative_vol)
    if min_rel_vol < 0.5:
        close_idx = np.argmin(relative_vol)+1
    return (labels==close_idx).astype(msk.dtype)*msk.max()

# ----------------------------------------------------------------------------
# test utils

# metric definition
def iou(inputs, targets, smooth=1):
    inter = (inputs & targets).sum()
    union = (inputs | targets).sum()
    return (inter+smooth)/(union+smooth)

def dice(inputs, targets, smooth=1, axis=(-3,-2,-1)):   
    """Dice score between inputs and targets.
    """
    inter = (inputs & targets).sum(axis=axis)   
    dice = (2.*inter + smooth)/(inputs.sum(axis=axis) + targets.sum(axis=axis) + smooth)  
    return dice.mean()

def versus_one(fct, in_path, tg_path, num_classes, single_class=None):
    """
    comparison function between in_path image and tg_path and using the criterion defined by fct
    """
    img1 = adaptive_imread(in_path)[0]
    print("input path",in_path)
    if len(img1.shape)==3:
        img1 = one_hot_fast(img1.astype(np.uint8), num_classes)[1:,...]
    if single_class is not None:
        img1 = img1[single_class,...]
    img1 = (img1 > 0).astype(int)
    
    img2 = adaptive_imread(tg_path)[0]
    print("target path",tg_path)
    if len(img2.shape)==3:
        img2 = one_hot_fast(img2.astype(np.uint8), num_classes)[1:,...]
    if single_class is not None:
        img2 = img2[single_class,...]
    img2 = (img2 > 0).astype(int)
    
    # remove background if needed
    if img1.shape[0]==(img2.shape[0]+1):
        img1 = img1[1:]
    if img2.shape[0]==(img1.shape[0]+1):
        img2 = img2[1:]
    
    if sum(img1.shape)!=sum(img2.shape):
        print("bug:sum(img1.shape)!=sum(img2.shape):")
        print("img1.shape", img1.shape)
        print("img2.shape", img2.shape)
        return
    out = fct(img1, img2)
    return out

# ----------------------------------------------------------------------------
# time utils

class Time:
    def __init__(self, name=None):
        self.name=name
        self.reset()
    
    def reset(self):
        print("Count has been reset!")
        self.start_time = time()
        self.count = 0
    
    def get(self):
        self.count += 1
        return time()-self.start_time
    
    def __str__(self):
        self.count += 1
        out = time() - self.start_time
        self.start_time=time()
        return "[DEBUG] name: {}, count: {}, time: {} seconds".format(self.name, self.count, out)

# ----------------------------------------------------------------------------
