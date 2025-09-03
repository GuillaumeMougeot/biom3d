"""Module to visualize and resize images or plot."""

try: import napari
except: pass
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # bug fix: change matplotlib backend 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------------------------------------------------------------------
# 3d viewer
def display_voxels(image: np.ndarray,
                   xlim: tuple[int, int],
                   ylim: tuple[int, int],
                   zlim: tuple[int, int],
                   save: bool = False,
                   ) -> None:
    """
    Plot a 3D volume from a 3D image using matplotlib.

    Parameters
    ----------
    image : numpy.ndarray
        3D numpy array representing the volume to display. Expected shape: (Z, Y, X).
    xlim : tuple of int
        Limits for the x-axis (min, max).
    ylim : tuple of int
        Limits for the y-axis (min, max).
    zlim : tuple of int
        Limits for the z-axis (min, max).
    save : bool, default=False
        If True, saves the plot as "voxel.png". Otherwise, shows the plot.

    Returns
    -------
    None
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(image)
    
    ax.set_xlim(xlim[0], xlim[1])  
    ax.set_ylim(ylim[0], ylim[1])  
    ax.set_zlim(zlim[0], zlim[1])
    
    plt.tight_layout()
    plt.savefig('voxel.png') if save else plt.show() 

def display_mesh(mesh:Poly3DCollection,
                  xlim: tuple[int, int],
                  ylim: tuple[int, int],
                  zlim: tuple[int, int],
                  save: bool = False,
                  ) -> None:
    """
    Plot a 3D volume from a 3D mesh using matplotlib.

    Parameters
    ----------
    mesh : Poly3DCollection
        3D numpy array representing the volume to display. Expected shape: (Z, Y, X).
    xlim : tuple of int
        Limits for the x-axis (min, max).
    ylim : tuple of int
        Limits for the y-axis (min, max).
    zlim : tuple of int
        Limits for the z-axis (min, max).
    save : bool, default=False
        If True, saves the plot as "voxel.png". Otherwise, shows the plot.

    Returns
    -------
    None
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.add_collection3d(mesh)
    
    ax.set_xlim(xlim[0], xlim[1])  
    ax.set_ylim(ylim[0], ylim[1])  
    ax.set_zlim(zlim[0], zlim[1])
    
    plt.tight_layout()
    plt.savefig('mesh.png') if save else plt.show() 

def napari_viewer(img: np.ndarray, pred: np.ndarray)->None:
    """
    Open image and prediction with Napari.
    
    Parameters
    ----------
    img: numpy.ndarray
        Image data.
    pred: numpy.ndarray
        Predicted mask (or just mask).

    Returns
    -------
    None
    """
    viewer = napari.view_image(img, name='original')
    viewer.add_image(pred, name='pred')
    viewer.layers['pred'].opacity=0.5
    viewer.layers['pred'].colormap='red'
    napari.run()

def resize_segmentation(segmentation: np.ndarray,
                        new_shape: tuple[int, ...],
                        order: int = 3,
                        ) -> np.ndarray:
    """
    Resize a segmentation map using one-hot encoding to avoid interpolation artifacts.

    Copied and adapted from the batch_generator library (Fabian Isensee).

    Parameters
    ----------
    segmentation : numpy.ndarray
        The segmentation map to resize. Can be 2D or 3D.
    new_shape : tuple of int
        The desired output shape. Must match the dimensionality of `segmentation`.
    order : int, default=3
        The interpolation order. Use 0 for nearest neighbor (recommended for labels), higher for smoother interpolation.

    Raises
    ------
    AssertionError
        If segmentaion shape and new shape don't the same number of dimensions.

    Returns
    -------
    reshaped: numpy.ndarray
        The resized segmentation map. Same dtype as input.
    """
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "New shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

def resize_3d(img:np.ndarray, 
              output_shape:tuple[int]|list[int]|np.ndarray[int], 
              order:int=3, 
              is_msk:bool=False, 
              monitor_anisotropy:bool=True, 
              anisotropy_threshold:int=3
              )->np.ndarray:
    """
    Resize a 3D image given an output shape.
    
    Parameters
    ----------
    img : numpy.ndarray
        Image to resample, expected shape (C, W, H, D) where C is the channel dimension.
    output_shape : tuple, list or numpy.ndarray
        Desired output shape. Must be of shape (C, W, H, D) or (W, H, D), (C,H,D) or (H,D) and match the dimensionality.
    order : int, default=3
        Interpolation order. Use 3 for smooth images, 0 for masks.
    is_msk : bool, default=False
        Whether the input is a mask. If True, uses nearest-neighbor-like interpolation.
    monitor_anisotropy : bool, default=True
        Whether to check for axis anisotropy and adapt resizing accordingly.
    anisotropy_threshold : int, default=3
        Threshold to detect anisotropy. If the ratio between largest and smallest spatial axis exceeds this, 
        anisotropy is triggered.

    Raises
    ------
    AssertionError
        If image not in 4D.
    AssertionError
        If output shape not in 3D or 4D. 

    Returns
    -------
    new_img : numpy.ndarray
        Resized image.
    """
    
    
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