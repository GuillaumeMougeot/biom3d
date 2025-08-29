from __future__ import annotations
from typing import Any

from numpy import ndarray
from biom3d.utils import deprecated
from .data_handler.file_handler import ImageManager

# This file contains only deprecated method, to delete when enough time has passed

#TODO use optional arguments
@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def sitk_imread(img_path:str, 
                     return_spacing:bool=True, 
                     return_origin:bool=False, 
                     return_direction:bool=False,
                     )->tuple[ndarray,dict[str,Any]]:
    """
    Image reader for more generic images formats.

    See the complete list here : https://simpleitk.readthedocs.io/en/master/IO.html#image-io

    Parameters
    ----------
    img_path: str
        Path to image file, must contain the extension.
    return_spacing:bool, default=True
        Whether to return spacing (not used)
    return_origin:bool, default=False
        Whether to return origin (not used)
    return_direction:bool, default=False
        Whether to return direction (not used)

    Returns
    -------
    img: numpy.ndarray
        The image contained in the file
    meta: dict from str to any
        The image metadata
    """
    return ImageManager._sitk_imread(img_path,return_spacing,return_origin,return_direction)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def adaptive_imread(img_path:str)->tuple[ndarray,dict[str,Any]]:
    """
    Load an image file.

    Use skimage imread or sitk imread depending on the file extension:
    * .tif | .tiif --> skimage.io.imread
    * .nii.gz --> SimpleITK.imread
    * .npy --> numpy.load

    Parameters
    ----------
    img_path: str
        Path to image file, must contain extension.

    Returns
    -------
    img: numpy.ndarray
        The image contained in the file.
    meta: dictionary from str to any
        The image metadata as a dict. Can be empty
    """
    return ImageManager.adaptive_imread(img_path)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def sitk_imsave(img_path:str, img:ndarray, metadata:dict[str,Any]={})->None:
    """
    Image saver for more generic images formats.

    See the complete list here : https://simpleitk.readthedocs.io/en/master/IO.html#image-io

    Parameters
    ----------
    img_path: str
        Path to image file, must contain extension.
    img: numpy.ndarray
        Image data.
    metadata: dictionary from str to any, default={}
        Image metadata. Following keys have default values if not found:
            * 'spacing'=(1,1,1)
            * 'origin'=(0,0,0)
            * 'direction'=(1., 0., 0., 0., 1., 0., 0., 0., 1.)

    Returns
    -------
    None
    """
    return ImageManager._sitk_imsave(img_path,img,metadata)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def adaptive_imsave(img_path:str, img:ndarray, img_meta:dict[str,Any]={})->None:
    """
    Save an image.

    Use skimage or sitk depending on the file extension:
    * .tif | .tiif --> ImageManager._tif_write_imagej
    * .nii.gz --> ImageManager._sitk_imsave
    * .npy --> numpy.save

    Parameters
    ----------
    img_path : str
        Path to the output file.
    img : numpy.ndarray
        Image array.
    metadata: dictionary from str to any, default={}
        Image metadata.

    Returns
    -------
    None
    """
    return ImageManager.adaptive_imsave(img_path,img,img_meta)

# ----------------------------------------------------------------------------
# tif metadata reader and writer
@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def tif_read_imagej(img_path:str, axes_order:str='CZYX')->tuple[ndarray,dict[str,Any]]:
    """Read tif file metadata stored in a ImageJ format.

    Adapted from: https://forum.image.sc/t/python-copy-all-metadata-from-one-multipage-tif-to-another/26597/8

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
    return ImageManager._tif_read_imagej(img_path,axes_order)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def tif_write_imagej(img_path:str, img:ndarray, img_meta:dict[str,Any])->None:
    """
    Write tif file using metadata in ImageJ format.
    
    Adapted from: https://forum.image.sc/t/python-copy-all-metadata-from-one-multipage-tif-to-another/26597/8

    Parameters
    ----------
    img_path : str
        Path to the output file.
    img : numpy.ndarray
        Image array.
    metadata: dictionary from str to any
        Image metadata.

    Returns
    -------
    None
    """
    return ImageManager._tif_write_imagej(img_path,img,img_meta)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def tif_read_meta(tif_path:str, display:bool=False)->dict[str,Any]:
    """
    Read the metadata of a tif file and stores them in a python dict.

    If there is a 'ImageDescription' tag, it transforms it as a dictionary

    Parameters
    ----------
    img_path : str
        Path to the output file.
    display : bool, default=False
        Whether to diplay metadata after reading

    Returns
    -------
    meta: dict of str to any
        Image's metadata.
    """
    return ImageManager._tif_read_meta(tif_path,display)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def tif_write_meta(data:ndarray,
                        meta:dict[str,Any],
                        out_path:str)->None:
    """
    Write data and metadata in 'out_path'.

    Parameters
    ----------
    data:numpy.
        Image data.
    meta: dict from str to any
        Image meta data, must contains 
            * 'ImageDescription'->'spacing' 
            * 'ImageDescription'->'unit'
            * 'XResolution'
            * 'YResolution'
    out_path: str
        File to save data.

    Returns
    -------
    None
    """
    return ImageManager._tif_write_meta(data,meta,out_path)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def tif_copy_meta(in_path1:str, in_path2:str, out_path:str)->None:
    """
    Store (metadata of in_path1 + data of in_path2) in out_path.

    Parameters
    ----------
    in_path1: str
        Path to file where we take metadata.
    in_path2: str
        Path to file where we take data
    out_path: str
        Path to new file.

    Returns
    -------
    None
    """
    return ImageManager._tif_copy_meta(in_path1,in_path2,out_path)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def tif_get_spacing(path:str, res:float=1e-6)->tuple[float,float,float]:
    """
    Get the image spacing stored in the metadata file.

    Parameters
    ----------
    path: str
        Path to file.
    res: float, default=1e-6
        Unit conversion factor applied to resolution values.
        For example, use 1e-6 to convert from microns to meters.

    Returns
    -------
    (xres,yres,zres): tuple of float
        Represent spacing on each dimension.
    """
    return ImageManager._tif_get_spacing(path,res)