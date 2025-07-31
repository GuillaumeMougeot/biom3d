from __future__ import annotations
from biom3d.utils import deprecated
from .data_handler.file_handler import ImageManager

# This file contains only deprecated method, to delete when enough time has passed

#TODO use optional arguments
@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def sitk_imread(img_path, return_spacing=True, return_origin=False, return_direction=False):
    """
    image reader for nii.gz files
    """
    return ImageManager._sitk_imread(img_path,return_spacing,return_origin,return_direction)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def adaptive_imread(img_path):
    """
    use skimage imread or sitk imread depending on the file extension:
    .tif --> skimage.io.imread
    .nii.gz --> SimpleITK.imread
    """
    return ImageManager.adaptive_imread(img_path)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def sitk_imsave(img_path, img, metadata={}):
    """
    image saver for nii gz files
    """
    return ImageManager._sitk_imsave(img_path,img,metadata)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
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
    return ImageManager.adaptive_imsave(img_path,img,img_meta)

# ----------------------------------------------------------------------------
# tif metadata reader and writer
@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
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
    return ImageManager._tif_read_imagej(img_path,axes_order)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def tif_write_imagej(img_path, img, img_meta):
    """Write tif file using metadata in ImageJ format.
    adapted from: https://forum.image.sc/t/python-copy-all-metadata-from-one-multipage-tif-to-another/26597/8
    """
    return ImageManager._tif_write_imagej(img_path,img,img_meta)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def tif_read_meta(tif_path, display=False):
    """
    read the metadata of a tif file and stores them in a python dict.
    if there is a 'ImageDescription' tag, it transforms it as a dictionary
    """
    return ImageManager._tif_read_meta(tif_path,display)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def tif_write_meta(data,meta,out_path):
    """
    write data and metadata in 'out_path'
    """
    return ImageManager._tif_write_meta(data,meta,out_path)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def tif_copy_meta(in_path1, in_path2, out_path):
    """
    store (metadata of in_path1 + data of in_path2) in out_path
    """
    return ImageManager._tif_copy_meta(in_path1,in_path2,out_path)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
# Marked deprecated in August 2025
def tif_get_spacing(path, res=1e-6):
    """
    get the image spacing stored in the metadata file.
    """
    return ImageManager._tif_get_spacing(path,res)