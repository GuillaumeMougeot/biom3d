from __future__ import annotations
from biom3d.utils import deprecated
import SimpleITK as sitk
import numpy as np
from skimage import io
import tifffile as tiff

#TODO use optional arguments
@deprecated("For image loading/saving, use an DataHandler for more flexibility")
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

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
def adaptive_imread(img_path):
    """
    use skimage imread or sitk imread depending on the file extension:
    .tif --> skimage.io.imread
    .h5 --> h5py
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

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
def sitk_imsave(img_path, img, metadata={}):
    """
    image saver for nii gz files
    """
    if 'spacing' not in metadata.keys():
        metadata['spacing']=(1,1,1)
    if 'origin' not in metadata.keys():
        metadata['origin']=(0,0,0)
    if 'direction' not in metadata.keys():
        metadata['direction']=(1., 0., 0., 0., 1., 0., 0., 0., 1.)
    img_out = sitk.GetImageFromArray(img)
    img_out.SetSpacing(metadata['spacing'])
    img_out.SetOrigin(metadata['origin'])
    img_out.SetDirection(metadata['direction'])
    sitk.WriteImage(img_out, img_path)

@deprecated("For image loading/saving, use an DataHandler for more flexibility")
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
                compression=('zlib'),
                compressionargs={'level': 1})
    elif extension == ".npy":
        np.save(img_path, img)
    else:
        sitk_imsave(img_path, img, img_meta)

# ----------------------------------------------------------------------------
# tif metadata reader and writer
@deprecated("For image loading/saving, use an DataHandler for more flexibility")
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
@deprecated("For image loading/saving, use an DataHandler for more flexibility")
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
            compression=('zlib'),
            compressionargs={'level': 1}
            )
@deprecated("For image loading/saving, use an DataHandler for more flexibility")
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
@deprecated("For image loading/saving, use an DataHandler for more flexibility")
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
@deprecated("For image loading/saving, use an DataHandler for more flexibility")
def tif_copy_meta(in_path1, in_path2, out_path):
    """
    store (metadata of in_path1 + data of in_path2) in out_path
    """
    in_meta = tif_read_meta(in_path1)
    data = tiff.imread(in_path2)
    tif_write_meta(data, in_meta, out_path)
@deprecated("For image loading/saving, use an DataHandler for more flexibility")
def tif_get_spacing(path, res=1e-6):
    """
    get the image spacing stored in the metadata file.
    """
    img_meta = tif_read_meta(path)

    xres = (img_meta["XResolution"][1]/img_meta["XResolution"][0])*res
    yres = (img_meta["YResolution"][1]/img_meta["YResolution"][0])*res
    zres = float(img_meta["ImageDescription"]["spacing"])*res
    return (xres, yres, zres)