from biom3d.utils import deprecated
from biom3d.utils import DataHandler,abs_listdir
import SimpleITK as sitk
import numpy as np
from skimage import io
import tifffile as tiff
from typing import Optional, Tuple
from os.path import isdir, join, dirname,exists,splitext,basename
from os import makedirs
import pickle

class FileHandler(DataHandler):       
    def __init__(self):
        super().__init__()    

    def _input_parse(self,img_path:str, 
                     msk_path:Optional[str]=None,
                     fg_path:Optional[str]=None,
                     img_inner_paths_list:Optional[list]=None,      
                     msk_inner_paths_list:Optional[list]=None,      
                     fg_inner_paths_list:Optional[list]=None,      

    ):      
        # fix bug path/folder/ to path/folder
        if basename(img_path)=='':
            img_path = dirname(img_path)
        if msk_path is not None and basename(msk_path)=='':
            msk_path = dirname(msk_path)
        if img_path!='': raise ValueError("[Error] img_dir must not be empty.")
        if not isdir(img_path) : raise ValueError("[Error] '{img_path}' is not a existing directory.")
        if msk_path != None and not isdir(msk_path) : raise ValueError(f"[Error] '{msk_path}' is not a existing directory.")
        if fg_path != None and not isdir(msk_path) : raise ValueError(f"[Error] '{fg_path}' is not a existing directory.")

        def create_path(folder_path:str,fname:list):
            listdir = []
            for i in fname:
                listdir.append(join(folder_path,i))
            return sorted(listdir)

        self.images:list = abs_listdir(img_path) if img_inner_paths_list is None else create_path(img_path,img_inner_paths_list)
        self._size :int = len(self.images)
        if msk_path is not None:
            self.masks:list = abs_listdir(msk_path) if msk_inner_paths_list is None else create_path(msk_path,msk_inner_paths_list)
            if self._size != len(self.masks): raise ValueError(f"Don't have the same number of images ('{self._size}') and masks ('{len(self.masks)}')")
        if fg_path is not None:
            self.fg:list = abs_listdir(fg_path) if fg_inner_paths_list is None else create_path(fg_path,fg_inner_paths_list)
            if self._size != len(self.fg): raise ValueError(f"Don't have the same number of images ('{self._size}') and masks ('{len(self.fg)}')")
        
        self._iterator :int = 0

    def _output_parse_preprocess(self,img_path:str, msk_path:Optional[str]=None, img_outdir:Optional[str]=None,msk_outdir:Optional[str]=None,fg_outdir:Optional[str] = None,use_tif:bool=False):
        self._use_tif = use_tif
        if img_outdir is None: # name the out dir the same way as the input and add the _out suffix
            img_outdir = img_path+'_out' 
            print("Image output path:", img_outdir)
        if msk_path is not None and msk_outdir is None:
            msk_outdir = msk_path+'_out' 
            print("Mask output path:", msk_outdir)
            if fg_outdir is None:
                # get parent directory of mask dir
                fg_outdir = join(dirname(msk_path), 'fg_out')
                print("Foreground output path:", fg_outdir)
        self.img_outdir=img_outdir 
        self.msk_outdir=msk_outdir
        self.fg_outdir =fg_outdir
        # create output directory if needed
        if not exists(self.img_outdir):
            makedirs(self.img_outdir, exist_ok=True)
        if msk_path is not None and not exists(self.msk_outdir):
            makedirs(self.msk_outdir, exist_ok=True)
            if msk_path is not None and not exists(self.fg_outdir):
                makedirs(self.fg_outdir, exist_ok=True)

    def _output_parse(self,msk_outdir:str):
        self.msk_outdir=msk_outdir
        # create output directory if needed
        if not exists(self.msk_outdir):
            makedirs(self.msk_outdir, exist_ok=True)

    def get_output(self,):
        img = self._saver.img_outdir
        msk = self._saver.msk_outdir if hasattr(self._saver,'msk_outdir') else None
        fg = self._saver.fg_outdir if hasattr(self._saver,'fg_outdir') else None
        return img,msk,fg
    
    def insert_prefix_to_name(self,fname:str,prefix:str):
        name = basename(fname)[0]
        name = prefix+'_'+name
        return name

    def close(self):
        if self._saver != None and self._saver != self : self._saver.close()

    def load(self,fname:str)->Tuple[np.ndarray,dict]:
        if isdir(fname) : raise ValueError(f"Expected an image, found a directory '{fname}'")
        if hasattr(self,'fg') and fname in self.fg : return pickle.pickle.load(open(fname, 'rb')),{}
        else :
            try : return ImageManager.adaptive_imread(fname)
            except : raise ValueError(f"Couldn't read image '{fname}', is it a valid tiff, nifty or numpy ?")

    def _save(self,fname:str,img:np.ndarray,save_fg:bool=False):
        fg_file = join(self.fg_outdir, fname+'.pkl')
        if hasattr(self,'_use_tif'): #In preprocess
            sname = splitext(basename(fname))[0]
            sname += fname+'.tif' if self._use_tif else fname+'.npy'
        else :
            sname = splitext(basename(fname))
        sname = join(self.img_outdir,sname)
        ImageManager.adaptive_imsave(sname,img)
        if save_fg:
            with open(fg_file, 'wb') as handle:
                pickle.dump(fg_file, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_output(self,input_path):
        out = basename(input_path)
        out = join(self.img_outdir,out)
        return out

    def extract_inner_path(self,list):
        path = []
        for i in list:
            path.append(basename(i))
        return sorted(path)
    
class ImageManager:
    @staticmethod
    def _sitk_imread(img_path, return_spacing=True, return_origin=False, return_direction=False):
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

    @staticmethod
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
                img, img_meta = ImageManager._tif_read_imagej(img_path)  # try loading ImageJ metadata for tif files
                return img, img_meta
            except:   
                img_meta = {}    
                try: img_meta["spacing"] = ImageManager._tif_get_spacing(img_path)
                except: img_meta["spacing"] = []
        
                return io.imread(img_path), img_meta 
        elif extension == ".npy":
            return np.load(img_path), {}
        else:
            return ImageManager._sitk_imread(img_path)
        
    @staticmethod
    def _sitk_imsave(img_path, img, metadata={}):
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

    @staticmethod
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
                ImageManager._tif_write_imagej(
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
            ImageManager._sitk_imsave(img_path, img, img_meta)

    # ----------------------------------------------------------------------------
    # tif metadata reader and writer
    @staticmethod
    def _tif_read_imagej(img_path, axes_order='CZYX'):
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

    @staticmethod
    def _tif_write_imagej(img_path, img, img_meta):
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

    @staticmethod
    def _tif_read_meta(tif_path, display=False):
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

    @staticmethod
    def _tif_write_meta(data,meta,out_path):
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

    @staticmethod
    def _tif_copy_meta(in_path1, in_path2, out_path):
        """
        store (metadata of in_path1 + data of in_path2) in out_path
        """
        in_meta = ImageManager._tif_read_meta(in_path1)
        data = tiff.imread(in_path2)
        ImageManager._tif_write_meta(data, in_meta, out_path)

    @staticmethod
    def _tif_get_spacing(path, res=1e-6):
        """
        get the image spacing stored in the metadata file.
        """
        img_meta = ImageManager._tif_read_meta(path)

        xres = (img_meta["XResolution"][1]/img_meta["XResolution"][0])*res
        yres = (img_meta["YResolution"][1]/img_meta["YResolution"][0])*res
        zres = float(img_meta["ImageDescription"]["spacing"])*res
        return (xres, yres, zres)

    
        

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