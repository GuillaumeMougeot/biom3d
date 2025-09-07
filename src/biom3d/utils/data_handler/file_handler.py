from .data_handler_abstract import DataHandler, OutputType
import SimpleITK as sitk
import numpy as np
from skimage import io
import tifffile as tiff
from typing import Any, Literal, Optional
from os.path import isdir, join, dirname,exists,basename,normpath
from os import makedirs, listdir
import pickle
from pathlib import Path
from sys import platform

class FileHandler(DataHandler):       
    def __init__(self):
        super().__init__()    
        
    def _input_parse(self,img_path:str, 
                     msk_path:Optional[str]=None,
                     fg_path:Optional[str]=None,
                     eval:Optional[Literal['label','pred']]=None,
                     img_inner_paths_list:Optional[list]=None,      
                     msk_inner_paths_list:Optional[list]=None,      
                     fg_inner_paths_list:Optional[list]=None,      
                     **kwargs,

    ):      
        # fix bug path/folder/ to path/folder
        img_path = normpath(img_path)
        if msk_path is not None :
            msk_path = normpath(msk_path)
        if img_path=='': raise ValueError("[Error] img_path must not be empty.")
        if not isdir(img_path) : raise ValueError(f"[Error] '{img_path}' is not a existing directory.")
        if msk_path != None and not isdir(msk_path) : raise ValueError(f"[Error] '{msk_path}' is not a existing directory.")
        if fg_path != None and not isdir(fg_path) : raise ValueError(f"[Error] '{fg_path}' is not a existing directory.")

        def create_path(folder_path:str,fname:list):
            listdir = []
            for i in fname:
                listdir.append(join(folder_path,i))
            return sorted(listdir)
        self._eval = eval

        self.images:list = self._recursive_path_list(img_path) if img_inner_paths_list is None else create_path(img_path,img_inner_paths_list)
        self._size :int = len(self.images)
        if msk_path is not None:
            self.masks:list = self._recursive_path_list(msk_path) if msk_inner_paths_list is None else create_path(msk_path,msk_inner_paths_list)
            if self._size != len(self.masks): raise ValueError(f"Don't have the same number of images ('{self._size}') and masks ('{len(self.masks)}')")
        if fg_path is not None:
            self.fg:list = self._recursive_path_list(fg_path) if fg_inner_paths_list is None else create_path(fg_path,fg_inner_paths_list)
            if self._size != len(self.fg): raise ValueError(f"Don't have the same number of images ('{self._size}') and foreground ('{len(self.fg)}')")
        self._fg_path_root = fg_path
        self._masks_path_root = msk_path
        self._images_path_root = img_path
        self._iterator :int = 0

    @staticmethod
    def _recursive_path_list(path):
        li = []
        def recursion(path):
            nonlocal li
            for e in listdir(path) :
                fname=join(path,e)
                if isdir(fname):
                    recursion(fname)
                elif e not in li :
                    li.append(fname)
        recursion(path)
        return sorted(li)


    def _output_parse_preprocess(self,img_path:str, msk_path:Optional[str]=None, img_outpath:Optional[str]=None,msk_outpath:Optional[str]=None,fg_outpath:Optional[str] = None,use_tif:bool=False,**kwargs):
        self._use_tif = use_tif
        img_path = normpath(img_path)
        if msk_path is not None :
            msk_path = normpath(msk_path)
        if img_outpath is None: # name the out dir the same way as the input and add the _out suffix
            img_outpath = img_path+'_out' 
            print("Image output path:", img_outpath)
        if msk_path is not None and msk_outpath is None:
            msk_outpath = msk_path+'_out' 
            print("Mask output path:", msk_outpath)
            if fg_outpath is None:
                # get parent directory of mask dir
                fg_outpath = join(dirname(msk_path), 'fg_out')
                print("Foreground output path:", fg_outpath)
        self.img_outpath=img_outpath 
        self.msk_outpath=msk_outpath
        self.fg_outpath =fg_outpath
        # create output directory if needed
        if not exists(self.img_outpath):
            makedirs(self.img_outpath, exist_ok=True)
        if msk_path is not None and not exists(self.msk_outpath):
            makedirs(self.msk_outpath, exist_ok=True)
            if msk_path is not None and not exists(self.fg_outpath):
                makedirs(self.fg_outpath, exist_ok=True)

        if platform=='win32':
            if self.img_outpath is not None: self.img_outpath = self.img_outpath.replace('\\','\\\\')
            if self.msk_outpath is not None: self.msk_outpath = self.msk_outpath.replace('\\','\\\\')
            if self.fg_outpath is not None: self.fg_outpath = self.fg_outpath.replace('\\','\\\\')

    def _output_parse(self,msk_outpath:str,model_name:Optional[str]=None,**kwargs):
        self.msk_outpath=msk_outpath
        # create output directory if needed
        if not exists(self.msk_outpath):
            makedirs(self.msk_outpath, exist_ok=True)

        # Used for prediction
        if model_name != None :
            self.msk_outpath = join(self.msk_outpath,model_name)
            makedirs(self.msk_outpath, exist_ok=True)

        if platform=='win32' and self.msk_outpath is not None: self.msk_outpath = self.msk_outpath.replace('\\','\\\\')

    @staticmethod
    def extract_inner_path(path_list):
        out_path_list = []
        for p in path_list:
            out_path_list.append(basename(p))
        return out_path_list

    def insert_prefix_to_name(self,fname:str,prefix:str):
        name = basename(fname)[0]
        name = join(dirname(fname),prefix+'_'+name)
        return name

    def close(self):
        if self._saver != None and self._saver != self : 
            self._saver.close()
            self._saver = None

    def open(self,img_path:str, 
                     msk_path:Optional[str]=None,
                     fg_path:Optional[str]=None,
                     img_inner_paths_list:Optional[list]=None,      
                     msk_inner_paths_list:Optional[list]=None,      
                     fg_inner_paths_list:Optional[list]=None,      
                     **kwargs,):
        self._input_parse(
            img_path=img_path,
            msk_path=msk_path,
            fg_path=fg_path,
            img_inner_paths_list=img_inner_paths_list,
            msk_inner_paths_list=msk_inner_paths_list,
            fg_inner_paths_list=fg_inner_paths_list,
            **kwargs)

    def load(self,fname:str)->tuple[np.ndarray,dict]:
        if isdir(fname) : raise ValueError(f"Expected an image, found a directory '{fname}'")
        if self.fg != None and fname in self.fg : return pickle.load(open(fname, 'rb')),{}
        else :
            try : return ImageManager.adaptive_imread(fname)
            except : raise ValueError(f"Couldn't read image '{fname}', is it a valid tiff, nifty or numpy ?")

    def _save(self,fname:str,img:np.ndarray,out_type:OutputType,**kwargs)->str:
        name_str = fname
        fname = Path(fname)
        try :
            if fname.is_relative_to(Path(self._images_path_root)):
                relative = fname.relative_to(Path(self._images_path_root))
            elif self._masks_path_root != None and fname.is_relative_to(Path(self._masks_path_root)):
                relative = fname.relative_to(Path(self._masks_path_root))
            elif self._fg_path_root != None and fname.is_relative_to(Path(self._fg_path_root)):
                relative = fname.relative_to(Path(self._fg_path_root))
        except TypeError :
            if fname.is_absolute(): relative=fname
            elif name_str.startswith('\\'): relative = Path(name_str.lstrip("\\"))
            elif name_str.startswith('/') : relative = Path(name_str.lstrip("/"))
        
        if out_type==OutputType.IMG:
            if hasattr(self,'_use_tif') and self._use_tif: #In preprocess
                relative = relative.with_suffix(".tif")
            ImageManager.adaptive_imsave(str(self.img_outpath / relative),img)
        elif out_type==OutputType.MSK or out_type==OutputType.PRED:
            if hasattr(self,'_use_tif') and self._use_tif : #In preprocess
                relative = relative.with_suffix(".tif")
            ImageManager.adaptive_imsave(str(self.msk_outpath / relative),img)
        elif out_type==OutputType.FG:
            relative = relative.with_suffix(".pkl")
            with open(self.fg_outpath / relative, 'wb') as handle:
                pickle.dump(img, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else :
            raise ValueError("Save only save an 'img', 'mask' or 'fg'")

        return str(self.msk_outpath / relative)
    
class ImageManager:
    """
    Static class to treat different image format.
    
    For the moment, the following format:
    - Numpy
    - Nifty
    - TIFF
    """

    @staticmethod
    def _sitk_imread(img_path:str, 
                     return_spacing:bool=True, 
                     return_origin:bool=False, 
                     return_direction:bool=False,
                     )->tuple[np.ndarray,dict[str,Any]]:
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

        Raises
        ------
        RuntimeError
            If image not in 3 or 4 dimensions.

        Returns
        -------
        img: numpy.ndarray
            The image contained in the file
        meta: dict from str to any
            The image metadata
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
    def adaptive_imread(img_path:str)->tuple[np.ndarray,dict[str,Any]]:
        """
        Load an image file.

        Use skimage imread or sitk imread depending on the file extension:
        
        - `.tif` | `.tiif` → skimage.io.imread
        - `.nii.gz` → SimpleITK.imread
        - `.npy` → numpy.load

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
    def _sitk_imsave(img_path:str, img:np.ndarray, metadata:dict[str,Any]={})->None:
        """
        Image saver for more generic images format.

        See the complete list here : https://simpleitk.readthedocs.io/en/master/IO.html#image-io

        Parameters
        ----------
        img_path: str
            Path to image file, must contain extension.
        img: numpy.ndarray
            Image data.
        metadata: dictionary from str to any, default={}
            Image metadata. Following keys have default values if not found:
                - 'spacing'=(1,1,1)
                - 'origin'=(0,0,0)
                - 'direction'=(1., 0., 0., 0., 1., 0., 0., 0., 1.)

        Returns
        -------
        None
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
    def adaptive_imsave(img_path:str, img:np.ndarray, img_meta:dict[str,Any]={})->None:
        """
        Save an image.

        Use skimage or sitk depending on the file extension:

        - `.tif` | `.tiif` → ImageManager._tif_write_imagej
        - `.nii.gz` → ImageManager._sitk_imsave
        - `.npy` → numpy.save

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
        extension = img_path[img_path.rfind('.'):].lower()
        makedirs(dirname(img_path), exist_ok=True)
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
    def _tif_read_imagej(img_path:str, axes_order:str='CZYX')->tuple[np.ndarray,dict[str,Any]]:
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
            
            img_meta["axes"] = series.axes

        
        return img, img_meta

    @staticmethod
    def _tif_write_imagej(img_path:str, img:np.ndarray, img_meta:dict[str,Any])->None:
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
    def _tif_read_meta(tif_path:str, display:bool=False)->dict[str,Any]:
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
    def _tif_write_meta(data:np.ndarray,
                        meta:dict[str,Any],
                        out_path:str,
                        )->None:
        """
        Write data and metadata in 'out_path'.

        Parameters
        ----------
        data: numpy.ndarray
            Image data.
        meta: dict from str to any
            Image meta data, must contains 
                - 'ImageDescription'->'spacing' 
                - 'ImageDescription'->'unit'
                - 'XResolution'
                - 'YResolution'
        out_path: str
            File to save data.

        Returns
        -------
        None
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
    def _tif_copy_meta(in_path1:str, in_path2:str, out_path:str)->None:
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
        in_meta = ImageManager._tif_read_meta(in_path1)
        data = tiff.imread(in_path2)
        ImageManager._tif_write_meta(data, in_meta, out_path)

    @staticmethod
    def _tif_get_spacing(path:str, res:float=1e-6)->tuple[float,float,float]:
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
        img_meta = ImageManager._tif_read_meta(path)

        xres = (img_meta["XResolution"][1]/img_meta["XResolution"][0])*res
        yres = (img_meta["YResolution"][1]/img_meta["YResolution"][0])*res
        zres = float(img_meta["ImageDescription"]["spacing"])*res
        return (xres, yres, zres)

    
        

