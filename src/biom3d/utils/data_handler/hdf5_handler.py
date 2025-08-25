import pickle
from typing import Literal, Optional
from .data_handler_abstract import DataHandler, OutputType
import h5py
import numpy as np
from pathlib import Path, PurePosixPath
from os.path import exists
import re

PATH_SEPARATOR = r"[\\/]"

class HDF5Handler(DataHandler):
    def __init__(self):
        super().__init__()
        self._output_fg_h5_file = None
        self._output_img_h5_file = None
        self._output_msk_h5_file = None
        self._msk_h5_file = None
        self._fg_h5_file = None
        self._img_h5_file = None
            
    def _input_parse(self,img_path:str, 
                     msk_path:Optional[str]=None,
                     fg_path:Optional[str]=None,
                     eval:Optional[Literal['msk','preds']]=None,
                     img_inner_paths_list:Optional[list]=None,      
                     msk_inner_paths_list:Optional[list]=None,      
                     fg_inner_paths_list:Optional[list]=None,      
                     **kwargs,
                    ):
        if not exists(img_path) : raise ValueError(f"File '{img_path}' can't be found.")
        if msk_path != None and not exists(msk_path) : raise ValueError(f"File '{msk_path}' can't be found.")
        if fg_path != None and not exists(fg_path) : raise ValueError(f"File '{fg_path}' can't be found.")
        self._msk_h5_file = None
        self._fg_h5_file = None
        self._img_h5_file = h5py.File(img_path, "r")

        self._eval=eval
        
        img_key = "raw" if eval is None else eval

        # TODO: For bacmman, Will need a refactor later for more flexibility
        if img_key not in self._img_h5_file : 
            img_key = 'inputs'
        if img_key not in self._img_h5_file : 
            k = list(self._img_h5_file.keys())
            self.close()
            raise KeyError(f"Key '{img_key}' not found in '{img_path}'. Available keys: {k}")
        self.images = self._get_paths(self._img_h5_file,img_key) if img_inner_paths_list is None else img_inner_paths_list
        self._size = len(self.images)

        if msk_path != None:
            self._msk_h5_file = h5py.File(msk_path, "r")
            if "label" not in self._msk_h5_file : 
                k = self._msk_h5_file.keys()
                self.close()
                raise KeyError(f"Key 'label' not found in '{msk_path}'. Available keys: {list(k)}")
            self.masks = self._get_paths(self._msk_h5_file,'label') if msk_inner_paths_list is None else msk_inner_paths_list
            msk_len = len(self.masks)
            if msk_len != self._size : 
                self.close()
                raise ValueError(f"Have '{self._size}' raws but '{msk_len}' labels")

        if fg_path != None:
            self._fg_h5_file = h5py.File(fg_path, "r")
            if "fg" not in self._fg_h5_file : 
                k = self._fg_h5_file.keys()
                self.close()
                raise KeyError(f"Key 'fg' not found in '{fg_path}'. Available keys: {list(k)}")
            self.fg = self._get_paths(self._fg_h5_file,'fg') if fg_inner_paths_list is None else fg_inner_paths_list
            fg_len = len(self.fg)
            if fg_len != self._size : 
                self.close()
                raise ValueError(f"Have '{self._size}' raws but '{fg_len}' foreground")
            
        self._fg_path_root = fg_path
        self._images_path_root = img_path
        self._masks_path_root = msk_path
        self.close()

    def insert_prefix_to_name(self,fname:str,prefix:str):
        fname = fname.strip("/")
        split_name = re.split(PATH_SEPARATOR,fname)
        if split_name[-2] in ["raw","label","pred","fg"]:
            split_name.insert(-3,split_name[-2])
            split_name[-2] = split_name[-1]

        split_name[-2] = prefix+"_"+split_name[-2]
        return "/".join(split_name)


    def _output_parse_preprocess(self,img_path:str, img_outpath:Optional[str]=None,msk_outpath:Optional[str]=None,fg_outpath:Optional[str] = None,**kwargs):
        if img_outpath is None: # name the out dir the same way as the input and add the _out suffix
            path = Path(img_path)
            img_outpath = str(path.with_name(path.stem+'_out.h5')) 
            print("Image output path:", img_outpath)
        if msk_outpath is None:
            msk_outpath = img_outpath
            print("Mask output path:", msk_outpath)
            if fg_outpath is None:
                # get parent directory of mask dir
                fg_outpath = img_outpath
                print("Foreground output path:", fg_outpath)
        self.img_outpath=img_outpath 
        self.msk_outpath=msk_outpath
        self.fg_outpath =fg_outpath
        
        self.close()

    def _output_parse(self,msk_outpath:str,model_name:Optional[str]=None,**kwargs):
        self.msk_outpath = msk_outpath
        if model_name != None and msk_outpath != self._images_path_root: 
            path = Path(self.msk_outpath)
            self.msk_outpath = str(path.with_name(path.stem+'_'+model_name+'.h5'))
        

    def _get_paths(self,archive:h5py.File,label:str)->list:
        paths = []
        def recursive(obj):
            nonlocal paths
            nonlocal label
            if isinstance(obj,h5py.Dataset):
                if label != "fg":
                    for i in range(obj.shape[0]):
                        paths.append(obj.name+"/"+str(i))
                else : paths.append(obj.name+"/"+"0")
            elif isinstance(obj, h5py.Group):
                for key in obj:
                    recursive(obj[key])
        recursive(archive[label])
        return sorted(paths)

    def open(self,img_path:str, 
                     msk_path:Optional[str]=None,
                     fg_path:Optional[str]=None,
                     eval:Optional[Literal['msk','preds']]=None,
                     img_inner_paths_list:Optional[list]=None,      
                     msk_inner_paths_list:Optional[list]=None,      
                     fg_inner_paths_list:Optional[list]=None,      
                     **kwargs,):
        if self._img_h5_file != None : self._img_h5_file.close()
        if self._msk_h5_file != None : self._msk_h5_file.close()
        if self._fg_h5_file != None : self._fg_h5_file.close()
        self._input_parse(img_path,msk_path,fg_path,eval,img_inner_paths_list,msk_inner_paths_list,fg_inner_paths_list,**kwargs)

    def close(self):
        if self._img_h5_file != None : self._img_h5_file.close()
        self._img_h5_file = None
        if self._msk_h5_file != None : self._msk_h5_file.close()
        self._msk_h5_file = None
        if self._fg_h5_file != None : self._fg_h5_file.close()
        self._fg_h5_file = None
        if self._output_fg_h5_file != None : self._output_fg_h5_file.close()
        self._output_fg_h5_file = None
        if self._output_img_h5_file != None : self._output_img_h5_file.close()
        self._output_img_h5_file = None
        if self._output_msk_h5_file != None : self._output_msk_h5_file.close()
        self._output_msk_h5_file = None
        if self._saver != None and self._saver != self : self._saver.close()

    def load(self,fname:str)->tuple[np.ndarray,dict]:
        fname = fname.strip("/")
        fname_split =re.split(PATH_SEPARATOR,fname)
        archive = fname_split[0]
        fname = "/".join(fname_split[:-1])
        index = int(fname_split[-1])

        try:
            if self._eval == "label" or self._eval == "pred":
                fname_split[0] = self._eval
                fname = "/".join(fname_split[:-1])
                self._img_h5_file = h5py.File(self._images_path_root,'r')
                file = self._img_h5_file
            elif archive == "raw" or archive == 'inputs':
                self._img_h5_file = h5py.File(self._images_path_root,'r')
                file = self._img_h5_file
            elif archive == "label":
                self._msk_h5_file = h5py.File(self._masks_path_root,'r')
                file = self._msk_h5_file
            elif archive == "fg":
                self._fg_h5_file = h5py.File(self._fg_path_root,'r')
                file = self._fg_h5_file
                blob = file[fname][()]  
                blob= pickle.loads(blob)
                return blob,{}
            
            img =  np.array(file[fname][index])
            meta = dict(file[fname].attrs.items())
            self.close()
            return img,meta
        except : raise KeyError(f"Given path doesn't exist '{fname}'")

    def _save(self,fname:str,img:np.ndarray,out_type:OutputType,overwrite=False)->str:
        fname=fname.strip("/")
        fname_split = re.split(PATH_SEPARATOR,fname)
        # Transfomr image name (from file for example) to a dataset to prevent conflict and futurely store metadata
        if not fname_split[-1].isdigit() : fname_split.append("0")
        if out_type==OutputType.PRED : 
            if "pred" not in fname_split :
                fname_split.insert(0,"pred")
            fname_split[0] = "pred"
            out_name = self.msk_outpath
            self._output_msk_h5_file = h5py.File(self.msk_outpath,'a')
            out = self._output_msk_h5_file
        elif out_type==OutputType.IMG : 
            if "raw" not in fname_split :
                fname_split.insert(0,"raw")
            fname_split[0] = "raw"
            out_name = self.img_outpath
            self._output_img_h5_file = h5py.File(self.img_outpath,'a')
            out = self._output_img_h5_file
        elif out_type==OutputType.MSK : 
            if "label" not in fname_split :
                fname_split.insert(0,"label")
            fname_split[0] = "label"
            out_name = self.msk_outpath
            self._output_msk_h5_file = h5py.File(self.msk_outpath,'a')
            out = self._output_msk_h5_file
        elif out_type==OutputType.FG : 
            fname_split[0] = "fg"
            out_name = self.fg_outpath
            self._output_fg_h5_file = h5py.File(self.fg_outpath,'a')
            out = self._output_fg_h5_file
        else : raise KeyError(f"Key '{out_type}' is not a valid OutputType'")
        if out_type != OutputType.FG :
            group_path = fname_split[:-2]
            path = "/".join(group_path)
            path = str(PurePosixPath(path))
            dset_name = fname_split[-2]
            group = out.require_group(path)
            # Here overwrite is used like that only because it is called after preprocess._split_single and we know there is only one element in the dataset
            if overwrite:
                if dset_name in group : del out[path+"/"+dset_name]
                maxshape = (None,) + img.shape
                dset = group.create_dataset(
                    name=dset_name,
                    shape=(1,) + img.shape,
                    maxshape=maxshape,
                    dtype=img.dtype
                )
                dset[0] = img
            elif dset_name not in group:
                maxshape = (None,) + img.shape
                dset = group.create_dataset(
                    name=dset_name,
                    shape=(1,) + img.shape,
                    maxshape=maxshape,
                    dtype=img.dtype
                )
                dset[0] = img
            else:
                dset = group[dset_name]
                dset.resize(dset.shape[0]+1,axis=0)
                dset[-1]=img      
        else:
            path = "/".join(fname_split)
            path = PurePosixPath(path)
            group_path = str(path.parent)
            dset_name = fname_split[-1]
            group = out.require_group(group_path)
            blob = pickle.dumps(img)
            group.create_dataset(dset_name, data=np.void(blob))

        self.close()
        return str(out_name)


