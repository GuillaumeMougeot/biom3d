from typing import Optional, Tuple
from .data_handler_abstract import DataHandler
import h5py
import numpy as np
import os
from os.path import join, dirname,exists,splitext,basename

class HDF5Handler(DataHandler):
    def __init__(self):
        super().__init__()
            
    def _input_parse(self,img_path:str, msk_path:Optional[str]=None,fg_path:Optional[str]=None,**kwargs):
        if not exists(img_path) : raise ValueError(f"File '{img_path}' can't be found.")
        if msk_path != None and not exists(msk_path) : raise ValueError(f"File '{msk_path}' can't be found.")
        if fg_path != None and not exists(fg_path) : raise ValueError(f"File '{fg_path}' can't be found.")
        self.msk_h5_file = None
        self.fg_h5_file = None
        self.img_h5_file = h5py.File(img_path, "r")
        if "raw" not in self.img_h5_file : 
            k = self.img_h5_file.keys()
            self.close()
            raise KeyError(f"Key 'raw' not found in '{img_path}'. Available keys: {list(k)}")
        self.images = self._get_paths(self.img_h5_file,'raw')
        self._size = len(self.images)

        if msk_path != None:
            self.msk_h5_file = h5py.File(msk_path, "r")
            if "label" not in self.msk_h5_file : 
                k = self.msk_h5_file.keys()
                self.close()
                raise KeyError(f"Key 'label' not found in '{msk_path}'. Available keys: {list(k)}")
            self.masks = self._get_paths(self.msk_h5_file,'label')
            msk_len = len(self.masks)
            if msk_len != self._size : raise ValueError(f"Have '{self._size}' raws but '{msk_len}' labels")

        if fg_path != None:
            self.fg_h5_file = h5py.File(fg_path, "r")
            if "fg" not in self.fg_h5_file : 
                k = self.fg_h5_file.keys()
                self.close()
                raise KeyError(f"Key 'fg' not found in '{fg_path}'. Available keys: {list(k)}")
            self.fg = self._get_paths(self.fg_h5_file,'fg')
            fg_len = len(self.fg)
            if fg_len != self._size : 
                print("[WARNING] Not the same number of foreground than image, they will be ignored")
                self.fg = None
                self.fg_h5_file.close()

    def insert_prefix_to_name(self,fname:str,prefix:str):
        name = basename(fname)[0]
        name = prefix+'_'+name
        return name


    def _output_parse_preprocess(self,img_path:str,img_outdir:Optional[str]=None,**kwargs):
        if img_outdir is None: # name the out dir the same way as the input and add the _out suffix
            img_outdir = splitext(basename(img_path))[0]+'_out.h5' 
            img_outdir = join(dirname(img_path),img_outdir)
            print("Image output path:", img_outdir)
        self.img_out_path = img_outdir
        self.img_out = h5py.File(img_outdir, "w")

    def _output_parse(self,img_path:str,msk_outdir:str,**kwargs):
        if msk_outdir is None: # name the out dir the same way as the input and add the _out suffix
            img_outdir = splitext(basename(img_path))[0]+'_out.h5' 
            img_outdir = join(dirname(img_path),img_outdir)
            print("Image output path:", img_outdir)
        self.img_out_path = img_outdir
        self.img_out = h5py.File(img_outdir, "w")
        

    def _get_paths(self,archive:h5py.File,label:str)->list:
        paths = []
        def recursive(obj):
            nonlocal paths
            if isinstance(obj,h5py.Dataset):
                for i in range(obj.shape[0]):
                    paths.append(obj.name+'/'+i)
            else :
                for i in obj:
                    recursive(i)
        recursive(archive[label])
        return sorted(paths)

    def get_output(self,):
        return self.img_out_path

    def open(self,img_path,msk_path=None):
        self.img_h5_file.close()
        if self.msk_h5_file != None : self.msk_h5_file.close()
        if self.fg_h5_file != None : self.fg_h5_file.close()
        self._input_parse(img_path,msk_path=msk_path)

    def close(self):
        self.img_h5_file.close()
        if self.msk_h5_file != None : self.msk_h5_file.close()
        if self.fg_h5_file != None : self.fg_h5_file.close()
        if self._saver != None and self._saver != self : self._saver.close()

    def load(self,fname:str)->Tuple[np.ndarray,dict]:
        archive = fname.split("/")[0]
        try:
            if archive == "raw":
                return np.array(self.img_h5_file[fname][:]),dict(self.img_h5_file[fname][:].attrs.items())
            elif archive == "label":
                return np.array(self.msk_h5_file[fname][:]),dict(self.msk_h5_file[fname][:].attrs.items())
            if archive == "fg":
                return np.array(self.fg_h5_file[fname][:]),dict(self.fg_h5_file[fname][:].attrs.items())
        except : raise KeyError("Given path doesn't exist")

    def _save(self,i:str,img:np.ndarray):
        archive = i.split("/")[0]
        if archive not in ['label','raw','fg'] : raise KeyError(f"Key '{archive}' not 'label', 'raw' or 'fg'")
        dset = self._out[archive]
        dset.resize(dset.shape[0]+1,axis=0)
        dset[-1]=img
    

