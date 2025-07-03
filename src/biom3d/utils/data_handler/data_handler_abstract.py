from abc import abstractmethod
from typing import Any, Callable, Optional, Type
from __future__ import annotations
from os.path import isdir,splitext

from biom3d.utils.data_handler.file_handler import FileHandler
from biom3d.utils.data_handler.hdf5_handler import HDF5Handler


class DataHandler :
    images: list
    masks: Optional[list]
    fg: Optional[list]
    _image_index:int #will be _iterator -1
    _iterator: int
    _size: int
    _saver:Optional[Type[DataHandler]]

    def __init__(self):
        self._image_index = -1
        self._iterator = 0

    @abstractmethod
    def _input_parse(self,img_path:str, msk_path:Optional[str]=None):
        pass

    @abstractmethod
    def _output_parse(self):
        pass

    @abstractmethod
    def _output_parse_preprocess(self):
        pass

    @abstractmethod
    def open(self,**kwargs):
        # It is advised to close the previous inputs if applicable
        self._input_parse(**kwargs)

    @abstractmethod
    def get_output(self):
        # Will return a tuple of path to the output (folder for FileHandler, path to archive for hdf5,...)
        pass

    @abstractmethod
    def get_output(self,input_path):
        # Will return a path to the output given an input
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def load(self,path):
        pass

    @abstractmethod
    def _save(self,fname,img,save_fg:bool=False):
        pass

    @abstractmethod
    def insert_prefix_to_name(self,fname:str,prefix:str):
        # Should insert a prefix to a name to create unique variation for the same name (it is used by Preprocess._split_single)
        pass

    def save(self,**kwargs):
        if self._saver == None : raise NotImplementedError("This handler is in read only")
        self._saver._saver(**kwargs)

    def reset_iterator(self):
        self._iterator = 0

    def next(self):
        self._image_index = self._iterator
        self._iterator += 1

    def has_next(self):
        return  self._iterator < self._size

    def iterate(self,function:Callable[...,Any],*args, **kwargs):
        for i in range(self._iterator, self._size,1):
            img = self.images[i]
            if self.masks is not None:
                mask = self.masks[i]
                function(img, mask, *args, **kwargs)
            else:
                function(img, *args, **kwargs)

    def __iter__(self):
        self.reset_iterator()
        return self

    @abstractmethod
    def __next__(self):
        # Should increment _iterator and _image_index and return the tuple (path to image, path to mask | None, path to foregroun | None) (type of path may depend of implementation) 
        if self._iterator >= self._size:
            raise StopIteration
        self._image_index = self._iterator
        self._iterator += 1

        img_path = self.images[self._image_index]
        msk_path = self.masks[self._image_index] if hasattr(self, 'masks') else None
        fg_path = self.fg[self._image_index] if hasattr(self, 'fg') else None

        return (img_path, msk_path,fg_path)

    def __len__(self):
        return self._size
    
    def __del__(self):
        try : self.close()
        except: pass

class DataHandlerFactory:
    EXTENSION_MAP: dict[str, Type['DataHandler']] = {
        ".h5": HDF5Handler,
        ".hdf5": HDF5Handler,
        "folder": FileHandler
    }

    @staticmethod
    def _detect_handler_type(path: str) -> Type['DataHandler']:
        if isdir(path):
            return DataHandlerFactory.EXTENSION_MAP["folder"]
        _, ext = splitext(path)
        ext = ext.lower()
        if ext in DataHandlerFactory.EXTENSION_MAP:
            return DataHandlerFactory.EXTENSION_MAP[ext]
        raise NotImplementedError(f"No handler found for extension: '{ext}'")
    
    @staticmethod
    def get(input:str,read_only:bool=False,preprocess:bool=False,output:Optional[str]=None,**kwargs):
        """
        Create a handler which type depend on the input extension.

        Parameters

        ----------
        input: str
            Path to input (Folder path, archive path, url,...).

        read_only: bool, default = False
            (Optional) Whether handler is in read only

        output: str, default = None
            (Optional) Path to output, is used if the output type is different from input

        preprocess: bool, default = False
            (Optional) If it is a preprocessing handler (will create more output)

        **kwargs:

            All existing parameters to existing handlers, currently
                img_path:str 
                    Generic : images output path                    
                msk_path:str, default=None,
                    Generic : mask output path
                fg_path:str, default = None
                    Generic : foreground output path
                img_outdir:str, default = None,
                    Generic : images output path
                msk_outdir:str, default = None
                    Generic : mask output path
                fg_outdir:str, default = None
                    Generic : foreground output path
                use_tif:bool, default = False
                    FileHandler, if should be saved as tif instead of npy
        Returns
        -------
        DataHandler
            A DataHandler specific to input and output type
        """
        if read_only and preprocess : raise ValueError("A preprocess handler need to write and can't be in read_only")
        handler = DataHandlerFactory._detect_handler_type(input)()
        handler._input_parse(**kwargs)
        if read_only:
            saver=None
        elif output == None:
            saver = handler
        else :
            saver = DataHandlerFactory._detect_handler_type(output)()

        if preprocess and not read_only:
            saver._output_parse_preprocess(**kwargs)
        elif not read_only:    
            saver._output_parse(**kwargs)

        handler._saver = saver
        return handler


        