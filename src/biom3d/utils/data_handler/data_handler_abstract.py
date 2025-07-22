from __future__ import annotations
from abc import abstractmethod
from typing import Any, Callable, Optional, Type

from enum import Enum

class OutputType(Enum):
    IMG = "img"
    MSK = "msk"
    FG = "fg"

class DataHandler :
    images: list
    masks: Optional[list]
    fg: Optional[list]
    _images_path_root:str
    _masks_path_root:Optional[str]
    _fg_path_root:Optional[str]
    _image_index:int #will be _iterator -1
    _preprocess : bool
    _iterator: int
    _size: int
    _saver:Optional[Type[DataHandler]]

    def __init__(self):
        self._image_index = -1
        self._iterator = 0
        self.fg = None
        self.masks = None
        self._saver  = None
        self._fg_path_root = None
        self._masks_path_root = None

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
    def get_output(self):
        # Will return a tuple of output list
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def load(self,path):
        pass

    @abstractmethod
    def _save(self,fname,img,type:OutputType):
        pass

    @abstractmethod
    def insert_prefix_to_name(self,fname:str,prefix:str):
        # Should insert a prefix to a name to create unique variation for the same name (it is used by Preprocess._split_single)
        pass

    def save(self,fname,img,type:OutputType):
        if self._saver == None : raise NotImplementedError("This handler is in read only")
        self._saver._save(fname,img,type)

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
        msk_path = self.masks[self._image_index] if self.masks != None else None
        fg_path = self.fg[self._image_index] if self.fg != None else None

        return (img_path, msk_path,fg_path)

    def __len__(self):
        return self._size
    
    def __del__(self):
        try : self.close()
        except: pass



        