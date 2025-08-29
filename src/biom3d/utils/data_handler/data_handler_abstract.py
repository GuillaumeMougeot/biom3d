"""
DataHandler class is a class made to abstract image loading and saving.

It implement iterator to easily iterate over dataset. This module define the abstract class.
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Literal, Optional
from numpy import ndarray

from enum import Enum

class OutputType(Enum):
    """Possible save type."""

    IMG = "img" ; """Saving an image."""
    MSK = "msk" ; """Saving a mask."""
    FG = "fg" ; """ Saving a foreground."""
    PRED = "pred"; """Saving a prediction"""

class DataHandler :
    """Abstract class that define the interface to save, load and iterate over dataset."""

    images: list ;"""A list of image paths."""
    masks: Optional[list]; """A list of mask paths."""
    fg: Optional[list];"""A list of foreground paths."""
    msk_outpath:Optional[str]; """A path to the masks output (preprocessed masks or predictions)."""
    _images_path_root:str; """The root of images path (eg: The path to .h5 file, the folder where all images are,...)."""
    _masks_path_root:Optional[str]; """The root of masks path (eg: The path to .h5 file, the folder where all masks are,...)."""
    _fg_path_root:Optional[str];"""The root of foregrounds path (eg: The path to .h5 file, the folder where all labels are,...)."""
    _image_index:int;"""The current index in images, masks and fg (at the same time). Is _iterator -1."""
    _iterator: int;"""Used to implement iterator. """
    _size: int;"""Used to implement len, is defined by len(images)."""
    _eval:Optional[Literal["label","pred"]]; """Define if handler is used to evaluation. It allow key based format handlers to override some key restriction to load label and prediction as images."""
    _saver:Optional[type[DataHandler]];"""DataHandler used to save, can be another DataHandler for different output format, self or None (read_only)."""

    def __init__(self):
        """Set default value to attributes, never call it outside a child class. All implementation shall call this one AND set default value to their specific attributes."""
        self._image_index = -1
        self._iterator = 0
        self.fg = None
        self.masks = None
        self._saver  = None
        self._fg_path_root = None
        self._masks_path_root = None
        self._images_path_root = None
        self._fg_path_root = None
        self.msk_outpath = None

    @abstractmethod
    def _input_parse(img_path:str, 
                     msk_path:Optional[str]=None,
                     fg_path:Optional[str]=None,
                     eval:Optional[Literal['label','pred']]=None,
                     img_inner_paths_list:Optional[list]=None,      
                     msk_inner_paths_list:Optional[list]=None,      
                     fg_inner_paths_list:Optional[list]=None,      
                     **kwargs)->None:
        """
        Parse and initialize the inputs. If you want to open files, established connection,etc, check wether it is compatible with multiprocessing and picklable or the data/batchloader will not work.

        Parameters
        ----------
        img_path: str
            Path to input images collection (folder, archive,...).

            .. note::
                It is not necessarily images, for example in eval(), we use a handler for predictions and another for ground truth.
            
        msk_path: str, default=None
            Path to input masks collection (folder, archive,...).

        fg_path: str, default=None
            Path to input foregrounds collection (folder, archive,...).

        eval: "label" | "pred" | None, default=None
            Tell your handler that it is to eval and that it should search for the mask or prediction key in your dataset.

            .. note::
                It is not used in FileHandler as it doesn't use keys, however in .h5, it will make it load images from label or prediction key instead of image key.

        img_inner_path: str
            Path to input images relative to image collection (path in .h5 file, subfolders,...).

        msk_inner_path: str
            Path to input masks relative to image collection (path in .h5 file, subfolders,...).

        fg_inner_path: str
            Path to input foregrounds relative to image collection (path in .h5 file, subfolders,...).

        **kwargs: Compatibility with other implementations.

        Raises
        ------
        ValueError
            If incorrect parameters (eg: non existing ressource,...).
        ConnectionError, HttpError, TimeoutError, ...
                Other exceptions related to the format may be raised; this is not an exhaustive list.
                We recommend not trying to catch these specifically in generic code.     

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def _output_parse(self,msk_outpath:str,model_name:Optional[str]=None,**kwargs)->None:
        """
        Parse and initialize the outputs.

        Parameters
        ----------
        msk_outpath: str, default=None
            Path to output masks collection (folder, archive,...), is created if not existing.
        
        model_name: str, default=None
            Is used to create sub collection specific to model in predictions (eg : predictions/MyModelName) to avoid overwrite.

        **kwargs: Compatibility with other implementations.

        Raises
        ------
        PermissionError, ConnectionError, HttpError, TimeoutError, ...
                Other exceptions related to the format may be raised; this is not an exhaustive list.
                We recommend not trying to catch these specifically in generic code.     

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def _output_parse_preprocess(self,
                                 img_path:str, 
                                 msk_path:Optional[str]=None, 
                                 img_outpath:Optional[str]=None,
                                 msk_outpath:Optional[str]=None,
                                 fg_outpath:Optional[str] = None,
                                 **kwargs)->None:
        """
        Parse and initialize the outputs for preprocessing.

        Parameters
        ----------
        img_path: str
            Path to input images collection (folder, archive,...).

        msk_path: str, default=None
            Path to input masks collection (folder, archive,...).

        img_outpath: str, default=None
            Path to output images collection (folder, archive,...), is created if not existing.

        msk_outpath: str, default=None
            Path to input masks collection (folder, archive,...).

        fg_outpath: str, default=None
            Path to foregrounds masks collection (folder, archive,...).

        **kwargs: Compatibility with other implementations.

        Raises
        ------
        PermissionError, ConnectionError, HttpError, TimeoutError, ...
                Other exceptions related to the format may be raised; this is not an exhaustive list.
                We recommend not trying to catch these specifically in generic code. 

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def open(self,**kwargs):
        """
        Is used to open others inputs. It is basically _input_parse(), however it should close current input (if relevant).

        Have a default implementation.

        Parameters
        ----------
        **kwargs: Same as _input_parse()

        Raises
        ------
        PermissionError, ConnectionError, HttpError, TimeoutError, ...
                Other exceptions related to the format may be raised; this is not an exhaustive list.
                We recommend not trying to catch these specifically in generic code. 
        """
        self._input_parse(**kwargs)

    @abstractmethod
    def get_output(self)->tuple[str,str,str]:
        """
        Return a tuple of three element with the paths of images output, mask output and foreground output. (eg : Path to folders, archive, URLs).

        Have a default implementation.

        Raises
        ------
        NotImplementedError:
            If DataHandler has no _saver (is in read_only)

        Returns
        -------
        img_outpath: str | None
            Path to images output collection.

        msk_outpath: str
            Path to masks output collection.

        fg_outpath : str | None
            Path to foregrounds output collection.
        """
        if self._saver == None : raise NotImplementedError("Cannot get output of read only DataHandler")
        img = self._saver.img_outpath if hasattr(self._saver,'img_outpath') else None
        msk =self._saver.msk_outpath 
        fg = self._saver.fg_outpath if hasattr(self._saver,'fg_outpath') else None
        return img,msk,fg

    @abstractmethod
    def close(self):
        """
        Close connections, open files,...

        Should always be called after the handler is not longer used. By default, this function is called on object destruction. Should not raise error.

        .. note::
            Be careful to also close the _saver and avoid RecursionError if case of self._saver = self.
        """
        pass

    @abstractmethod
    def load(self,path:str)->tuple[ndarray,dict]:
        """
        Load a ressource at given path. It is not necessary to check if ressource is in images, masks or foreground.
    
        .. note:: 
            It is assumed that all the inputs are in the same format (treatable with same handler). It is also assume that foreground are a blob.
        
        Parameters
        ----------
        path:str
            The path to the ressource to load, generally given by the iterator (or self.image[i])
        
        Example
        -------
        >>> for img_path,msk_path,fg_path in handler :
        >>>     img,metadata=handler.load(img_path)
        >>>     msk,_=handler.load(msk_path)
        >>>     fg,_=handler.load(fg_path)

        Raises
        ------
        PermissionError, ConnectionError, HttpError, TimeoutError, ...
                Other exceptions related to the format may be raised; this is not an exhaustive list.
                We recommend not trying to catch these specifically in generic code.         
        Returns
        -------
        img: numpy.ndarray
            The image as a numpy ndarray
        metadata: dict
            Image metadata in a dictionary.
        """
        pass

    @abstractmethod
    def _save(self,fname:str,img:ndarray,out_type:OutputType|str,**kwargs)->str:
        """
        Save an image, mask or foreground. To differentiate between the three, we use :class:`OutputType`.
    
        The ressource will be saved in the output path corresponding to out_type, following the same inner path.

        Example
        -------
        >>> 'Raw/Dataset1/1.tif' will be saved in 'Raw_out/Dataset1/1.tif' if called with out_type = 'img'.

        Parameters
        ----------
        fname:str
            The path of loaded ressource, generally given by the iterator (or self.image[i]), it will be used to determine the path to save.
        img:numpy.ndarray
            The image to save.
        out_type: OutputType | "msk" | "pred" | "raw" | "fg"
            Determine the output type and so the saved path is determine by this (output root) + fname
        **kwargs
            All existing parameters to existing handlers, currently

                overwrite: boolean, default=False
                    HDF5Handler: Will force to overwrite date. Is used only in preprocessing._split_single()

        Raises
        ------
        PermissionError, ConnectionError, HttpError, TimeoutError, ...
                Other exceptions related to the format may be raised; this is not an exhaustive list.
                We recommend not trying to catch these specifically in generic code. 

        Returns
        -------
        path: str
            The path of the resource saved.
        """
        pass

    @abstractmethod
    def insert_prefix_to_name(self,fname:str,prefix:str):
        """
        Insert a prefix to a name to create unique variation for the same name (it is used by Preprocess._split_single).
    
        Example
        -------
        >>> handler.insert_prefix_to_name('Raw/1.tif','0_') -> 'Raw/0_1.tif'

        Returns
        -------
        path: str
            A new path including the prefix.
        """
        pass

    def save(self,fname,img,out_type:OutputType|str,**kwargs)->str:
        """
        Public interface of _save.
        
        It does basic checks then delegate to self._saver._save().

        Parameters
        ----------
        fname:str
            The path of loaded ressource, generally given by the iterator (or self.image[i]), it will be used to determine the path to save.
        img:numpy.ndarray
            The image to save.
        out_type: OutputType | "msk" | "pred" | "raw" | "fg"
            Determine the output type and so the saved path is determine by this (output root) + fname
        **kwargs
            All existing parameters to existing handlers, currently

                overwrite: boolean, default=False
                    HDF5Handler: Will force to overwrite date. Is used only in preprocessing._split_single()
    
        Raises
        ------
        ValueError : 
            If OutputType is 'img' or 'fg' and not in preprocess (so said output has not been initialized), or non existing value in enum (incorrect OutputType).
        
        NotImplementedError : 
            If _saver is None (is in read_only)

        Others
            All error raised by _save()   

        Returns
        -------
        path: str
            The path to saved ressource.
        """
        if isinstance(out_type, str):
            out_type = OutputType(out_type)
        if not self._preprocess and (out_type==OutputType.IMG or out_type==OutputType.FG):
            raise ValueError("Type IMG or FG can't be used if handler isn't in preprocessor mode")
        if self._saver == None : raise NotImplementedError("This handler is in read only")
        return self._saver._save(fname,img,out_type,**kwargs)

    def reset_iterator(self)->None:
        """Reset the _iterator value to 0."""
        self._iterator = 0

    def __iter__(self)->None:
        """Return a new iterator (by calling reset_iterator)."""
        self.reset_iterator()
        return self

    def __next__(self)->tuple[str,str,str]:
        """
        Increments _iterator and _image_index and return a tuple of paths.

        Raises
        ------
            StopIteration

        Returns
        ------- 
        img_path:str
            The path to current image.
        msk_path:str | None
            The path to current mask.
        fg_path:str | None
            The path to current foreground
        """
        # Should increment _iterator and _image_index and return the tuple (path to image, path to mask | None, path to foreground | None) (type of path may depend of implementation) 
        if self._iterator >= self._size:
            raise StopIteration
        self._image_index = self._iterator
        self._iterator += 1
        img_path = self.images[self._image_index]
        msk_path = self.masks[self._image_index] if self.masks != None else None
        fg_path = self.fg[self._image_index] if self.fg != None else None
        return (img_path, msk_path,fg_path)

    def __len__(self)->int:
        """Return the handler's size, so the number of images."""
        return self._size
    
    def __del__(self)->None:
        """Will try to call self.close() on destruction."""
        try : self.close()
        except: pass

    def __enter__(self)->DataHandler:
        """Support for 'with' statement."""
        return self
    
    def __exit__(self,exc_type, exc_value, traceback)->False:
        """Support for 'with' statement: auto-close resources"""
        try:self.close()
        except Exception as e:
            print(f"[Warning] Error during datahandler close(): {e}")
        return False


        