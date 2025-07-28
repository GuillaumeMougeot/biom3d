from os.path import isdir,splitext,exists
from os import lstat,getcwd,access,W_OK
from typing import Optional, Type
from urllib.parse import urlparse

from .data_handler_abstract import DataHandler
from .file_handler import FileHandler
from .hdf5_handler import HDF5Handler

class DataHandlerFactory:
    EXTENSION_MAP: dict[str, Type['DataHandler']] = {
        ".h5": HDF5Handler,
        ".hdf5": HDF5Handler,
        "folder": FileHandler
    }

    @staticmethod
    def _is_url(path: str) -> bool:
        return urlparse(path).scheme in ("http", "https", "ftp", "s3")
    
    @staticmethod
    def is_nonexistent_folder(path: str) -> bool:
        def is_path_valid(pathname: str) -> bool:
            # Assume path is valid unless proven otherwise
            try:
                if not pathname or pathname.isspace():
                    return False
                lstat(pathname)
                return True
            except (OSError, ValueError):
                return True  # We allow non-existing paths
            except Exception:
                return False
        
        def is_path_creatable(pathname: str) -> bool:
            # Check for writing authorisation
            dirname = dirname(pathname) or getcwd()
            return access(dirname, W_OK)
    
        if DataHandlerFactory._is_url(path):
            return False
        _, ext = splitext(path)
        if ext:  # Probably a file
            return False
        
        try:
            return is_path_valid(path) and (
                exists(path) or is_path_creatable(path)
            )
        except OSError:
            return False


    # Would need update if URL are added
    @staticmethod
    def _detect_handler_type(path: str) -> Type['DataHandler']:
        if isdir(path) or DataHandlerFactory.is_nonexistent_folder(path):
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
                img_outpath:str, default = None,
                    Generic : images output path
                msk_outpath:str, default = None
                    Generic : mask output path
                fg_outpath:str, default = None
                    Generic : foreground output path
                use_tif:bool, default = False
                    FileHandler, if should be saved as tif instead of npy
        Returns
        -------
        DataHandler
            A DataHandler specific to input and output type
        """
        if read_only and preprocess : raise ValueError("A preprocess handler need to write and can't be in read_only")
        INPUT = DataHandlerFactory._detect_handler_type(input)
        handler = INPUT()
        handler._input_parse(**kwargs)
        handler._preprocess=preprocess
        if read_only:
            saver=None
        elif output == None:
            saver = handler
        else :
            OUTPUT = DataHandlerFactory._detect_handler_type(output)
            saver = OUTPUT() if OUTPUT != INPUT else handler

        if not read_only and preprocess :
            saver._preprocess=preprocess
            saver._output_parse_preprocess(**kwargs)
        elif not read_only:    
            saver._preprocess=preprocess
            saver._output_parse(**kwargs)


        handler._saver = saver
        return handler
