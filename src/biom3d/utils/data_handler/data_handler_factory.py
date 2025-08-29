"""Class to instantiate a DataHandler depending on the input and output type."""

from os.path import isdir,splitext,exists,dirname
from os import lstat,getcwd,access,W_OK
from typing import Optional
from urllib.parse import urlparse

from .data_handler_abstract import DataHandler
from .file_handler import FileHandler
from .hdf5_handler import HDF5Handler

class DataHandlerFactory:
    """Class to instantiate a DataHandler depending on the input and output type."""
    
    EXTENSION_MAP: dict[str, type['DataHandler']] = {
        ".h5": HDF5Handler,
        ".hdf5": HDF5Handler,
        "folder": FileHandler
    }

    @staticmethod
    def _is_url(path: str) -> bool:
        """
        Check if given path is an URL.

        Parameters
        ----------
        path: str
            The path to test.

        Returns
        -------
        boolean:
            Whether path is an URL or not.
        """
        return urlparse(path).scheme in ("http", "https", "ftp", "s3")
    
    @staticmethod
    def _is_nonexistent_folder(path: str) -> bool:
        """
        Check if given path refer to a folder that doesn't exist yet.

        Parameters
        ----------
        path: str
            The path to test.

        Returns
        -------
        boolean:
            Wether the path refer to a non existing folder or not.
        """
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
            dir = dirname(pathname) or getcwd()
            return access(dir, W_OK)
    
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

    @staticmethod
    def _detect_handler_type(path: str) -> type['DataHandler']:
        """
        Extract the data format from path and return `DataHandler` subclass fit to treat it if existing, raise `NotImplementedError` else.

        Parameters
        ----------
        path: str
            The path to test

        Raises
        ------
        NotImplementedError:
            If handler or input data not found.

        Returns
        -------
        Datahandler:
            A DataHandler that can treat the data format given by path.
        """
        if isdir(path) or DataHandlerFactory._is_nonexistent_folder(path):
            return DataHandlerFactory.EXTENSION_MAP["folder"]
        _, ext = splitext(path)
        ext = ext.lower()
        if ext in DataHandlerFactory.EXTENSION_MAP:
            return DataHandlerFactory.EXTENSION_MAP[ext]
        raise NotImplementedError(f"No handler found for extension: '{ext}'")
    
    @staticmethod
    def get(input:str,read_only:bool=False,preprocess:bool=False,output:Optional[str]=None,**kwargs)->DataHandler:
        """
        Create a handler which type depend on the input extension.

        Parameters
        ----------
        input: str
            Path to input (Folder path, archive path, url,...). This path will be used as the image path.

        read_only: bool, default = False
            (Optional) Whether handler is in read only.

        output: str, default = None
            (Optional) Path to output, is used if the output type is different from input.

        preprocess: bool, default = False
            (Optional) If it is a preprocessing handler (will create more output).

        **kwargs:

            All existing parameters to existing handlers, currently                 
                msk_path:str, default=None,
                    Generic : mask output path
                fg_path:str, default = None
                    Generic : foreground output path
                eval: "label" | "pred" | None, default=None
                    HDF5Hanlder (and all others that use keys) : Tell your handler that it is to eval and that it should search for the label or prediction key in your dataset key.
                img_inner_paths_list, default=None
                    Generic : A list of path comming from a specific root (eg: The paths inside a .h5 file), used in data/batch loaders.     
                msk_inner_paths_list, default=None
                    Generic : A list of path comming from a specific root (eg: The paths inside a .h5 file), used in data/batch loaders    
                fg_inner_paths_list, default=None
                    Generic : A list of path comming from a specific root (eg: The paths inside a .h5 file), used in data/batch loaders          
                img_outpath:str, default = None,
                    Generic : images output path
                msk_outpath:str, default = None
                    Generic : mask output path
                fg_outpath:str, default = None
                    Generic : foreground output path
                model_name:str, default = None
                    Generic : Used for prediction, if different than `None`, it will be added at the end of path (eg: predictions/MyModelName, predictions.h5["MyModelName"])
                use_tif:bool, default = False
                    FileHandler : If should be saved as tif instead of npy.

        Raises
        ------
        ValueError:
            If parameters `read_only` and `preprocess` are both `True`.

        Returns
        -------
        DataHandler
            A DataHandler specific to input and output type
        """
        if read_only and preprocess : raise ValueError("A preprocess handler need to write and can't be in read_only")
        INPUT = DataHandlerFactory._detect_handler_type(input)
        kwargs["img_path"]=input
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
