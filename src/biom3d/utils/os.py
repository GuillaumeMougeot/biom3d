"""This submodule provides function for file and directory managment."""
from datetime import datetime
import os

def create_save_dirs(log_dir:str, 
                     desc:str, 
                     dir_names:list[str]=['model', 'logs', 'images'], 
                     return_base_dir:bool=False,
                     )->list[str]:
    """
    Create a directory structure for saving models, logs, images, etc.

    Parameters
    ----------
    log_dir: str
        Root directory in which to create the new structure.
    desc: str
        Name of the model that will be appended to the timestamp.
    dir_names: list of str, default=['model', 'logs', 'images']
        Names of the subdirectories to create inside the base directory.
    return_base_dir: bool, default=False
        If True, the base directory path is included in the returned list.

    Returns
    -------
    list_dirs : list of str
        List of full paths to the created subdirectories. Contains root if return_base_dir is True
    """
    list_dirs = []
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = current_time + '-' + desc
    base_dir = os.path.join(log_dir, base_dir)
    for name in dir_names:
        list_dirs += [os.path.join(base_dir, name)]
        if not os.path.exists(list_dirs[-1]):
            os.makedirs(list_dirs[-1])
    if return_base_dir:
        return [base_dir] + list_dirs
    else:
        return list_dirs

# ----------------------------------------------------------------------------
# os utils

def abs_path(root:str, listdir_:list[str])->list[str]:
    """
    Convert a list of filenames into absolute paths using the given root.

    Parameters
    ----------
    root: str
        Root directory to prepend to each filename.
    listdir_: list of str
        List of filenames or relative paths.

    Returns
    -------
    list_abs_paths: list of str
        List of absolute paths constructed by joining `root` and each element of `listdir_`.

    Notes
    ----- 
    Is not recursive.
    """
    listdir = listdir_.copy()
    for i in range(len(listdir)):
        listdir[i] = os.path.join(root, listdir[i])
    return listdir

def abs_listdir(path:str)->list[str]:
    """
    List all files in a directory and return their absolute paths (sorted).

    Parameters
    ----------
    path: str
        Path to the directory to list.

    Returns
    -------
    list_abs_paths: list of str
        Sorted list of absolute paths for each file in the directory.
    """
    return abs_path(path, sorted(os.listdir(path)))
