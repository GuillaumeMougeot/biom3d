# ----------------------------------------------------------------------------
# a set of utility functions 
# content:
#  - read folds from a csv file
#  - create logs and models directories
# ----------------------------------------------------------------------------
from datetime import datetime
import os 

# ----------------------------------------------------------------------------
# create logs and models directories

def create_save_dirs(log_dir, desc, dir_names=['model', 'logs', 'images'], return_base_dir=False):
    """
    Creates saving folders. 

    Arguments:
        dir_names: a list of name of the desired folders.
                   e.g.: ['images','cpkt','summary']
    
    Returns:
        list_dirs: a list of path of the corresponding folders.
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

def abs_path(root, listdir_):
    listdir = listdir_.copy()
    for i in range(len(listdir)):
        listdir[i] = os.path.join(root, listdir[i])
    return listdir

def abs_listdir(path):
    return abs_path(path, sorted(os.listdir(path)))
