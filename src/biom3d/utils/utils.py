# ----------------------------------------------------------------------------
# a set of utility functions 
# content:
#  - read folds from a csv file
#  - create logs and models directories
# ----------------------------------------------------------------------------
import numpy as np
from datetime import datetime
from time import time 
import os 
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # bug fix: change matplotlib backend 

import warnings
import functools

def deprecated(reason="This function is deprecated."):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(
                f"Call to deprecated function {func.__name__}(). {reason}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapped
    return decorator

# ----------------------------------------------------------------------------
# read folds from a csv file
#TODO use verbose
def get_train_test_df(df, verbose=True):
    """
    Return the train set and the test set
    """
    train_set = np.array(df[df['hold_out']==0].iloc[:,0])
    test_set = np.array(df[df['hold_out']==1].iloc[:,0])
    return train_set, test_set

def get_folds_df(df, verbose=True):
    """
    Return of folds in a list of list
    """
    folds = []
    if df.empty:
        print("[Warning] one of the data DataFrame is empty!")
        return []
    nbof_folds = df['fold'].max()+1
    if verbose:
        print("Number of folds in df: {}".format(nbof_folds))
    
    size_folds = []
    for i in range(nbof_folds):
        folds += [list(df[df['fold']==i].iloc[:,0])]
        size_folds += [len(folds[-1])]
    if verbose:
        print("Size of folds: {}".format(size_folds))
    return folds

def get_folds_train_test_df(df, verbose=True, merge_test=True):
    """
    Return folds from the train set and the test set in a list of list.
    Output: (train_folds, test_folds)
    If merge_test==True then the test folds are merged in one list.
    """
    if verbose:
        print("Training set:")
    train_folds = get_folds_df(df[df['hold_out']==0], verbose)
    
    if verbose:
        print("Testing set:")
    test_folds = get_folds_df(df[df['hold_out']==1], verbose)
    
    if merge_test:
        test_folds_merged = []
        for i in range(len(test_folds)):
            test_folds_merged += test_folds[i]
        test_folds = test_folds_merged
    return train_folds, test_folds

def get_splits_train_val_test(df):
    """
    the splits contains [100%,50%,25%,10%,5%,2%,the rest] of the dataset
    return the train set as a list of list,
    the val and test set as lists
    """
    nbof_splits = df['split'].max()+1
    valset = list(df[(df['split']==-1)*(df['fold']==0)*(df['hold_out']==0)]['filename'])
    testset = list(df[(df['hold_out']==1)]['filename'])
    train_splits = []
    for i in range(nbof_splits):
        train_splits += [list(df[(df['split']==i)*(df['fold']!=0)*(df['hold_out']==0)].iloc[:,0])]
    # adds the whole dataset in the begging of the train_splits list
    train_splits = [list(df[(df['fold']!=0)*(df['hold_out']==0)].iloc[:,0])] + train_splits
    return train_splits, valset, testset

def get_splits_train_val_test_overlapping(df):
    """
    CAREFUL: works only if the splits contains [1/(2**0), 1/(2**1), ..., 1/(2**n), 1/(2**n)] of the training dataset 
    the splits contains of the dataset.
    "overlapping" indicates that every smaller set is contained into all bigger sets.
    return the train set as a list of list,
    the val and test set as lists
    """
    nbof_splits = df['split'].max()+1
    valset = list(df[(df['split']==-1)*(df['fold']==0)*(df['hold_out']==0)]['filename'])
    testset = list(df[(df['hold_out']==1)]['filename'])
    train_splits = []
    for i in range(nbof_splits):
        train_splits += [list(df[(df['split']>=i)*(df['fold']!=0)*(df['hold_out']==0)].iloc[:,0])]
        
    # adds the last set 
    train_splits += [list(df[(df['split']==(nbof_splits-1))*(df['fold']!=0)*(df['hold_out']==0)].iloc[:,0])]
    return train_splits, valset, testset

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

# ----------------------------------------------------------------------------
# determine network dynamic architecture

def convert_num_pools(num_pools,roll_strides=True):
    """
    Set adaptive number of pools
        for example: convert [3,5,5] into [[1 2 2],[2 2 2],[2 2 2],[2 2 2],[1 2 2]]
    """
    max_pool = max(num_pools)
    strides = []
    for i in range(len(num_pools)):
        st = np.ones(max_pool)
        num_zeros = max_pool-num_pools[i]
        for j in range(num_zeros):
            st[j]=0
        if roll_strides : st=np.roll(st,-num_zeros//2)
        strides += [st]
    strides = np.array(strides).astype(int).T+1
    strides = strides.tolist()
    return strides

# ----------------------------------------------------------------------------
# time utils

class Time:
    def __init__(self, name=None):
        self.name=name
        self.reset()
    
    def reset(self):
        print("Count has been reset!")
        self.start_time = time()
        self.count = 0
    
    def get(self):
        self.count += 1
        return time()-self.start_time
    
    def __str__(self):
        self.count += 1
        out = time() - self.start_time
        self.start_time=time()
        return "[DEBUG] name: {}, count: {}, time: {} seconds".format(self.name, self.count, out)

# ----------------------------------------------------------------------------
