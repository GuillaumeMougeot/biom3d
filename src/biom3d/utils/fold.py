"""This submodule provides functions to split, save and load folds."""

import numpy as np
from pandas import DataFrame

#TODO use verbose
def get_train_test_df(df:DataFrame, verbose:bool=True)->tuple[np.ndarray,np.ndarray]:
    """
    Extract train and test sets from a DataFrame based on the 'hold_out' column.

    Parameters
    ----------
    df: pandas.DataFrame
        The dataset containing a 'hold_out' column with 0 (train) and 1 (test) labels.
    verbose: bool, default=True
        If True, enables debug printing (currently unused).

    Returns
    -------
    train_set : numpy.ndarray
        Array of training filenames (or sample IDs).
    test_set : numpy.ndarray
        Array of test filenames (or sample IDs).
    """
    train_set = np.array(df[df['hold_out']==0].iloc[:,0])
    test_set = np.array(df[df['hold_out']==1].iloc[:,0])
    return train_set, test_set

def get_folds_df(df:DataFrame, verbose:bool=True)->list[list[str]]:
    """
    Extract folds from a DataFrame into a list of lists.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame with a 'fold' column indicating fold assignment.
    verbose: bool, default=True
        If True, prints the number and size of the folds.

    Returns
    -------
    list of list
        List of folds, each being a list of filenames (or sample IDs).
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

def get_folds_train_test_df(df:DataFrame, 
                            verbose:bool=True, 
                            merge_test:bool=True,
                            )->tuple[list[list[str]],list[list[str]]|list[str]]:
    """
    Extract fold groups from both train and test sets.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame with 'hold_out' and 'fold' columns.
    verbose: bool, default=True
        If True, prints debug info.
    merge_test: bool, default=True
        If True, test folds are merged into one list.

    Returns
    -------
    train_folds: list of list
        List of training folds, each being a list of filenames.
    test_folds: list or list of list
        Test set either as a merged list or as a list of folds.
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

def get_splits_train_val_test(df:DataFrame)->tuple[list[list[str]],list[str],list[str]]:
    """
    Create dataset splits of different sizes, along with validation and test sets.

    Assumes columns:
    - 'split': indicates split index (e.g., 0=50%, 1=25%, etc.)
    - 'fold': used to separate training and validation
    - 'hold_out': 0=train/val, 1=test
    - 'filename': sample identifier

    The splits contains [100%,50%,25%,10%,5%,2%,the rest] of the dataset

    Returns
    -------
    train_splits: list of list
        List of training splits (first is the full training set, followed by reduced ones).
    valset: list
        List of filenames used for validation.
    testset: list
        List of filenames used for testing.
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

def get_splits_train_val_test_overlapping(df:DataFrame)->tuple[list[list[str]],list[str],list[str]]:
    """
    Create overlapping training splits plus validation and test sets.

    Each smaller training subset is fully included in all larger ones.
    Used for dataset scaling experiments (e.g., 100%, 50%, 25%, etc.).

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame with 'split', 'fold', 'hold_out', and 'filename' columns.

    Returns
    -------
    train_splits: list of list
        List of overlapping training subsets.
    valset: list
        List of filenames used for validation.
    testset: list
        List of filenames used for testing.

    Notes
    -----
    Only works if the splits follow descending powers of two.
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
