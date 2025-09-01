"""This module implement several version of one hot encoding."""

from typing import Literal, Optional
import numpy as np
from numba import njit,prange


def one_hot(values:np.ndarray, num_classes:Optional[int]=None)->np.ndarray:
    """
    Convert an integer array to one-hot encoding using NumPy.

    Parameters
    ----------
    values : numpy.ndarray
        Integer array of labels to encode.
    num_classes : int, optional
        Total number of classes. If None, inferred as max(values)+1.

    Returns
    -------
    numpy.ndarray
        One-hot encoded array of shape `(num_classes, *values.shape)`, dtype int64.

    Notes
    -----
    - If max value is 255, values are normalized to {0,1}.
    - Unique values are re-indexed to consecutive integers before encoding.
    """
    if num_classes==None: n_values = np.max(values) + 1
    else: n_values = num_classes
        
    # WARNING! potential bug if we have 255 label
    # this function normalize the values to 0,1 if it founds that the maximum of the values if 255
    if values.max()==255: values = (values / 255).astype(np.int64) 
    
    # re-order values if needed
    # for examples if unique values are [2,124,178,250] then they will be changed to [0,1,2,3]
    uni, inv = np.unique(values, return_inverse=True)
    if np.array_equal(uni, np.arange(len(uni))):
        values = np.arange(len(uni))[inv].reshape(values.shape)
        
    out = np.eye(n_values)[values]
    return np.moveaxis(out, -1, 0).astype(np.int64)

@njit
def one_hot_fast_v1(values:np.ndarray, num_classes:Optional[int]=None):
    """
    Numba-accelerated one-hot encoding with simple class heuristics.

    Parameters
    ----------
    values : numpy.ndarray
        Integer array of labels to encode.
    num_classes : int, optional
        Number of classes. If None, inferred from unique values.

    Returns
    -------
    numpy.ndarray
        One-hot encoded array of shape `(num_classes, *values.shape)`, dtype uint8.

    Warnings
    --------
    - If number of unique values < num_classes, missing classes are appended after max value.
    - If max value exceeds num_classes, behavior might be unexpected.
    - For binary classes, applies thresholding if input is not in {0,1}.

    """
    # get unique values
    uni = np.sort(np.unique(values)).astype(np.uint8)

    if num_classes==None: 
        n_values = len(uni)
    else: 
        n_values = num_classes
    
        # if the expected number of class is two then apply a threshold
        if n_values==2 and (len(uni)>2 or uni.max()>1):
            print("[Warning] The number of expected values is 2 but the maximum value is higher than 1. Threshold will be applied.")
            values = (values>uni[0]).astype(np.uint8)
            uni = np.array([0,1]).astype(np.uint8)
        
        # add values if uni is incomplete
        if len(uni)<n_values: 
            # if the maximum value of the array is greater than n_value, it might be an error but still, we add values in the end.
            if values.max() >= n_values:
                print("[Warning] The maximum values in the array is greater than the provided number of classes, this might be unexpected and might cause issues.")
                while len(uni)<n_values:
                    uni = np.append(uni, np.uint8(uni[-1]+1))
            # add missing values in the array by considering that each values are in 0 and n_value
            else:
                uni = np.arange(0,n_values).astype(np.uint8)
        
    # create the one-hot encoded matrix
    out = np.zeros((n_values, *values.shape), dtype=np.uint8)
    for i in range(n_values):
        out[i] = (values==uni[i]).astype(np.uint8)
    return out

@njit
def one_hot_fast(values: np.ndarray, 
                 num_classes: Optional[int] = None, 
                 mapping_mode: Literal['strict','remap','pad'] = 'strict'):
    """
    Transform an integer array into a one-hot encoded array with robust mapping control.

    This function is accelerated with Numba and designed to be a safe, standalone utility.

    Parameters
    ----------
    values: numpy.ndarray
        The integer label array to be encoded.
    num_classes: int, optional
        The total number of classes. If None, this is 
        inferred from the unique values in the array, and `mapping_mode` is
        forced to 'remap'.
    mapping_mode: 'strict','remap' or 'pad', default='strict'
        Controls how input values are mapped to class channels:

        - 'strict' (Default): Safest mode. Requires all values to be within the
          range [0, num_classes-1]. Raises a ValueError if any value is outside
          this range.

        - 'remap': For arbitrarily numbered labels. Remaps the `N` unique values
          in the input array to `[0, 1, ..., N-1]`. Requires that the number of
          unique values equals `num_classes`.

        - 'pad': For correctly-numbered labels where some classes may be missing.
          Creates channels for all classes in `range(num_classes)` and populates
          the ones present in `values`. Raises a ValueError if any value is
          outside the `[0, num_classes-1]` range.

    Raises
    ------
    ValueError 
        If the input values are incompatible with the chosen mode or unknown mapping_mode.

    Returns
    -------
    numpy.ndarray
        The one-hot encoded array of shape `(num_classes, *values.shape)` and dtype `np.uint8`.
    """
    uni = np.unique(values)

    # --- 1. Handle `num_classes = None` (Inference Mode) ---
    if num_classes is None:
        num_classes = len(uni)
        mapping_mode = 'remap' # Remapping is the only logical mode here

    # --- 2. Validate input and prepare for encoding based on mode ---
    if mapping_mode == 'strict':
        if uni.min() < 0 or uni.max() >= num_classes:
            raise ValueError(
                f"In 'strict' mode, all values must be in [0, {num_classes-1}], "
                f"but found values from {uni.min()} to {uni.max()}."
            )
        # In strict mode, the values are already correct. We just encode them.

    elif mapping_mode == 'pad':
        if uni.min() < 0 or uni.max() >= num_classes:
            raise ValueError(
                f"In 'pad' mode, all values must be in [0, {num_classes-1}], "
                f"but found values from {uni.min()} to {uni.max()}."
            )
        # Similar to strict, the values are correct, and the encoding loop will handle padding.

    elif mapping_mode == 'remap':
        if len(uni) != num_classes:
            raise ValueError(
                f"In 'remap' mode, the number of unique values ({len(uni)}) must "
                f"equal num_classes ({num_classes})."
            )
        # Create a lookup table for efficient remapping.
        # This is much faster than searching for each value.
        # Note: This part is not easily JIT-able in a simple way with a hash map.
        # But we can pre-process the `values` array before the Numba loop.
        # The following logic is for a pure-python version, we'll adapt for Numba.

        # Numba-friendly remapping:
        # We need to create a new `values` array where original values are replaced by their index.

        flat_values = values.ravel()
        remapped = np.empty_like(flat_values)

        for idx in prange(flat_values.size):
            val = flat_values[idx]
            for u in range(len(uni)):
                if val == uni[u]:
                    remapped[idx] = u
                    break

        values =  remapped.reshape(values.shape)    
    else:
        raise ValueError(f"Unknown mapping_mode: '{mapping_mode}'")

    # --- 3. Perform the one-hot encoding ---
    # This part is now simple and safe because the data has been validated/corrected.
    out = np.zeros((num_classes, *values.shape), dtype=np.uint8)
    # Using prange for potential parallelization on the outer loop
    for i in prange(num_classes):
        out[i] = (values == i).astype(np.uint8)

    return out