"""This submodule provides function for neural network."""
import numpy as np

def convert_num_pools(num_pools:list[int],roll_strides:bool=True)->list[list[int]]:
    """
    Generate adaptive pooling stride configurations based on the number of pools per axis.

    This utility transforms a list indicating the number of pooling operations along each axis
    into a stride pattern usable for downsampling layers in convolutional architectures.

    Parameters
    ----------
    num_pools: list of int
        List indicating how many pooling steps to apply per axis.
        For example, [3, 5, 5] means:
        - axis 0 will be pooled 3 times,
        - axis 1 and 2 will be pooled 5 times each.
    roll_strides: bool, default=True
        If True, zero-padding is symmetrically centered (rolled), otherwise zeros are left-aligned.

    Returns
    -------
    strides : list of list of int
        A 2D list of shape (max(num_pools), len(num_pools)) representing stride values.
        Each inner list corresponds to the stride per axis at a given depth.
        Strides are either 1 (no pooling) or 2 (pooling).
    
    Examples
    --------
    >>> convert_num_pools([3, 5, 5])
    [[1, 2, 2],
     [2, 2, 2],
     [2, 2, 2],
     [2, 2, 2],
     [1, 2, 2]]
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