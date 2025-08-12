
import numpy as np


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