"""This submodule contains thresolding and filter that are used in post processing."""

from skimage import measure
import numpy as np

def compute_otsu_criteria(im:np.ndarray, th:float)->float:
    """
    Compute the Otsu criteria value for a given threshold on the image.

    This function implements the core step of Otsu's method, which evaluates
    the within-class variance weighted by class probabilities for a specific threshold.
    The goal is to find the threshold minimizing this weighted variance.
    Found here: https://en.wikipedia.org/wiki/Otsu%27s_method.

    Parameters
    ----------
    im : numpy.ndarray
        Grayscale input image as a 2D numpy array.
    th : float
        Threshold value to evaluate.

    Returns
    -------
    float
        Weighted sum of variances for the two classes separated by the threshold.
        Returns `np.inf` if one class is empty (to ignore this threshold).
    """
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one of the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

    return weight0 * var0 + weight1 * var1

def otsu_thresholding(im:np.ndarray)->float:
    """
    Compute the optimal threshold for an image using Otsu's method.

    This function searches for the threshold value that minimizes the
    weighted within-class variance of the thresholded image.

    Parameters
    ----------
    im : numpy.ndarray
        Grayscale input image as a 2D numpy array.

    Returns
    -------
    float
        Optimal threshold value computed using Otsu's method.
    """
    threshold_range = np.linspace(im.min(), im.max()+1, num=255)
    criterias = [compute_otsu_criteria(im, th) for th in threshold_range]
    best_th = threshold_range[np.argmin(criterias)]
    return best_th

def dist_vec(v1:np.ndarray,v2:np.ndarray)->float:
    """
    Euclidean distance between two vectors (np.array).

    Parameters
    ----------
    v1 : numpy.ndarray
        Vector 1
    v2 : numpy.ndarray
        Vector 2

    Returns
    -------
    float
        Euclidean distance between v1 and v2.
    """
    v = v2-v1
    return np.sqrt(np.sum(v*v))

def center(labels:np.ndarray, idx:int)->np.ndarray:
    """
    Compute the barycenter of pixels belonging to a specific label.

    Parameters
    ----------
    labels : numpy.ndarray
        Label image array where each pixel has an integer label.
    idx : int
        Label index for which to compute the barycenter.

    Returns
    -------
    numpy.ndarray
        Coordinates of the barycenter as a 1D array (e.g. [y, x] or [z, y, x] depending on dimensions).
        If no pixels with the given label are found, returns an empty array.
    """
    return np.mean(np.argwhere(labels == idx), axis=0)

def closest(labels:np.ndarray, num:int)->int:
    """
    Find the label index of the object closest to the center of the image.

    The function computes the barycenter of all objects (labels 1 to num),
    then returns the label of the object whose barycenter is closest to the image center.

    Parameters
    ----------
    labels : numpy.ndarray
        Label image array where each pixel has an integer label.
    num : int
        Number of labels (excluding background) to consider.

    Returns
    -------
    int
        The label index (1-based) of the object closest to the image center.
        Returns 1 if no objects are found.
    """
    labels_center = np.array(labels.shape)/2
    centers = [center(labels,idx+1) for idx in range(num)]
    dist = [dist_vec(labels_center,c) for c in centers]
    # bug fix, return 1 if dist is empty:
    if len(dist)==0:
        return 1
    else:
        return np.argmin(dist)+1

def keep_center_only(msk:np.ndarray)->np.ndarray:
    """
    Keep only the connected component in the mask that is closest to the image center.

    Parameters
    ----------
    msk : numpy.ndarray
        Binary mask (2D or 3D) where connected components are to be analyzed.

    Returns
    -------
    numpy.ndarray
        Mask with only the connected component closest to the center.
        The returned mask has the same dtype as input, with values 0 or 255.
    """
    labels, num = measure.label(msk, background=0, return_num=True)
    close_idx = closest(labels,num)
    return (labels==close_idx).astype(msk.dtype)*255

def volumes(labels:np.ndarray)->np.ndarray:
    """
    Compute the volume (pixel or voxel count) of each label in the label image.

    Parameters
    ----------
    labels : numpy.ndarray
        Label image array where each pixel has an integer label.

    Returns
    -------
    numpy.ndarray
        Array of counts of pixels per label, sorted by label index ascending.
    """
    return np.unique(labels, return_counts=True)[1]

def keep_big_volumes(msk:np.ndarray, thres_rate:float=0.3)->np.ndarray:
    """
    Return a mask keeping only the largest connected components based on a volume threshold.

    The threshold is computed as: min_volume = thres_rate * otsu_thresholding(volumes)
    where `volumes` are the sizes of all connected components (excluding background),
    and `otsu_thresholding` finds an adaptive threshold on the volumes distribution.

    Parameters
    ----------
    msk : numpy.ndarray
        Input binary mask.
    thres_rate : float, default=0.3
        Multiplier for the threshold on volumes.

    Returns
    -------
    numpy.ndarray
        Mask with only the connected components whose volume is greater than the threshold.
        Background remains zero.
    """
    # transform image to label
    labels, num = measure.label(msk, background=0, return_num=True)

    # if empty or single volume, return msk
    if num <= 1:
        return msk

    # compute the volume
    unq_labels,vol = np.unique(labels, return_counts=True)

    # remove bg
    unq_labels = unq_labels[1:]
    vol = vol[1:]

    # compute the expected volume
    # expected_vol = np.sum(np.square(vol))/np.sum(vol)
    # min_vol = expected_vol * thres_rate
    min_vol = thres_rate*otsu_thresholding(vol)

    # keep only the labels for which the volume is big enough
    unq_labels = unq_labels[vol > min_vol]

    # compile the selected volumes into 1 image
    s = (labels==unq_labels[0])
    for i in range(1,len(unq_labels)):
        s += (labels==unq_labels[i])

    return s

def keep_biggest_volume_centered(msk:np.ndarray)->np.ndarray:
    """
    Return a mask with only the connected component closest to the image center, provided its volume is not too small compared to the largest connected component. Otherwise, return the largest connected component.

    "Too small" means its volume is less than half of the largest component.

    The returned mask intensities are either 0 or `msk.max()`.

    Parameters
    ----------
    msk : numpy.ndarray
        Input binary mask.

    Returns
    -------
    numpy.ndarray
        Mask with only one connected component kept.
    """
    labels, num = measure.label(msk, background=0, return_num=True)
    if num <= 1: # if only one volume, no need to remove something
        return msk
    close_idx = closest(labels,num)
    vol = volumes(labels)
    relative_vol = [vol[close_idx]/vol[idx] for idx in range(1,len(vol))]
    # bug fix, empty prediction (it should not happen)
    if len(relative_vol)==0:
        return msk
    min_rel_vol = np.min(relative_vol)
    if min_rel_vol < 0.5:
        close_idx = np.argmin(relative_vol)+1
    return (labels==close_idx).astype(msk.dtype)*msk.max()
