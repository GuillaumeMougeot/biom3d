from skimage import measure
import numpy as np

def compute_otsu_criteria(im, th):
    """Otsu's method to compute criteria.
    Found here: https://en.wikipedia.org/wiki/Otsu%27s_method
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

def otsu_thresholding(im):
    """Otsu's thresholding.
    """
    threshold_range = np.linspace(im.min(), im.max()+1, num=255)
    criterias = [compute_otsu_criteria(im, th) for th in threshold_range]
    best_th = threshold_range[np.argmin(criterias)]
    return best_th

def dist_vec(v1,v2):
    """
    euclidean distance between two vectors (np.array)
    """
    v = v2-v1
    return np.sqrt(np.sum(v*v))

def center(labels, idx):
    """
    return the barycenter of the pixels of label = idx
    """
    return np.mean(np.argwhere(labels == idx), axis=0)

def closest(labels, num):
    """
    return the index of the object the closest to the center of the image.
    num: number of label in the image (background does not count)
    """
    labels_center = np.array(labels.shape)/2
    centers = [center(labels,idx+1) for idx in range(num)]
    dist = [dist_vec(labels_center,c) for c in centers]
    # bug fix, return 1 if dist is empty:
    if len(dist)==0:
        return 1
    else:
        return np.argmin(dist)+1

def keep_center_only(msk):
    """
    return mask (msk) with only the connected component that is the closest 
    to the center of the image.
    """
    labels, num = measure.label(msk, background=0, return_num=True)
    close_idx = closest(labels,num)
    return (labels==close_idx).astype(msk.dtype)*255

def volumes(labels):
    """
    returns the volumes of all the labels in the image
    """
    return np.unique(labels, return_counts=True)[1]

def keep_big_volumes(msk, thres_rate=0.3):
    """
    Return the mask (msk) with less labels/volumes. Select only the biggest volumes with
    the following strategy: minimum_volume = thres_rate * np.sum(np.square(vol))/np.sum(vol)
    This computation could be seen as the expected volume if the variable volume follows the 
    probability distribution: p(vol) = vol/np.sum(vol) 
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

def keep_biggest_volume_centered(msk):
    """
    return mask (msk) with only the connected component that is the closest 
    to the center of the image if its volumes is not too small ohterwise returns
    the biggest object (different from the background).
    (too small meaning that its volumes shouldn't smaller than half of the biggest one)
    the final mask intensities are either 0 or msk.max()
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
