#---------------------------------------------------------------------------
# Dataset preparation to fasten the training
#   -normalization
#   -expand dims and one_hot encoding
#   -saving to tif file
#---------------------------------------------------------------------------

from sys import platform
import sys 
import numpy as np
import os 
from tqdm import tqdm
import argparse
import pandas as pd

from biom3d.auto_config import auto_config, data_fingerprint
from biom3d.utils import one_hot_fast, resize_3d, save_python_config,DataHandlerFactory

#---------------------------------------------------------------------------
# Define the CSV file for KFold split

def hold_out(df, ratio=0.1, seed=42):
    """
    Select a set of element from the first column of df.
    The size of the set is len(set)*ratio.
    It is randomly selected.
    The results are stored in a new column in df called 'hold_out'.
    The results is a 0/1 list: 1=selected, 0=not selected
    
    Args:
        df: pd.DataFrame
        ratio: float in [0,1]
        seed: np.random.seed initialisation
    Return:
        df: pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    l = np.array(df.iloc[:,0])
    
    # shuffle the list 
    permut = rng.permutation(len(l))
    inv_permut = np.argsort(permut)
    
    # split the shuffled list
    split = int(len(l)*ratio)
    indices = np.array([1]*split+[0]*(len(l)-split))
    
    # unpermut the list of indice to get back the original
    sort_indices = indices[inv_permut]
    
    # add columns
    df['hold_out'] = sort_indices
    return df

def strat_kfold(df, k=4, seed=43):
    """
    Stratified Kfold. 
    Same as kfold but pay attention to balance the train and test sets.
    df must contains a column named 'hold_out'.
    """
    rng = np.random.default_rng(seed)
    l = np.array(df.iloc[:,0])
    
    holds_out = np.array(df['hold_out'])
    indices_all = np.arange(len(l))
    
    # retrieve train/test indices 
    indices_test = indices_all[holds_out==1]
    indices_train = indices_all[holds_out==0]
    
    # split the list in k folds and shuffle it 
    def split_indices(l):
        kfold_size = len(l)//k
        indices = []
        for i in range(k):
            indices += [i]*kfold_size
        # the remaining indices are randomly assigned
        if len(l[len(indices):])>0:
            for i in range(len(l[len(indices):])):
                alea = rng.integers(0,k,dtype=int)
                indices += [alea]
        indices = np.array(indices)
        assert len(indices) == len(l)
        rng.shuffle(indices) # shuffle the indices
        return indices
    
    folds_train = split_indices(indices_train)
    folds_test = split_indices(indices_test)
    
    # merge folds at the right place
    merge = np.zeros_like(l)
    merge[holds_out==1] = folds_test
    merge[holds_out==0] = folds_train
    
    # add the column to the DataFrame
    df['fold'] = merge
    return df

def generate_kfold_csv(filenames, csv_path, hold_out_rate=0., kfold=5, seed=42):
    """From a list of filenames create a CSV containing three columns:
    - filename: image filename
    - hold: 0 or 1, whether to consider this image as test set
    - fold: 0, 1, ..., K, each corresponds to the validation set images

    Parameters
    ----------
    filenames : list of str
        List of the filenames (not the absolute path).
    csv_path : str
        Path of the output csv file.
    hold_out_rate : float, default=0.
        Float between 0 and 1, rate with which the test split will be selected. 
    kfold : int, default=5
        Number of fold that will be defined
    seed : int, default=42
        Random seed for numpy.random
    """
    df = pd.DataFrame(filenames, columns=['filename'])
    df = hold_out(df, ratio=hold_out_rate, seed=seed)
    df = strat_kfold(df, k=kfold, seed=seed)
    df.to_csv(csv_path, index=False)

#---------------------------------------------------------------------------
# 3D segmentation preprocessing

def resize_img_msk(img, output_shape, msk=None):
    new_img = resize_3d(img, output_shape, order=3)
    if msk is not None:
        new_msk = resize_3d(msk, output_shape, is_msk=True, order=1)
        return new_img, new_msk
    else: 
        return new_img

def get_resample_shape(input_shape, spacing, median_spacing):
    input_shape = np.array(input_shape)
    spacing = np.array(spacing)
    median_spacing = np.array(median_spacing)
    if len(input_shape)==4:
        input_shape=input_shape[1:]
    return np.round(((spacing/median_spacing)[::-1]*input_shape)).astype(int)

def sanity_check(msk, num_classes=None):
    """Check if the mask is correctly annotated.
    """
    uni = np.sort(np.unique(msk))
    if num_classes is None:
        num_classes = len(uni)
        
    assert isinstance(num_classes,int)
    assert num_classes >= 2
    
    if len(msk.shape)==4:
        if msk.shape[0]==1:
            return sanity_check(msk[0], num_classes=num_classes)
        # if we have 4 dimensions in the mask, we consider it one-hot encoded
        # and thus we perform a sanity check for each channel
        else:
            new_msk = []
            for i in range(msk.shape[0]):
                new_msk+=[sanity_check(msk[i], num_classes=2)]
            return np.array(new_msk)
            
    cls = np.arange(num_classes)
    if np.array_equal(uni,cls):
        # the mask is correctly annotated
        return msk
    else:
        # there is something wrong with the annotations
        # depending on the case we make automatic adjustments
        # or we through an error message
        print("[Warning] There is something abnormal with the annotations. Each voxel value must be in range {} but is in range {}.".format(cls, uni))
        if num_classes==2:
            uni2, counts = np.unique(msk,return_counts=True)
            thr = uni2[np.argmax(counts)]
            print("[Warning] All values equal to the most frequent value ({}) will be set to zero.".format(thr))
            # then we apply a threshold to the data
            # for instance: unique [2,127,232] becomes [0,1], 0 being 2 and 1 being 127 and 232
            return (msk != thr).astype(np.uint8)
        elif np.all(np.isin(uni, cls)):
            # then one label is missing in the current mask... but it should work
            print("[Warning] One or more labels are missing.")
            return msk
        elif len(uni)==num_classes:
            # then we re-annotate the unique values in the mask
            # for instance: unique [2,127,232] becomes [0,1,2]
            print("[Warning] Annotation are wrong in the mask, we will re-annotate the mask.")
            new_msk = np.zeros(msk.shape, dtype=msk.dtype)
            for i,c in enumerate(uni):
                new_msk[msk == c] = i
            return new_msk
        else:
            # case like [2,18,128,254] where the number of classes should be 3 are impossible to decide...
            print("[Error] There is an error in the labels that could not be solved automatically.")
            raise RuntimeError

def seg_preprocessor(
    img, 
    img_meta,
    msk=None,
    num_classes=None,
    use_one_hot = False,
    remove_bg = False, 
    median_spacing=[],
    clipping_bounds=[],
    intensity_moments=[],
    channel_axis=0,
    num_channels=1,
    seed = 42,
    ):
    """Segmentation pre-processing.
    """
    do_msk = msk is not None

    # read image and mask
    spacing = None if 'spacing' not in img_meta.keys() else img_meta['spacing']

    if do_msk: 
        # sanity check
        msk = sanity_check(msk, num_classes)

    # expand image dim
    if len(img.shape)==3:
        # keep the input shape, used for preprocessing before prediction
        original_shape = img.shape
        img = np.expand_dims(img, 0)
    elif len(img.shape)==4:
        # we consider as the channel dimension, the smallest dimension
        # if it is the last dim, then we move it to the first
        # the size of other dimensions of the image should be bigger than the channel dim.
        if np.argmin(img.shape)==channel_axis and img.shape[channel_axis]==num_channels:
            img = np.swapaxes(img, 0, channel_axis)
        else:
            print("[Error] Invalid image shape:", img.shape)
        
        # keep the input shape, used for preprocessing before prediction
        original_shape = img.shape
    else:
        raise ValueError("[Error] Invalid image shape for 3D image {}. Skipping image...".format(img.shape))

    assert img.shape[0]==num_channels, "[Error] Invalid image shape {}. Expected to have {} numbers of channel at {} channel axis.".format(img.shape, num_channels, channel_axis)

    # one hot encoding for the mask if needed
    if do_msk and len(msk.shape)!=4: 
        if use_one_hot:
            msk = one_hot_fast(msk, num_classes)
            if remove_bg:
                msk = msk[1:]
        else:
            msk = np.expand_dims(msk, 0)
    elif do_msk and len(msk.shape)==4:
        # normalize each channel
        msk = (msk > msk.min()).astype(np.uint8)

    assert len(img.shape)==4
    if do_msk: assert len(msk.shape)==4

    # clip img
    if len(clipping_bounds)>0:
        img = np.clip(img, clipping_bounds[0], clipping_bounds[1])

    # normalize the image and msk
    # z-score normalization for the image
    if len(intensity_moments)>0:
        img = (img-intensity_moments[0])/intensity_moments[1]
    else:
        img = (img-img.mean())/img.std()
    
    # enhance contrast
    # img = exposure.equalize_hist(img)

    # range image in [-1, 1]
    # img = (img - img.min())/(img.max()-img.min()) * 2 - 1

    # resample the image and mask if needed
    if len(median_spacing)>0 and spacing is not None and len(spacing)>0:
        output_shape = get_resample_shape(img.shape, spacing, median_spacing)
        if do_msk:
            img, msk = resize_img_msk(img, msk=msk, output_shape=output_shape)
        else:
            img = resize_3d(img, output_shape)

    # set image type
    img = img.astype(np.float32)
    if do_msk: msk = msk.astype(np.uint16)
    
    # foreground computation
    if do_msk:
        rng = np.random.default_rng(seed)
        fg={}
        if use_one_hot: start = 0 if remove_bg else 1
        else: start = 1
        for i in range(start,len(msk) if use_one_hot else msk.max()+1):
            fgi = np.argwhere(msk[i] == 1) if use_one_hot else np.argwhere(msk[0] == i)
            if len(fgi)>0:
                num_samples = min(len(fgi), 10000)
                fgi_idx = rng.choice(np.arange(len(fgi)), size=num_samples, replace=False)
                fgi = fgi[fgi_idx,:]
            else:
                fgi = []
            fg[i] = fgi

        if len(fg)==0:
            print("[Warning] Empty foreground!")

    # return
    if do_msk:
        return img, msk, fg 
    else:
        img_meta["original_shape"] = original_shape
        return img, img_meta

#---------------------------------------------------------------------------
# 3D segmentation preprocessing
# Nifti convertion (Medical segmentation decathlon)
# normalization: z-score
# resampling
# intensity normalization
# one_hot encoding

class Preprocessing:
    """A helper class to transform nifti (.nii.gz) and Tiff (.tif or .tiff) images to .npy format and to normalize them.

    Parameters
    ----------
    img_path : str
        Path to the input image collection
    img_outpath : str
        Path to the output image collection
    msk_path : str, optional
        Path to the input mask collection
    msk_outpath : str, optional
        Path to the output mask collection
    fg_outpath : str, optional
        Foreground location, eventually later used by the dataloader.
    num_classes : int, optional
        Number of classes (channel) in the masks. Required by the 
    remove_bg : bool, default=True
        Whether to remove the background in the one-hot encoded mask. Remove the background is done when training with sigmoid activations instead of softmax.
    median_size : list, optional
        Median size of the image dataset. Is used to check the channel axis.
    median_spacing : list, optional
        A list of length 3 containing the median spacing of the input images. Median_spacing must not be transposed: for example, median_spacing might be [0.8, 0.8, 2.5] if median shape of the training image is [40,224,224].
    clipping_bounds : list, optional
        A list of length 2 containing the intensity clipping boundary. In nnUNet implementation it corresponds to the 0.5 an 99.5 percentile of the intensities of the voxels of the training images located inside the masks regions.
    intensity_moments : list, optional
        Mean and variance of the intensity of the images voxels in the masks regions. These values are used to normalize the image. 
    use_tif : bool, default=True
        Use tif format to save the preprocessed images instead of npy format.
    split_rate_for_single_img : float, default=0.2
        If a single image is present in image/mask folders, then the image/mask are split in 2 portions of size split_rate_for_single_img*largest_dimension for validation and split_rate_for_single_img*(1-largest_dimension) for training.
    num_kfolds : int, default=5
        Number of K-fold for cross validation.
    """
    def __init__(
        self,
        img_path,
        img_outpath = None,
        msk_path = None, # if None, only images are preprocesses not the masks
        msk_outpath = None,
        fg_outpath = None, # foreground location, eventually used by the dataloader
        num_classes = None, # just for debug when empty masks are provided
        use_one_hot = False,
        remove_bg = False, # keep the background in labels 
        median_size = [],
        median_spacing=[],
        clipping_bounds=[],
        intensity_moments=[],
        use_tif=False, # use tif instead of npy 
        split_rate_for_single_img=0.25,
        num_kfolds=5,
        ):
               

        self.img_path=img_path
        self.msk_path=msk_path
        self.handler = DataHandlerFactory.get(
            self.img_path,
            preprocess=True,
            output=img_outpath,
            img_path = self.img_path,
            msk_path = self.msk_path,
            img_outpath = img_outpath,
            msk_outpath = msk_outpath,
            fg_outpath = fg_outpath,
            use_tif=use_tif,
        )


        # create csv along with the img folder
        self.csv_path = os.path.join(os.path.dirname(img_path), 'folds.csv')

        self.num_classes = num_classes

        self.remove_bg = remove_bg

        # median size serves to determine the number of channel
        # and the channel axis
        self.median_size = np.array(median_size)

        self.num_channels = 1
        self.channel_axis = 0
        self.img_outpath, self.msk_outpath, self.fg_outpath = self.handler.get_output()

        # if the 3D image has 4 dimensions then there is a channel dimension.
        if len(self.median_size)==4:
            # the channel dimension is consider to be the smallest dimension
            # this could cause problem in case where there are more z than c for instance...
            self.num_channels = np.min(median_size)
            self.channel_axis = np.argmin(self.median_size)
            if self.channel_axis != 0:
                print("[Warning] 4 dimensions detected and channel axis is {}. All image dimensions will be swapped.".format(self.channel_axis))
            self.median_size[[0,self.channel_axis]] = self.median_size[[self.channel_axis,0]]
            self.median_size = self.median_size[1:]
        self.median_spacing = np.array(median_spacing)
        self.clipping_bounds = np.array(clipping_bounds)
        self.intensity_moments = intensity_moments
        self.use_tif = use_tif

        self.split_rate_for_single_img = split_rate_for_single_img

        self.use_one_hot = use_one_hot

        self.img_len = len(self.handler)
        self.num_kfolds = num_kfolds
        if self.num_kfolds * 2 > self.img_len:
            self.num_kfolds = max(self.img_len // 2, 2)
            print("[Warning] The number of images {} is smaller than twice the number of folds {}. The number of folds will be reduced to {}.".format(self.img_len, num_kfolds * 2, self.num_kfolds))
        
    def _split_single(self):
        """
        if there is only a single image/mask in each folder, then split them both in two portions with self.split_rate_for_single_img
        """

        # read image and mask
        img, metadata = self.handler.load(self.handler.images[0])
        msk, _ = self.handler.load(self.handler.masks[0])

        # determine the slicing indices to crop an image along its maximum dimension
        idx = lambda start,end,shape: tuple(slice(s) if s!=max(shape) else slice(start,end) for s in shape)

        # slicing indices of the image
        # validation is cropped along its largest dimension in the interval [0, self.split_rate_for_single_img*largest_dim]
        # training is cropped along its largest dimension in the interval [self.split_rate_for_single_img*largest_dim, largest_dim]
        s = max(img.shape)
        val_img_idx = idx(start=0, end=int(np.floor(self.split_rate_for_single_img*s)), shape=img.shape)
        train_img_idx = idx(start=int(np.floor(self.split_rate_for_single_img*s)), end=s, shape=img.shape)

        # idem for the mask indices
        s = max(msk.shape)
        val_msk_idx = idx(start=0, end=int(np.floor(self.split_rate_for_single_img*s)), shape=msk.shape)
        train_msk_idx = idx(start=int(np.floor(self.split_rate_for_single_img*s)), end=s, shape=msk.shape)

        # crop the images and masks
        val_img = img[val_img_idx]
        train_img = img[train_img_idx]
        val_msk = msk[val_msk_idx]
        train_msk = msk[train_msk_idx]

        # save the images and masks 
        # validation names start with a 0
        # training names start with a 1
        handler_tmp = DataHandlerFactory.get(
            self.img_path,
            output=self.msk_path,
            preprocess=True,
            read_only=False,
            img_path= self.img_path,
            msk_path=self.msk_path,
            use_tif=self.use_tif,
            )

        # validation
        val_img_path = self.handler.insert_prefix_to_name(self.handler.images[0],'0')
        handler_tmp.save(val_img_path,val_img,"img")
        val_msk_path = self.handler.insert_prefix_to_name(self.handler.masks[0],'0')
        handler_tmp.save(val_msk_path,val_msk,"msk")

        train_img_path = self.handler.insert_prefix_to_name(self.handler.images[0],'1')
        handler_tmp.save(train_img_path,train_img,"img")
        train_msk_path = self.handler.insert_prefix_to_name(self.handler.masks[0],'1')
        handler_tmp.save(train_msk_path,train_msk,"msk")

        
        images,masks,_ = handler_tmp.get_output()
        self.handler.open(images,masks)
        handler_tmp.close()

        # generate the csv file
        df = pd.DataFrame([train_img_path, val_img_path], columns=['filename'])
        df['hold_out'] = [0,0]
        df['fold'] = [1,0]
        df.to_csv(self.csv_path, index=False)
        return metadata
    
    def run(self, debug=False):
        """Start the preprocessing.

        Parameters
        ----------
        debug : boolean, default=False
            Whether to display image filenames while preprocessing.
        """
        print("Preprocessing...")
        # if there is only a single image/mask, then split them both in two portions
        image_was_split = False
        if self.img_len==1 and self.msk_path is not None:
            print("Single image found per folder. Split the images...")
            split_meta = self._split_single()
            image_was_split = True
        
        if debug: ran = self.handler
        else: ran = tqdm(self.handler,file=sys.stdout)
        for i,m,_ in ran:
            # print image name if debug mode
            if debug: 
                print("[{}/{}] Preprocessing:{}".format(self.handler._image_index,len(self.handler),i))

            img,img_meta = self.handler.load(i)
            if self.msk_path is not None:
                msk, _ = self.handler.load(m)
                img, msk, fg = seg_preprocessor(
                    img                 =img, 
                    img_meta            =img_meta if not image_was_split else split_meta,
                    msk                 =msk,
                    num_classes         =self.num_classes,
                    use_one_hot         =self.use_one_hot,
                    remove_bg           =self.remove_bg, 
                    median_spacing      =self.median_spacing,
                    clipping_bounds     =self.clipping_bounds,
                    intensity_moments   =self.intensity_moments,
                    channel_axis        =self.channel_axis,
                    num_channels        =self.num_channels,
                    )
            else:
                img, _ = seg_preprocessor(
                    img                 =img, 
                    img_meta            =img_meta,
                    median_spacing      =self.median_spacing,
                    clipping_bounds     =self.clipping_bounds,
                    intensity_moments   =self.intensity_moments,
                    channel_axis        =self.channel_axis,
                    num_channels        =self.num_channels,
                    )

            # sanity check to be sure that all images have the save number of channel
            s = img.shape
            if len(s)==4: # only for images with 4 dimensionalities
                if i==0: self.num_channels = s[0]
                else: assert len(s)==4 and self.num_channels==s[0], "[Error] Not all images have {} channels. Problematic image: {}".format(self.num_channels, i)

            # save image
            self.handler.save(i,img,"img")

            # save mask
            if self.msk_outpath is not None: 
                self.handler.save(m,msk,"msk")
                self.handler.save(m,fg,"fg")

        # create csv file
        filenames = sorted(self.handler.get_output()[0])
        if not image_was_split:
            generate_kfold_csv(filenames, self.csv_path, kfold=self.num_kfolds)

        print("Done preprocessing!")

#---------------------------------------------------------------------------

def auto_config_preprocess(
        img_path, 
        msk_path, 
        num_classes, 
        config_dir, 
        base_config, 
        img_outpath=None,
        msk_outpath=None,
        use_one_hot=False,
        ct_norm=False,
        remove_bg=False, 
        use_tif=False,
        desc="unet", 
        max_dim=128,
        num_epochs=1000,
        num_workers=6,
        skip_preprocessing=False,
        no_auto_config=False,
        logs_dir='logs/',
        print_param=False,
        debug=False,
        ):
    """Helper function to do auto-config and preprocessing.
    """
    
    median_size, median_spacing, mean, std, perc_005, perc_995 = data_fingerprint(img_path, msk_path if ct_norm else None)
    if not print_param:
        print("Data fingerprint:")
        print("Median size:", median_size)
        print("Median spacing:", median_spacing)
        print("Mean intensity:", mean)
        print("Standard deviation of intensities:", std)
        print("0.5% percentile of intensities:", perc_005)
        print("99.5% percentile of intensities:", perc_995)
        print("")

    if ct_norm:
        if not print_param: print("Computing data fingerprint for CT normalization...")
        clipping_bounds = [perc_005, perc_995]
        intensity_moments = [mean, std]
        if not print_param: print("Done!")
    else:
        clipping_bounds = []
        intensity_moments = []

    p=Preprocessing(
        img_path=img_path,
        msk_path=msk_path,
        img_outpath=img_outpath,
        msk_outpath=msk_outpath,
        num_classes=num_classes+1,
        use_one_hot=use_one_hot,
        remove_bg=remove_bg,
        use_tif=use_tif,
        median_spacing=median_spacing,
        median_size=median_size,
        clipping_bounds=clipping_bounds,
        intensity_moments=intensity_moments,
    )


    if not skip_preprocessing:
        p.run(debug=debug)


    if not no_auto_config:
        if not print_param: print("Start auto-configuration")
        handler = DataHandlerFactory.get(
            img_path,
            read_only=True,
            output=None,
            img_path = img_path,
        )

        

        batch, aug_patch, patch, pool = auto_config(
            median=p.median_size,
            img_path=img_path if p.median_size is None else None,
            max_dims=(max_dim, max_dim, max_dim),
            max_batch = len(handler)//20, # we limit batch to avoid overfitting
            )
        
        # convert path for windows systems before writing them
        if platform=='win32':
            if p.csv_path is not None: p.csv_path = p.csv_path.replace('\\','\\\\')

        config_path = save_python_config(
            config_dir=config_dir,
            base_config=base_config,

            # store hyper-parameters in the config file:
            IMG_PATH=p.img_outpath,
            MSK_PATH=p.msk_outpath,
            FG_PATH=p.fg_outpath,
            CSV_DIR=p.csv_path,
            NUM_CLASSES=num_classes,
            NUM_CHANNELS=p.num_channels,
            CHANNEL_AXIS=p.channel_axis,
            BATCH_SIZE=batch,
            AUG_PATCH_SIZE=aug_patch,
            PATCH_SIZE=patch,
            NUM_POOLS=pool,
            MEDIAN_SPACING=median_spacing,
            CLIPPING_BOUNDS=clipping_bounds,
            INTENSITY_MOMENTS=intensity_moments,
            DESC=desc,
            NB_EPOCHS=num_epochs,
            NUM_WORKERS=num_workers,
            LOG_DIR=logs_dir,
        )

        if not print_param: print("Auto-config done! Configuration saved in: ", config_path)
        if print_param:
            print(batch)
            print(patch)
            print(aug_patch)
            print(pool)
            print(config_path)

        return config_path

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Dataset preprocessing for training purpose.")
    parser.add_argument("--img_path","--img_dir",dest="img_path", type=str,required=True,
        help="Path of the images collection")
    parser.add_argument("--msk_path","--msk_dir",dest="msk_path", type=str, default=None,
        help="(default=None) Path to the masks/labels collection")
    parser.add_argument("--img_outpath","--img_outdir",dest="img_outpath", type=str, default=None,
        help="(default=None : Current directory) Path to the ouput of the preprocessed images")
    parser.add_argument("--msk_outpath","--msk_outdir",dest="msk_outpath", type=str, default=None,
        help="(default=None : Current directory) Path to the output of the preprocessed masks/labels")
    parser.add_argument("--num_classes", type=int, default=1,
        help="(default=1) Number of classes (types of objects) in the dataset. The background is not included.")
    parser.add_argument("--max_dim", type=int, default=128,
        help="(default=128) max_dim^3 determines the maximum size of patch for auto-config.")
    parser.add_argument("--num_epochs", type=int, default=1000,
        help="(default=1000) Number of epochs for the training.")
    parser.add_argument("--num_workers", type=int, default=6,
        help="(default=6) Number of workers for the training. Half of it will be used for validation.")
    parser.add_argument("--config_dir", type=str, default='configs/',
        help="(default=\'configs/\') Configuration folder to save the auto-configuration.")
    parser.add_argument("--logs_dir", type=str, default='logs/',
        help="(default=\'logs/\') Builder output folder to save the model.")
    parser.add_argument("--base_config", type=str, default=None,
        help="(default=None) Optional. Path to an existing configuration file which will be updated with the preprocessed values.")
    parser.add_argument("--desc", type=str, default='unet_default',
        help="(default=unet_default) Optional. A name used to describe the model.")
    parser.add_argument("--use_tif", default=False,  action='store_true', dest='use_tif',
        help="(default=False) Whether to use tif format to save the preprocessed images instead of npy format. Tif files are easily readable with viewers such as Napari and takes fewer disk space but are slower to load and may slow down the training process.") 
    parser.add_argument("--use_one_hot", default=False,  action='store_true', dest='use_one_hot',
        help="(default=False) Whether to use one hot encoding of the mask. Can slow down the training.") 
    parser.add_argument("--remove_bg", default=False,  action='store_true', dest='remove_bg',
        help="(default=False) If use one hot, remove the background in masks. Remove the bg to use with sigmoid activation maps (not softmax).") 
    parser.add_argument("--no_auto_config", default=False,  action='store_true', dest='no_auto_config',
        help="(default=False) For debugging, deactivate auto-configuration.") 
    parser.add_argument("--ct_norm", default=False,  action='store_true', dest='ct_norm',
        help="(default=False) Whether to use CT-Scan normalization routine (cf. nnUNet).") 
    parser.add_argument("--skip_preprocessing", default=False,  action='store_true', dest='skip_preprocessing',
        help="(default=False) Whether to skip the preprocessing. Only for debugging.") 
    parser.add_argument("--remote", default=False,  action='store_true', dest='remote',
        help="(default=False) Whether to print auto-config parameters. Used for remote preprocessing using the GUI.") 
    parser.add_argument("--debug", default=False,  action='store_true', dest='debug',
        help="(default=False) Debug mode. Whether to print all image filenames while preprocessing.")    
    args = parser.parse_args()

    auto_config_preprocess(
        img_path=args.img_path, 
        msk_path=args.msk_path, 
        num_classes=args.num_classes, 
        config_dir=args.config_dir, 
        base_config=args.base_config, 
        img_outpath=args.img_outpath,
        msk_outpath=args.msk_outpath,
        use_one_hot=args.use_one_hot,
        ct_norm=args.ct_norm,
        remove_bg=args.remove_bg, 
        use_tif=args.use_tif,
        desc=args.desc, 
        max_dim=args.max_dim,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        skip_preprocessing=args.skip_preprocessing,
        no_auto_config=args.no_auto_config,
        logs_dir=args.logs_dir,
        print_param=args.remote,
        debug=args.debug,

        )

#---------------------------------------------------------------------------

