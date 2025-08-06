# Pre processing 
Here we will explain how the pre processing work and how to modify it. Some steps only apply on masks.

## Preprocessing function
There are several steps, with some optionnals :
1. **(Image only) Dimension standardisation :** First it will change images to  `(C,D,H,W)` for 3D images and `(C,1,H,W)` for 2D images. The missing dimensions are added, if your image already have a channel dimension be cautious that it is the smaller dimension as Biom3d consider the smallest dimension as the smallest. For example `(D=8,C=11,H=512,W=512)` would cause a problem.
2. **(Mask only) Corrections :**  If the masks is in 2D, it will add new dimension. Then it will determine what kind of encoding your masks are in :
   1. If the masks still has 3 dimension, it use `label` encoding. It collect the unique values of your voxels and assure there are in the given number of classes. If you have more uniques values that classes (eg `[0,1,2,3]` for 3 classes), it will crash. If you have the same number of values and classes but wrong values (eg `[1,2,8]`for 3 classes) it will rearrange them to `[0,1,2,...,number of classes]`. If there are only 2 classes, it will treat them the same way as `binary` encoding. We then add the channel dimension for standardisation.
   2. If the masks has 4 dimension, it use `binary` encoding, so each voxel has a `0` or `1` for each channel. The strategy to fix potential error is to get all values and consider the most numerous value to be the background and the rest to be our channel.
   ```
   1,1,1            0,0,0 
   1,0,2  Will give 0,1,1 With here 0 as background and 1 as class.
   1,1,0            0,0,1   
   ```
   3. The `onehot` encoding isn't detected and must be forced (see below). This is a particular case of `binary` where each voxel has always only one channel at `1`. There is no correction with this encoding, it will throw an error if this is not respected.
   
   For the moment, the only way to force an encoding type is to call directly the function with specific parameter.

   Then in the case of not a `label` encoding and with a specific parameter, we go back to the initial number of dimension. This is not the default behaviour of this function.
3. **(Image only) Intensity transformation :** The value are clipped between the percentiles 0.5% and 99.5% (computed on the whole dataset), only if parameter `--ct_norm` is given to the module. Then we apply a zero mean unit variance normalization, if `--ct_norm` was used, we use mean and standard deviation of the whole dataset, else of the image.
4. **Spatial resampling :** In the case where spacings are given in the metadata of the image, the median spacing of the dataset and image spacing are used to resample the image so they all have the same spacing (being the median spacing).
5. **(Mask only) :** For each class, we get a maximum of 10,000 voxels of the class and store them in an array. Those array are used for training.

That are the preprocessing done by Biom3d on each image, however, Biom3d does other things during preprocessing.

## Data fingerprint
This is a `auto_config` function used before each preprocessing that collect and compute data from the whole dataset before doing the actual preprocessing. It will do :
1. **Spacing :** For each image, it will read the metadata to store the spacing (if spacing is in metadata), at the same time, it ensure that all images have the same number of dimension.
2. **(Mask only) Sampling :** If a path to the masks is given, it will also for each mask get a random sample of values on the image associated with the mask where the mask define this is not background.
3. **Fingerpring computation :** We use the samples to compute the `mean`, `standard deviation`, `percentile 0.5%` and `percentile 99.5%`. Those value are used by the preprocessing.

You can retrieve those information in the `config.yaml` file of your model :
```yaml
MEDIAN_SPACING: &id004 [] <- Empty because dataset didn't have metadata
CLIPPING_BOUNDS: &id005 
- 0.0041880556382238865 <- Percentile 0.5%
- 0.9984244108200073 <- Percentile 99.5%
INTENSITY_MOMENTS: &id006
- 0.4780538082122803 <- Mean
- 0.2981347143650055 <- Standard deviation
```

## Image splitting
In case you are working with a dataset of only one image, this image will be split in two for training purpose (not in the predictions). We slice the image on the biggest dimension with a ratio of 25% validation and 75% training. 

## Folds
The preprocessing generate a file named `folds.csv`. For example (we added spaces for lisibility ):
```
filename,     hold_out,  fold
img/img1.tif, 0,         0
img/img2.tif, 0,         0
img/img3.tif, 0,         1
img/img4.tif, 0,         1
img/img5.tif, 0,         2
img/img6.tif, 0,         2
img/img7.tif, 1,         0
img/img8.tif, 1,         1
```
Let's break it down :
- Filename is the image path (relative to where you starded the command from). Every images of your dataset are represented in this file.
- Hold_out can have 2 values : `0` and `1`. The value `1` means that this image will only be used at the end of the training to evaluate model performance, `0` means that it can be use anywhere.
- Fold : A fold is a subset of your dataset. During training, we use different groups of folds validation to mix the data and avoid overtraining. We can use folds 1 and 2 for a first validation and 0 and 1 for another.

This file is not necessary for training (fold are computed if needed), but it better to use one for reproductibility sake.