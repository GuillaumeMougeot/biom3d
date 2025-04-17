# Frequently asked questions

## "FileNotFoundError: [Errno 2] No such file or directory" or "FileNotFoundError: [Errno 2] No such file or directory"

Usually appears when image files in image folder and mask files in mask folder do not have the **exact** same name and extension. Please make sur that both your images and masks have all the same names and extensions.

## NaN loss

Can be caused by:
* Not up to data CUDA/CuDNN version (see: https://discuss.pytorch.org/t/half-precision-convolution-cause-nan-in-forward-pass/117358/4).
* Half-precision. Try to set the USE_FP16 parameter to False in the config file.

## ValueError : Images don't have the same shape :

Happen when images you want to train or predict on don't all have the same dimensions. Can be cause by opening the image with Napari and transfering it to Fiji. You can either reimport the raw images or remove the problematics ones.

## [Error] Invalid image shape (x,_, _, _). Expected to have 1 numbers of channel at 0 channel axis.
Can be caused by:
* One of your image hasn't the number of channel the model has been trained on, you can fix it by removing problematic image.
* The model didn't registered the number of channel it work with. If you are sure the problem isn't from the dataset (see above) you can add to the config.yaml (in log) the line "num_channels: x" to the kwarg part of the PREPROCESSOR part :
```yaml
    PREPROCESSOR:
        fct: Seg
        kwargs:
            median_spacing: *id004
            clipping_bounds: *id005
            intensity_moments: *id006
            channel_axis: 0
            num_channels: 1 <- (for example)
```