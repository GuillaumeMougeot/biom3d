# Frequently asked questions

## "FileNotFoundError: [Errno 2] No such file or directory" or "FileNotFoundError: [Errno 2] No such file or directory"

Usually appears when image files in image folder and mask files in mask folder do not have the **exact** same name and extension. Please make sur that both your images and masks have all the same names and extensions.

## NaN loss

Can be caused by:
* Not up to data CUDA/CuDNN version (see: https://discuss.pytorch.org/t/half-precision-convolution-cause-nan-in-forward-pass/117358/4).
* Half-precision. Try to set the USE_FP16 parameter to False in the config file.