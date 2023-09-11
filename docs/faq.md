# Frequently asked questions

## NaN loss

Can be caused by:
* Not up to data CUDA/CuDNN version (see: https://discuss.pytorch.org/t/half-precision-convolution-cause-nan-in-forward-pass/117358/4).
* Half-precision. Try to set the USE_FP16 parameter to False in the config file.