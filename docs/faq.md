# Frequently asked questions

## "FileNotFoundError: [Errno 2] No such file or directory" or "FileNotFoundError: [Errno 2] No such file or directory"

Usually appears when image files in image folder and mask files in mask folder do not have the **exact** same name and extension. Please make sur that both your images and masks have all the same names and extensions.

## OMERO
When using OMERO prediction, you can ecounter two issues :
### recv() returned zero
You may encounter this :
```
Traceback (most recent call last):
  File "/usr/lib/python3.11/tkinter/__init__.py", line 1948, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/biom3d/gui.py", line 1747, in predict
    biom3d.omero_uploader.run(username=self.send_to_omero_connection.username.get(),
  File "/usr/local/lib/python3.11/dist-packages/biom3d/omero_uploader.py", line 179, in run
    if project and not conn.getObject('Project', project):
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/omero/gateway/__init__.py", line 3300, in getObject
    result = self.getQueryService().findByQuery(
             ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/omero/gateway/__init__.py", line 2596, in getQueryService
    return self._proxies['query']
           ~~~~~~~~~~~~~^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/omero/gateway/__init__.py", line 1489, in __getitem__
    raise Ice.ConnectionLostException
Ice.ConnectionLostException: Ice.ConnectionLostException:
recv() returned zero
```
It means OMERO connection fail, it can be caused by :
* Wrong username/password
* Just a simple connection error

The simpliest way to solve it is by reentering your credentials and try again. If it persist it may be a connection with the server error, if you can't acces your OMERO server with other means (Webclient, insight,...), contact your IT support.

### Uploading
Using OMERO uploading can be quite long. For the moment, there isn't much feedbacks to indicate that upload is finished (we are working on it). On GUI, the `Start` button is white while it is running and become red when it is finished. Also, in the terminal, you see somethong like this :
```
Importing: Test_dataset_predictions/Unit/2.tif
Uploading: Test_dataset_predictions/Unit/2.tif
Hashes:
  80d2a040ed8058338ea0c162033b57d8703ba4fd

Imported Image ID: 301018
```
each time an image is uploaded, you can keep track of uploading with that. However there are cases when nothing is shown in terminal and it is still uploaded. If you are not using the GUI (and can't see the button feedback), we recommand checking in real time the destination folder in OMERO.

## NaN loss

Can be caused by:
* Not up to data CUDA/CuDNN version (see: https://discuss.pytorch.org/t/half-precision-convolution-cause-nan-in-forward-pass/117358/4).
* Half-precision. Try to set the USE_FP16 parameter to False in the config file.

## ValueError : Images don't have the same number of dimensions :

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