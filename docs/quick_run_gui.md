# Quick run with Graphical User Interface

Starting the Graphical User Interface of biom3d depends on the type of installation you followed:
* If you installed biom3d with the directly link toward the executable file, you can simply double-click on the downloaded file.
* If you installed biom3d with the API or the source code, you can start the GUI with `python -m biom3d.gui`.

## Splash screen

<p align="center">
  <img src="_static/image/gui_splash.PNG" />
</p>

Biom3d comes with 2 modes: local or remote. 'Local' means that the computation will be executed on your computer. 'Remote' means that the computations will be executed on a distant computer where the API of biom3d has been installed. The aspect of the GUI will slightly change depending on the chosen mode.

If you have installed biom3d with the local version simply click on the 'Start locally' button to start.

If you have installed biom3d with the remote version, you must then complete the required fields. The first one is the IP address of your remote computer (where the API of biom3d is installed). The second and third one is your user name and password to connect to the remote computer. (Please ignore the forth one, it is deprecated).

## Preprocessing 

The preprocessing is executed locally independently of the choice to start locally or remotely. The current images and masks format that are accepted by the GUI are TIFF files ('.tif' extension) and NIFTI ('.nii.gz' extension).

> Note: The goals of the preprocessing are to standardize the input image and mask formats and to fasten the training process. During the preprocessing the images and masks will be converted to TIFF files ('.tif' extension). Each voxel intensity of the images will be Z-normalized (with a subtraction by the mean intensity and a division by the standard deviation of the intensities in one image). 

### Local

<p align="center">
  <img src="_static/image/gui_local_preprocess.PNG" />
</p>

Browse through your folders to locate your image folder and mask folder, where your images and masks are stored in TIFF or NIFTI format.

Enter then the number of classes in your masks. The number of classes are the number of objects inside your images. For example, if you have annotated in your mask a pancreas with the label 1 and a tumor with label 2, you can entre '2' in this third field'.

The forth and fifth fields are optional. They indicate in which folder the preprocessed images and masks will be stored. By default, the preprocessed images and masks will be stored along the original images and masks folders.

Press then the 'Start' button to start the preprocessing. You can see in the terminal opened with biom3d if any error message appears.

### Remote

<p align="center">
  <img src="_static/image/gui_remote_preprocess.PNG" />
</p>

The 5 first fields corresponds are similar to the local version of the GUI. Follow the previous section for more details.

Once the preprocessing is done, you can send the preprocessed folders to your remote computer. First chose a nice and unique name for your new dataset and complete the last field. Then send your dataset by pressing the 'Send data to remote server' button.

## Train

### Local