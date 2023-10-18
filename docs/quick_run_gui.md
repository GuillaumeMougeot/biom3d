# The Graphical User Interface

**The new GUI is finally here !**

Starting the Graphical User Interface of biom3d depends on the type of installation you followed:
* If you installed biom3d with the directly link toward the executable file, you can simply double-click on the downloaded file.
* If you installed biom3d with the API or the source code, you can start the GUI with `python -m biom3d.gui`.

> Warning: the version of the GUI on GitHub is remote only, which means that you must also install the API on a server. If you would like to train your deep learning model locally and use the GUI, please install the API. 

## Splash screen

<p align="center">
  <img src="_static/image/gui_splash.PNG" />
</p>

Biom3d comes with 2 modes: local or remote. 'Local' means that the computation will be executed on your computer. 'Remote' means that the computations will be executed on a distant computer where the API of biom3d has been installed. The aspect of the GUI will slightly change depending on the chosen mode.

If you have installed biom3d with the local version simply click on the 'Start locally' button to start, you can choose a path to store your files in the field over the button, by default, the files are stored in the directory where biom3d have been launched.

If you have installed biom3d with the remote version, you must then complete the required fields. The first one is the IP address of your remote computer (where the API of biom3d is installed). The second and third one is your user name and password to connect to the remote computer, the forth one is the path to your virtual environment (if you don't have a virtual environment leave it empty).


## Preprocess & Train

### Local
<p align="center">
  <img src="_static/image/gui_local_preprocess&train.PNG" />
</p>


The preprocessing is executed locally. The current images and masks format that are accepted by the GUI are TIFF files ('.tif' extension) and NIFTI ('.nii.gz' extension).

> Note: The goals of the preprocessing are to standardize the input image and mask formats and to fasten the training process. During the preprocessing the images and masks will be converted to TIFF files ('.tif' extension). Each voxel intensity of the images will be Z-normalized (with a subtraction by the mean intensity and a division by the standard deviation of the intensities in one image). 

:warning: You have **two options** ,you can either **choose a dataset to preprocess and auto-configure** or if you already preprocessed a dataset **load a configuration file**.
#### OPTION 1 : 
#### Dataset selection

Browse through your folders to locate your image folder and mask folder, where your images and masks are stored in TIFF or NIFTI format.

> Warning: Path should not include spaces or special charaters.

#### Training configuration


First choose a name for your model. The model name does not have to be unique because the date of the training will be added automatically to the beginning of the model name. 

Enter then the number of classes in your masks. The number of classes are the number of objects inside your images. For example, if you have annotated in your mask a pancreas with the label 1 and a tumor with label 2, you can entre '2' in this third field'.

Once the preprocessing data fields completed, configure the training hyper-parameters by  pressing the "Auto-configuration" button . The "Auto-configuration" will choose for you the best configuration except the number of epochs which should be define manually, after that you can change some parameters manually if you want. 

> Note: The default value of the number of epochs is 100 but 100 is quite small and should be increased if needed. 

> Note: The rest of the hyper-parameters is automatically set depending on the median size of the 3D images of the dataset. 3D images are often too big to fit into memory when training a deep learning model, so their number and size must be regulated. The default values have be setup for a computer having a GPU of 12Go of VRAM. In the case where you have access to a larger GPU it could be interesting to increase the values of the training configuration. The batch size is a positive integer defining the number of images that will be used passed to the model simultaneously. A batch size of 2 is a good default to allow the model to see simultaneously several images and not too big to prevent any memory problem. The patch size is a triplet of positive integers defining the size of the crop applied to a 3D image. Each patch will be randomly rotated to give to the model different point of view. Unfortunately, the rotation creates black regions in the corner of the image. To avoid this artefact, the augmented patch size defines the size of a slightly bigger patch on which the rotation will be applied before the real patching. The number of pooling in the UNet is the number of time an image patch will be divided by 2. Hence, if one of the pooling dimension is set to 3 then the patch size will be divided by 8 and so the patch size should be dividable by 8! And this is true for all 3 dimensions.
#### OPTION 2 : 
#### Loading a configuration file
<p align="center">
  <img src="_static/image/gui_local_preprocess&train_loadconfig.PNG" />
</p>


If you have already preprocessed your dataset and configured your training parameters you can load the configuration file by clicking on 'Datset is already preprocessed ?' check box, and then select your configuration file.

To load the configuration file click on 'Load config' button. 

#### Start the training!

Start the training by pressing the "Start" button and follow the training process in the terminal. Once the training is finished ("Training done!" will appear below the "Start" button).

### Remote

<p align="center">
  <img src="_static/image/gui_remote_preprocess&train.PNG" />
</p>

For the Remote version of the GUI you can either send a preprocessed folders to your remote computer by choosing a nice and unique name for your new dataset, then send your dataset by pressing the 'Send Dataset' button.

Or you can select a dataset that's already on the server, by using the dropdown menu 'Choose a dataset'. 


The training configuration and the training start are similar to the local version. Follow the above subsection to get more details.


Once the training starts you can display the learning curves by clicking on 'Plot Learning Curves' button.



## Predict

Once your model is trained congratulation you are ready for production! You can now use your model on new raw data with the "Predict" tab. Prediction can also be done from Omero dataset and the results can be stored in Omero. Some details are provided in the third and forth subsections below. 

The general idea behind prediction is: 1. choose a new unannotated dataset 2. choose a trained model 3. start the prediction.

### Local (without Omero)

<p align="center">
  <img src="_static/image/gui_local_predict.PNG" />
</p>

First, select your image data folder with the first "Browse" button. 

Second, select your model folder with the second "Browse" button. The model folder is named "date-time-model_name" (for example "20221005-122923-hrnet_pancreas") and should contain 3 sub-folders ("image","log","model").

Third, select the output directory with the third "Browse" button or click on 'Send predictions to omero' button to store the prediction on Omero (more details in the last section).

Fourth, choose prediction options : Keep the biggest object only or keep big objects only, or none.

Finally, press the "Start" button to start your prediction.

### Remote (without Omero)

<p align="center">
  <img src="_static/image/gui_remote_predict.PNG" />
</p>

First, select your image directory from the drop-down menu or send a new image directory by pressing first the "Browse" button, finding your image directory on your local computer and then pressing the "Send data" button to send your image directory to the remote server.

Second, select one of the model existing on the remote server with the drop-down menu in the "Model selection" frame.

Third, choose prediction options : **Keep the biggest object only** or **Keep big objects only**, or none.

Fourth, press the "Start" button to start the prediction. You can follow the prediction process in the terminal.

Fifth, once prediction are finished you can download them from the drop-down menu in the "Download predictions" frame and choosing a download location on your local computer with the "Browse button". Download then your results with the "Get data" button.

### Local (with Omero)

<p align="center">
  <img src="_static/image/gui_local_predict_omero.PNG" />
</p>

When clicking on the "Use omero" tick box, two new frames should appear and replace the previous "Input directory" frame. In the first frame called "Connection to Omero", set your Omero server, user name and password. In the second frame called "Selection of Omero dataset", you can run the prediction over an Omero Dataset (a folder containing images). In the same frame, then set the identifier (ID) of your dataset. 

<p align="center">
  <img src="_static/image/omero_dataset_id.PNG" />
</p>

The next frames are similar to the one without Omero, please follow the steps starting from the second one in the "Local (without Omero)" sub-section for more details. 

### Remote (with Omero)

<p align="center">
  <img src="_static/image/gui_remote_predict_omero.PNG" />
</p>

Please follow the first step of the "Local (with Omero)" sub-section and then the steps of the "Remote (without Omero)" sub-section starting from the second step to get all the details.

### Send predictions to Omero (Local and Remote)

When clicking on the "Send predictions to omero" tick box, a new frame should appear and replace the previous "Output directory" frame. The frame is called "Connection to Omero server", to set your Omero server, user name and password. In the same frame, you have to choose the output Project ID (where a new dataset will be created) then set a name to that dataset. 

### Local 

<p align="center">
  <img src="_static/image/gui_local_send_to_omero.PNG" />
</p>


For Local Gui click on 'Browse' and select the dataset to send. After that click on 'Start' button.

### Remote

<p align="center">
  <img src="_static/image/gui_remote_send_to_omero.PNG" />
</p>




Finally, for Remote Gui in the drop-down menu choose the dataset to send and click on 'Send to Omero' button, 


