# Biom3d

A simple and unofficial implementation of [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

The goal of biom3d (and of the original nnUNet) is to automatically configured the training of a U-Net deep learning model for 3D semantic segmentation.

This implementation is more flexible for developers than the original nnUNet implementation: easier to read/understand and easier to edit.

This implementation does not include ensemble learning and the possibility to use 2D U-Net or 3D-Cascade U-Net yet. However, these options could easily be adapted to this implementation if needed.

## Installation

Requirements:
* A NVidia GPUs (at least a Geforce GTX 1080)
* Windows 10 or Ubuntu 18.04 (other OS have not been tested)
* Python 3.8 or 3.9 or 3.10 (newer or older version have not been tested)
* CUDA & CuDNN (cf [Nvidia doc](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html))
* (optional but recommended) Use Conda or Pip-env. The later is recommended: use `python3 -m venv env` to create a virtual environment and `source env/bin/activate` to activate it. 
* pytorch==1.12.1 (please be sure that you have access to your NVidia GPU)

Once pytorch installed, the installation can be completed with the following command:

```
pip3 install -r requirements.txt
```

Or with the following:

```
pip3 install SimpleITK==2.1.1 pandas==1.4.0 scikit-image==0.19.0 tensorboard==2.8.0 tqdm==4.62.3 numpy==1.21.2 matplotlib==3.5.3 PyYAML==6.0 torchio==0.18.83 protobuf==3.19.3
```

Optional: If planning to use omero_dowloader to download datasets/projects from omero, please install omero-py with the following command:

```
pip3 install omero-py
```

For Windows users, careful with the previous package: you might need to install [Visual Studio C++ 14.0](https://stackoverflow.com/questions/29846087/error-microsoft-visual-c-14-0-is-required-unable-to-find-vcvarsall-bat) to install `zeroc` dependency.

## Usage

Two options:
* If you have a trained model (you can use one of the publicly available one), you can do [predictions](#prediction) directly.
* If you do not have a trained model, you must train one.

Three steps to train a new model:
* [data conversion to tif format (both images and ground truth masks)](#preprocessing)
* [configuration file definition](#configuration-file-definition)
* [training](#training)

### Preprocessing

Preprocessing consists in transforming the training images and masks to the appropriate format for both training and prediction.

#### Folder structure

The training images and masks must all be placed inside two distinct folders:

    training_folder
    ├── images
    │   ├── image_01.tif
    │   ├── image_02.tif
    │   └── ...
    └── masks
        ├── image_01.tif
        ├── image_01.tif
        └── ...

About the naming, the only constraint is that the images and masks have the exact same name. All the folders can have any name and the folder structure does not matter.

#### Image format

To help formating the images to the correct format, we have written a preprocessing script (preprocess.py). More details are available in [the next section](#helper-function).

Constraints:
- The images and masks must be .tif files. 
- The images and masks must all have 4 dimensions: (channel, height, width, depth).
- Each dimension of each image must be identical to each dimension of the corresponding mask, expect for the channel dimension.
- Images must be stored in float32 format (numpy.float32).
- Masks must be stored in byte format (numpy.byte) or int64 format (numpy.int64 or python int type).
- Masks values must be 0 or 1. Each mask channel represents one type of object. Masks do not have to be 'one-hot' encoded as we use sigmoid activation and not softmax activation. 

Recommandations: (the training might work well without these constraints)
- Images values must be Z-normalized 

#### Helper function

We defined a function in `biom3d/preprocess.py` to help preprocess the images.

Here is an example of how to use it:

```
python biom3d/preprocess.py --img_dir path/to/image/folder --img_out_dir path/to/preprocessed/image/folder --msk_dir path/to/mask/folder --msk_out_dir path/to/preprocessed/mask/folder --auto_config
```

The `--auto_config` option help you complete the configuration file by providing you the ideal patch size, batch size and number of poolings depending of the median size of the dataset images.

### Configuration file definition

All of the hyper-parameters are defined in the configuration file. The configuration files are stored in Python format in the `configs` folder.

There are two types of hyper-parameters in the configuration file: builder parameters and modules parameters. Builder parameters are written as follows: `NAME=value`. Some builder parameters are optional. Here is the exhaustive list of builder parameters:

```python
# Folder where pre-processed images are stored
IMG_DIR = 'data/pancreas/tif_imagesTr_small'

# Folder where pre-processed masks are stored
MSK_DIR = 'data/pancreas/tif_labelsTr_small'

# (optional) path to the .csv file storing "filename,hold_out,fold", where:
# "filename" is the image name,
# "hold_out" is either 0 (training image) or 1 (testing image),
# "fold" (non-negative integer) indicates the k-th fold, 
# by default fold 0 of the training image (hold_out=0) is the validation set.
CSV_DIR = 'data/pancreas/folds_pancreas.csv'

# folder where the training logs will be stored, including:
# - model .pth files (state_dict)
# - image snapshots of model training (only if USE_IMAGE_CLBK is True)
# - logs with this configuration stored in .yaml format and tensorboard logs
LOG_DIR = 'logs/'

# model name
DESC = 'unet_mine-pancreas_21'

#---------------------------------------------------------------------------
# PASTE AUTO-CONFIG RESULT HERE

# batch size
BATCH_SIZE = 2

# patch size passed to the model
PATCH_SIZE = [40,224,224]

# larger patch size used prior rotation augmentation to avoid "empty" corners.
AUG_PATCH_SIZE = [48,263,263]

# median spacing is used only during prediction to normalize the output images
# it is commented here because we did not noticed any improvemet
# MEDIAN_SPACING=[0.79492199, 0.79492199, 2.5]

#---------------------------------------------------------------------------
# training configs

# whether to store also the best model 
SAVE_BEST = True 

# number of epochs
NB_EPOCHS = 1000


LR_START = 1e-2 # comment if need to reload learning rate after training interruption
# LR_MILESTONES = [100, NB_EPOCHS//2, NB_EPOCHS-100]
# LR_T_MAX = NB_EPOCHS
WEIGHT_DECAY = 3e-5

USE_DEEP_SUPERVISION = False
NUM_POOLS = [3,5,5]
NUM_CLASSES=2
USE_SOFTMAX=False 



USE_FP16 = True

NUM_WORKERS = 12

#---------------------------------------------------------------------------
# callback setup

SAVE_MODEL_EVERY_EPOCH = 1
USE_IMAGE_CLBK = True
VAL_EVERY_EPOCH = SAVE_MODEL_EVERY_EPOCH
SAVE_IMAGE_EVERY_EPOCH = SAVE_MODEL_EVERY_EPOCH
USE_FG_CLBK = True

#---------------------------------------------------------------------------

```

### Training

### Prediction



