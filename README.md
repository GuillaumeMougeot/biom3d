# biom3d

A simple and unofficial implementation of [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

The goal of nnUNet is to automatically configured the training of a U-Net deep learning model for 3D semantic segmentation.

This implementation is more flexible for developers than the original nnUNet implementation: easier to read/understand and easier to edit.

This implementation does not include ensemble learning and the possibility to use 2D U-Net or 3D-Cascade U-Net. However, these options could easily be adapted for this implementation if needed.

## Installation

## Usage

Three steps:
* [data conversion to tif format (both images and masks)](#preprocessing)
* config file definition
* run training 

### Preprocessing

Preprocessing consists in transforming the training images and masks to the appropriate format.

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

Constraints:
- The images and masks must be .tif files. 
- The images and masks must all have 4 dimensions: (channel, height, width, depth).
- Images must be stored in float32 format (numpy.float32)
- Masks must be stored in byte format (numpy.byte)
- Masks values must be 0 or 1. (Masks do not have to be 'one-hot' encoded)

Recommandations: (the training might work well without these constraints)
- Images values must be Z-normalized 

#### Helper function




