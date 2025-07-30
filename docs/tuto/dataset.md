# Creating a Dataset
Here we will see how to create a dataset to use in Biom3d and some common mistakes.


## Format
For the moment, only 3 format are supported. 

### Folders
The simpliest way to create a Dataset is to use folders. You will need just a folder where you will place all your images. You can still use subfolder to separate your data if you want. For the instant, Biom3d support the following images formats :
- TIFF
- NIFTY
- NUMPY

#### Structure and naming
Let's say you have 2 datasets and you want to merge them into one (to do a single batch of predictions for example) while still separating them. You can do :
```
Raws
├───Dataset1
│   ├───1.tif
│   ├───2.tif
│   └───...
└───Dataset2
    └───...   
```
And give to Biom3d the folder raw. It will keep the structure in prediction in preprocessing :
```
Raws_preprocess
├───Dataset1
│   ├───1.tif
│   ├───2.tif
│   └───...
└───Dataset2
    └───...   

Or 

Prediction
└───MyModelName
    ├───Dataset1
    │   ├───1.tif
    │   ├───2.tif
    │   └───...
    └───Dataset2
        └───...
```
If you do training, or evaluation, you will need a label (or masks) folder. It work exactly the same way. However the label folder should follow exactly the same structure as raw.
```
Labels
├───Dataset1
│   ├───1.tif <- is the label of Raw/Dataset1/1.tif
│   ├───2.tif <- is the label of Raw/Dataset1/2.tif
│   └───...
└───Dataset2
    └───...   
```

Concerning the naming of the folder/images, you can do whatever you want (as long as label has the same structure as raw).

#### Common mistakes
##### Preprocessing wiht multiple datasets
If you have several datasets and you want to train a different model on each of them, you will preprocess the data. 
```
├───Raw1
│   └───...
├───Raw2
│   └───...
├───Label1
│   └───...
└───Label2
    └───...   
```
However, if you're note careful they may be an error of type `"Different number of masks/foreground and images"`. This is because the output of the different preprocessing are mixing. By default, preprocessing create output folder where Biom3d is started so you will have something like :
```
├───Raw1
│   └───...
├───Raw1_out
│   └───...
├───Raw2
│   └───...
├───Raw2_out
│   └───...
├───Label1
│   └───...
├───Label1_out
│   └───...
├───Label2
│   └───...
├───Label2_out
│   └───...
└───fg_out
    └───...   
```
Noticed something ? Yes you have a preprocessed folder per `raw` and `label` folder, but only one for foreground, which mean that it mix both datasets. And that is the source of the error.
```
A simple way to solve this is to correctly separate your data set.
├───Dataset1
│   ├───Raw1
│   └───Label1
└───Dataset2
    ├───Raw2
    └───Label2  
```
Then to start Biom3d from the `Dataset` folder :
- In command line, simply open terminal directly in folder or cd inside then write the preprocess (and eventually train) command. If you are only doing preprocessing, you can also not change the structure but use `--img_outpath`, `--msk_outpath` and `--fg_outpath`. Note that by default, Biom3d call it's img and mask output the same name as the input with `_out` suffix and the foreground outdir `fg_out`.
- With GUI, before clicking on start locally, select the dataset folder with the folder selector.

##### Evaluation on prediction
A similar issue can happen with predictions and evaluation. When you do a a prediction and give an output folder, the predictions will actually be stored in a subfolder. They will be stored in `MyPredictionFolder/MyModelName`, not directly in `MyPredictionFolder` to avoid overwrite. So if you do an evaluation, give the correct subfolder or you may have an error. If you're using a function that does prediction and evaluation at the same time, it will be done automatically.

### HDF5
You can also use HDF5 to store a dataset (that can be a composition of multiple datasets). For the moment they are not usable in the GUI, only in command line.

#### Structure
Unlike the folder, there is a naming convention, all raw images have to be in a `raw` dataset or group and masks in `label`. 
```
MyDataset.h5
├───raw
│   ├───0
│   ├───1
│   └───...
└───label
    └───...  

Or
MyDataset.h5
├───raw
│   ├───dataset1
│   │   ├───0
│   │   └───...
│   └───dataset2
│       ├───0
│       └───...
└───label
    ├───dataset1
    │   ├───0
    │   └───...
    └───dataset2
        ├───0
        └───...

```
Like the folder, the path for the mask must be the same as the path for image, however they can be in different `.h5`, as long as they have the correct key. The preprocessing will create a `fg` group and a dataset per image with the same path as the mask. For example the mask `/label/dataset1/0` will have the foreground `/fg/dataset1/0/blob`. The blob is a special format for foreground. Prediction are stored behind the key `pred`, the prediction of `/raw/dataset1/0` is in `/pred/dataset1/0`. When you do a prediction and give the output_path, let's say `pred.h5` it will be stored in `pred_MyModelName.h5`, except if you give the same file as the input : `biom3d.pred -i data.h5 -o data.h5 -l MyModel` will add the key prediction in `data.h5`.

#### Commons mistakes
##### Doing multiple preprocessing
Due to some restriction, their are no overwriting, meaning that if you do multiple preprocessing over the same datas, you may encounter an error at the generation of the output file. We recomment either deleting/moving away the output file between preprocessing, or if for training, separating the preprocessing step and the training step and give different output files at each time. Or doing the same thing as for the folder and separate your data between your preprocessing.

*Note : this is possible to set different output files for images, label and foreground in preprocessing, if you only give the `img_outpath` parameter, it will also store labels and foregrounds into.*

##### Evaluation on prediction
This is actually the same problem as for the folder, if you do several prediction with the same model, it will not overwrite but add, so if you do an evaluation after, you will encounter an error stating you don't have the same number of labels and predictions. A simple way to fix it is to specify a different output file for each batch of prediction :
```shell
python -m biom3d.pred -i data.h5 -o pred1.h5 -l MyModel
python -m biom3d.pred -i data.h5 -o pred2.h5 -l MyModel
```

### OMERO
You can also use OMERO dataset, however, contrarly to the preceding format, you have to use different module to use them (we are currently working on this). For prediction, use `omeror_pred`, and training `omero_preprocess_train`, or it is included in the `gui`. 
It will download on your computer the dataset as folders, use them, and eventually upload the created folderss. 

## Extra
You can mix HDF5 and folders :
```shell
python -m biom3d.pred -i data.h5 -o prediction_folder -l MyModel
```
Will use h5 as input but will use folder as output. That is the case for every command including input and output. The format is automatically recognize depending on the path you give (`.h5` for HDF5 and a path to a folder (not necessarily existing for output) for the folder), the sole restriction is that the format must be the same for all input/output :
```shell
# This is valid
python -m biom3d.preprocess --img_path img_folder --msk_path msk_folder --img_outpath data.h5
python -m biom3d.preprocess --img_path img_folder --msk_path msk_folder --img_outpath data.h5 --msk_outpath mask.h5

# This is not 
python -m biom3d.preprocess --img_path img_folder --msk_path msk_folder --img_outpath data.h5 --msk_outpath mask_folder_out
python -m biom3d.preprocess --img_path img_folder --msk_path msk.h5 --img_outpath data.h5 --msk_outpath mask.h5