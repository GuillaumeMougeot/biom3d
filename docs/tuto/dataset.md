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

####s Common mistakes
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
You can also use HDF5 to store a dataset (that can be a composition of multiple datasets). 

#### Structure
Same as folder.
*More documentation on the way*

#### Commons mistakes
*More documentation on the way*

### OMERO
You can also use OMERO dataset, however, contrarly to the preceding format, you have to use different module to use them (we are currently working on this). For prediction, use `omeror_pred`, and training `omero_preprocess_train`, or it is included in the `gui`. 