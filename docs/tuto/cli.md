# The Command Line Interface

Biom3d has different entry points with which you will be able to interact. We will go from the quickest approach, which will let you run a training in a single line of code, to the more detailed ones, which will let you manually adjust some of the hyper-parameters of Biom3d. 

You can find examples of command lines [in the bash folder on the GitHub repo of Biom3d](https://github.com/GuillaumeMougeot/biom3d/tree/main/bash) starting with `run_`. 

> **Warning**: For Windows users, the paths are here written in "linux-like format". You will have to change '/' symbols to '``\``' symbols in the command lines. 

## The quickest way

Let's say you would like to train a deep learning model with on one of the tasks of the [Medical Segmentation Decathlon](http://medicaldecathlon.com/). For example, Task07 goal is to segment the pancreas and its tumor in CT-scans. If you download the data folder of Task07, you should see the following data structure:

    Task07_Pancreas
    ├── imagesTr
    │   ├── pancreas_001.nii.gz
    │   ├── pancreas_004.nii.gz
    │   └── ...
    ├── imagesTs
    │   ├── pancreas_002.nii.gz
    │   ├── pancreas_003.nii.gz
    │   └── ...
    ├── labelsTr
    │   ├── pancreas_001.nii.gz
    │   ├── pancreas_004.nii.gz
    │   └── ...
    └── dataset.json

In the dataset.json file you should see that, we have two types of objects in this dataset "pancreas" and "cancer" and that the images are CT-scans.

To train a deep learning model on this dataset with biom3d, go inside the folder of your dataset and execute the following command:

```
python -m biom3d.preprocess_train\
 --img_dir Task07_Pancreas/imagesTr\
 --msk_dir Task07_Pancreas/labelsTr\
 --num_classes 2\
 --ct_norm
```

That's it! Your data will be automatically preprocessed and stored in a folder along the existing ones. Your training pipeline, involving hyper-parameters such as the patch size, will be automatically configured depending on the dataset meta-data, such as the median image shape. And the training should start!

The `--ct_norm` option is optional and should be used only with CT-scan images.

This command should create five new folders: two folders with the same names as your image and label folders with the extension `_out`, a `fg_out` folder, a `configs` folder and a `logs` folder. The `_out` folders contain the preprocessed images and labels. The `fg_out` folders contains a list of `.pkl` files with the same name as the images and each containing the foreground locations. The `configs` folder contains a Python script: the configuration file of your training. The `logs` folder contains one folder which contains the saved model and the training logs. The resulting folder architecture should look like this:

    main_dir
    ├── Task07_Pancreas
    │   ├── fg_out
    │   │   ├── pancreas_001.pkl
    │   │   ├── pancreas_004.pkl
    │   │   └── ...
    │   ├── imagesTr
    │   │   ├── pancreas_001.nii.gz
    │   │   ├── pancreas_004.nii.gz
    │   │   └── ...
    │   ├── imagesTr_out
    │   │   ├── pancreas_001.npy
    │   │   ├── pancreas_004.npy
    │   │   └── ...
    │   ├── imagesTs
    │   │   ├── pancreas_002.nii.gz
    │   │   ├── pancreas_003.nii.gz
    │   │   └── ...
    │   ├── labelsTr
    │   │   ├── pancreas_001.nii.gz
    │   │   ├── pancreas_004.nii.gz
    │   │   └── ...
    │   ├── labelsTr_out
    │   │   ├── pancreas_001.npy
    │   │   ├── pancreas_004.npy
    │   │   └── ...
    │   └── dataset.json
    ├── configs
    │   └── 20240427-170528-config_default.py
    └── logs
        └── 20240427-170528-unet_default
            ├── image
            ├── log
            └── model

Once the training is finished, you can use your model to predict new images with the following command:

```
python -m biom3d.pred\
 --log logs\20240427-170528-unet_default\
 --dir_in Task07_Pancreas/imagesTs\
 --dir_out Task07_Pancreas/predsTs
```

The parameter `--log` is the path of the sub-folder that has been created in the `logs` folder. It should be something like: `logs/20240427-170528-unet_default`.

## Train with your own images: another example with tif files

Let's say you want now to train a model with tif files. No worries we got you! The previous command still works!

For example, let's say that you have a folder structure that looks like this:

    training_folder
    ├── images
    │   ├── image_01.tif
    │   ├── image_02.tif
    │   └── ...
    └── masks
        ├── image_01.tif
        ├── image_01.tif
        └── ...

Let's suppose that you have only one type of object in your masks. To train a new model, simply execute the following command:

```
python -m biom3d.preprocess_train\
 --img_dir training_folder/images\
 --msk_dir training_folder/masks\
 --num_classes 1\
 --desc my_new_model
```

The optional `--desc` option here is used to change the name of the configuration file and the model name. It is just to make it look nice.

That's it! The preprocessing and training should start.

> **Warning**: About the file/folder naming, the only constraint is that the images and masks have the exact same name. All the folders can have any name with **no space** in it and the parent folder structure does not matter. 

> **Warning**: Constraints on image format:
> * The images and masks must be .tif files or .nii.gz file. If using another format then install biom3d from source and edit `biom3d.utils.adaptive_imread` and `biom3d.utils.adaptive_imsave`... or preprocess your images to have the proper format.
> * The images and masks must all have 3 or 4 dimensions: (height, width, depth) or (channel, height, width, depth).
> * Each dimension of each image must be identical to each dimension of the corresponding mask, expect for the channel dimension.
> * Masks values must be either 0 or 1 if stored in (channel, height, width, depth) format, or must be in 0,1,...,N where N is the number of classes.

Prediction can be run with the aforementioned command (cf. section above).

## Preprocess first, train after

Let's now suppose that we would like to decompose the preprocessing from the training. This could be particularly useful when debugging...

To start a preprocessing only with the previous example, here is the command:

```
python -m biom3d.preprocess\
 --img_dir training_folder/images\
 --msk_dir training_folder/masks\
 --num_classes 1\
 --desc another_example
```

The only difference is that we used `biom3d.preprocess` instead of `biom3d.preprocess_train`. The command should create 4 folders: two folders with the same names as your image and label folders with the extension `_out`, a `fg_out` folder and a `configs` folder. If the `configs` folder already exists, it will simply add a new configuration file in it. 

Now to start the training you can run the following command:

```
python -m biom3d.train\
 --config configs/20240427-170528-config_default.py
```

Where `configs/20240427-170528-config_default.py` is the path to your configuration file that must have been newly created into your `config` folder. The training should start.

If for any reason, the training has been interrupted, you can restart it form the latest checkpoint with the following command. 

```
python -m biom3d.train\
 --log logs/20240427-170528-unet_default
```

Where `logs/20240427-170528-unet_default` is the path of the log folder containing the model, images and curves. 

Prediction can be run with the aforementioned command (cf. first section above).

## Evaluate

Once some predictions have been made, Biom3d can let you evaluate your trained model on a test set. The folder architecture of the test set should look like this:

    evaluation_folder
    ├── masks
    │   ├── image_01.tif
    │   ├── image_02.tif
    │   └── ...
    └── predictions
        ├── image_01.tif
        ├── image_01.tif
        └── ...

You can now evaluate your trained model with the following command:

```
python -m biom3d.eval\
 --dir_pred evaluation_folder/predictions\
 --dir_lab evaluation_folder/masks\
 --num_classes 1
```

This should print the Dice score of each prediction and the average one.

Additionally, you can sequentially run the prediction first and then the evaluation with only one command:

```
python -m biom3d.pred\
 --name seg_eval\
 --log logs/20230510-181401-unet_default\
 --dir_in data/msd/Task06_Lung/imagesTr_test\
 --dir_out data/msd/Task06_Lung/preds\
 --dir_lab data/msd/Task06_Lung/labelsTr_test
```

## Omero prediction

For Omero users, you can use the following command to make a prediction on one of your Omero dataset or Omero project:

```
python -m biom3d.omero_pred\
 --obj Dataset:ID\
 --log logs/20240427-170528-unet_default\
 --target folder/where/omero/images/are/downloaded\
 --dir_out folder/where/predictions/will/be/stored\
 --username your_username\
 --password your_password\
 --hostname your_hostname\
 --upload_id your_project_id
```

Please complete each of the field above. The Omero dataset ID can be found in your Omero browser here:

<p align="center">
  <img src="_static/image/omero_dataset_id.PNG" />
</p>

The flag `--upload_id` is optional and can be use to upload the prediction results back into Omero. 

## Even more

For even more details on the Command Line Interface, please check our inline documentation with one of the following commands:

```
python -m biom3d.preprocess_train --help
```

```
python -m biom3d.preprocess --help
```

```
python -m biom3d.train --help
```

```
python -m biom3d.pred --help
```

```
python -m biom3d.eval --help
```

```
python -m biom3d.omero_pred --help
```
