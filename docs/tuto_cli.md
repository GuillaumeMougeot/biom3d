# Use the Command Line Interface

Biom3d has different entry points with which you will be able to interact. We will go from the quickest approach, which will let you run a training in a single line of code, to the more detailed ones, which will let you manually adjust some of the hyper-parameters of Biom3d.

> **Warning**: For Windows users, the paths are here written in "linux-like format". You will have to change '/' symbols to '\\' symbols in the command lines. 

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
    └── labelsTr
    │   ├── pancreas_002.nii.gz
    │   ├── pancreas_003.nii.gz
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

This command should create five new folders: two folders with the same names as your image and label folders with the extension `_out`, a `fg_out` folder, a `configs` folder and a `logs` folder. The `_out` folders contain the preprocessed images and labels. The `fg_out` folders contains a list of `.pkl` files with the same name as the images and each containing the foreground locations. The `configs` folder contains a Python script: the configuration file of your training. The `logs` folder contains one folder which contains the saved model and the training logs. 

Once the training is finished, you can use your model to predict new images with the following command:

```
python -m biom3d.pred\
 --log path/to/logs/sub-folder\
 --dir_in Task07_Pancreas/imagesTs\
 --dir_out Task07_Pancreas/predsTs
```

The parameter `--log` is the path of the sub-folder that has been created in the `logs` folder. It should be something like: `logs/20230412-154857-unet_default`.

## Another example with tif files!

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

## Preprocess, train, predict and evaluate

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

Where `logs/20240427-170528-unet_default` is the path of the 

