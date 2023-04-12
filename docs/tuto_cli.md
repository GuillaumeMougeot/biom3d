# Use the Command Line Interface

Biom3d has different entry points with which you will be able to interact. We will go from the quickest approach, which will let you run a training in a single line of code, to the more detailed ones, which will let you manually adjust some of the hyper-parameters of Biom3d.

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

In the dataset.json file you should see that, indeed, we have two types of objects in this dataset "pancreas" and "cancer". 

To train a deep learning model on this dataset with biom3d, go inside the folder of your dataset and execute the following command:

```
python -m biom3d.preprocess_train\
 --img_dir Task07_Pancreas/imagesTr\
 --msk_dir Task07_Pancreas/labelsTr\
 --num_classes 2
```

That's it ! Your data will be automatically preprocessed and stored in a folder along the existing ones. Your training pipeline, involving hyper-parameters such as the patch size, will be automatically configured depending on the dataset meta-data, such as the median image shape. And the training should start!

This command should create four new folders: two folders with the same names as your image and label folders with the extension `_out`, a `configs` folder and a `logs` folder. The `_out` folders contain the preprocessed images and labels. The `configs` folder contains a Python script: the configuration file of your training. The `logs` folder contains one folder which contains the saved model and the training logs. 

Once the training finished, you can use your model to predict new images with the following command:

```
python -m biom3d.pred\
 --bui_dir path/to/logs/sub-folder\
 --dir_in Task07_Pancreas/imagesTs\
 --dir_out Task07_Pancreas/predsTs
```

The parameter `--bui_dir` is the path of the sub-folder that has been created in the `logs` folder. It should be something like: `logs/20230412-154857-unet_default`.
