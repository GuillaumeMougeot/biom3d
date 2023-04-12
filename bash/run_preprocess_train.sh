#!/bin/sh

python -m biom3d.preprocess_train\
 --img_dir /home/gumougeot/all/codes/python/biom3d/data/pancreas/imagesTs_tiny\
 --msk_dir /home/gumougeot/all/codes/python/biom3d/data/pancreas/labelsTs_tiny\
 --num_classes 2