#!/bin/sh

python -m biom3d.preprocess\
 --img_dir data/btcv/Training_small/img\
 --msk_dir data/btcv/Training_small/label\
 --num_classes 13\
 --ct_norm