#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

python -m biom3d.preprocess\
 --img_dir data/btcv/Training_official/img\
 --msk_dir data/btcv/Training_official/label\
 --num_classes 13\
 --max_dim 224\
 --desc unet_btcv\
 --ct_norm\
 --skip_preprocessing

# python -m biom3d.preprocess\
#  --img_dir data/msd/Task01_BrainTumour/imagesTr_train\
#  --msk_dir data/msd/Task01_BrainTumour/labelsTr_train\
#  --num_classes 3\
#  --skip_preprocessing

# python -m biom3d.preprocess\
#  --img_dir data/nucleus/aline_nucleus_48h24hL/img\
#  --msk_dir data/nucleus/aline_nucleus_48h24hL/msk_chromo\
#  --num_classes 1

# python -m biom3d.preprocess\
#  --img_dir data/msd/Task06_Lung/imagesTr_train\
#  --msk_dir data/msd/Task06_Lung/labelsTr_train\
#  --num_classes 1\
#  --desc unet_lung\
#  --ct_norm