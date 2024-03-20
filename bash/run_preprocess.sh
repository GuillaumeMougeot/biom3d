#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

# python -m biom3d.preprocess\
#  --img_dir data/btcv/Training_official/img\
#  --msk_dir data/btcv/Training_official/label\
#  --num_classes 13\
#  --max_dim 224\
#  --desc unet_btcv\
#  --ct_norm\
#  --skip_preprocessing

# python -m biom3d.preprocess\
#  --img_dir data/msd/Task01_BrainTumour/imagesTr_train\
#  --msk_dir data/msd/Task01_BrainTumour/labelsTr_train\
#  --num_classes 3\
#  --desc unet_brain\
#  --skip_preprocessing

# python -m biom3d.preprocess\
#  --img_dir data/msd/Task09_Spleen/imagesTr\
#  --msk_dir data/msd/Task09_Spleen/labelsTr\
#  --num_classes 1\
#  --max_dim 160\
#  --desc unet_spleen\
#  --ct_norm

# python -m biom3d.preprocess\
#  --img_dir data/nucleus/aline_nucleus_48h24hL/img\
#  --msk_dir data/nucleus/aline_nucleus_48h24hL/msk_chromo\
#  --num_classes 1\
#  --desc chromo_48h24hL\
#  --debug

# python -m biom3d.preprocess\
#  --img_dir data/msd/Task06_Lung/imagesTr_train\
#  --msk_dir data/msd/Task06_Lung/labelsTr_train\
#  --num_classes 1\
#  --desc unet_lung\
#  --ct_norm

# python -m biom3d.preprocess\
#  --img_dir data/msd/Task05_Prostate/imagesTr\
#  --msk_dir data/msd/Task05_Prostate/labelsTr\
#  --num_classes 2\
#  --max_dim 128\
#  --desc unet_prostate

# python -m biom3d.preprocess\
#  --img_dir data/nucleus/official/train/img\
#  --msk_dir data/nucleus/official/train/msk\
#  --num_classes 1\
#  --desc nucleus_official\
#  --use_tif

python -m biom3d.preprocess\
 --img_dir data/reims/large/img\
 --msk_dir data/reims/large/msk\
 --num_classes 1\
 --desc reims_large
