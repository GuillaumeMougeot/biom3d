#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

# python -m biom3d.preprocess_train\
#  --img_dir /home/gumougeot/all/codes/python/biom3d/data/pancreas/imagesTs_tiny\
#  --msk_dir /home/gumougeot/all/codes/python/biom3d/data/pancreas/labelsTs_tiny\
#  --num_classes 2

# python -m biom3d.preprocess_train\
#  --img_dir data/msd/Task02_Heart/imagesTr\
#  --msk_dir data/msd/Task02_Heart/labelsTr\
#  --num_classes 1

# python -m biom3d.preprocess_train\
#  --img_dir data/msd/Task06_Lung/imagesTr_train\
#  --msk_dir data/msd/Task06_Lung/labelsTr_train\
#  --num_classes 1\
#  --ct_norm

# python -m biom3d.preprocess_train\
#  --img_dir data/btcv/Training_official/img\
#  --msk_dir data/btcv/Training_official/label\
#  --num_classes 13\
#  --ct_norm

# python -m biom3d.preprocess_train\
#  --img_dir data/btcv/Training_small/img\
#  --msk_dir data/btcv/Training_small/label\
#  --num_classes 13\
#  --ct_norm

# python -m biom3d.preprocess_train\
#  --img_dir data/nucleus/official/train/img\
#  --msk_dir data/nucleus/official/train/msk\
#  --num_classes 1

# python -m biom3d.preprocess_train\
#  --img_dir data/mito/train/img\
#  --msk_dir data/mito/train/msk\
#  --num_classes 1\
#  --desc unet_mito

# python -m biom3d.preprocess_train\
#  --img_dir data/msd/Task01_BrainTumour/imagesTr_train\
#  --msk_dir data/msd/Task01_BrainTumour/labelsTr_train\
#  --num_classes 3\
#  --desc unet_brain

# python -m biom3d.preprocess_train\
#  --img_dir data/nucleus/aline_nucleus_48h24hL/img\
#  --msk_dir data/nucleus/aline_nucleus_48h24hL/msk\
#  --num_classes 1\
#  --desc nucleus_48h24hL

# python -m biom3d.preprocess_train\
#  --img_dir data/nucleus/chromo/img\
#  --msk_dir data/nucleus/chromo/msk\
#  --num_classes 1\
#  --desc nucleus_chromo

# python -m biom3d.preprocess_train\
#  --img_dir data/nucleus/official/train/img\
#  --msk_dir data/nucleus/official/train/msk\
#  --num_classes 1\
#  --desc nucleus_official


# python -m biom3d.preprocess_train\
#  --img_dir data/nucleus/aline_48h72hL/img\
#  --msk_dir data/nucleus/aline_48h72hL/msk_chromo\
#  --num_classes 1\
#  --desc nucleus_48h72hL\
#  --num_epochs 300


# python -m biom3d.preprocess_train\
#  --img_dir data/nucleus/aline_all/img\
#  --msk_dir data/nucleus/aline_all/msk\
#  --num_classes 1\
#  --desc nucleus_aline_all

# python -m biom3d.preprocess_train\
#  --img_dir data/reims/img_stack\
#  --msk_dir data/reims/msk_stack\
#  --num_classes 1\
#  --desc reims

# python -m biom3d.preprocess_train\
#  --img_dir data/reims/large/img_split\
#  --msk_dir data/reims/large/msk_split\
#  --num_classes 1\
#  --desc reims_large

python -m biom3d.preprocess_train\
 --img_dir data/reims/large/img\
 --msk_dir data/reims/large/msk\
 --num_classes 1\
 --desc reims_large_full