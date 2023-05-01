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
#  --img_dir data/btcv/Training_official/img\
#  --msk_dir data/btcv/Training_official/label\
#  --num_classes 13\
#  --ct_norm

# python -m biom3d.preprocess_train\
#  --img_dir data/nucleus/nucleus_tiny/img\
#  --msk_dir data/nucleus/nucleus_tiny/msk\
#  --num_classes 1

python -m biom3d.preprocess_train\
 --img_dir data/mito/train/img\
 --msk_dir data/mito/train/msk\
 --num_classes 1