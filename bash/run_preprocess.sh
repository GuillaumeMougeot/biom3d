#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

python -m biom3d.preprocess\
 --img_dir data/btcv/Training_official/img\
 --msk_dir data/btcv/Training_official/label\
 --num_classes 13\
 --ct_norm

# python -m biom3d.preprocess\
#  --img_dir data/msd/Task02_Heart/imagesTr\
#  --msk_dir data/msd/Task02_Heart/labelsTr\
#  --num_classes 1\
#  --ct_norm