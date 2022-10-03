#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

# run train pancreas dataset:
# python train.py\
#  --name seg_pred_eval\
#  --config configs.unet_pancreas\
#  --dir_in /home/gumougeot/all/codes/python/3dnucleus/data/pancreas/imagesTs_small\
#  --dir_out /mnt/52547A99547A8011/data/preds\
#  --dir_lab /home/gumougeot/all/codes/python/3dnucleus/data/pancreas/labelsTs_small

# run train lung dataset:
# python train.py\
#  --name seg_pred_eval\
#  --config configs.unet_lung\
#  --dir_in /home/gumougeot/all/codes/python/3dnucleus/data/lung/imagesTs\
#  --dir_lab /home/gumougeot/all/codes/python/3dnucleus/data/lung/labelsTs
#   --log /home/gumougeot/all/codes/python/3dnucleus/logs/20220819-113730-unet_mine-lung

# run train chromocenter dataset:
# python train.py\
#  --name train\
#  --config configs.unet_chromo

# run train triplet:
python biome3d/train.py\
 --name train\
 --config configs.nnet-triplet_pancreas

# run train DINO:
# python train.py --config configs.vgg-dino_pancreas

# python train.py\
#  --name seg_pred_eval\
#  --config configs.unet_pancreas\
#  --dir_in /home/gumougeot/all/codes/python/3dnucleus/data/pancreas/imagesTs_small\
#  --dir_out /mnt/52547A99547A8011/data/preds\
#  --dir_lab /home/gumougeot/all/codes/python/3dnucleus/data/pancreas/labelsTs_small

# python train.py\
#  --name pretrain_seg_pred_eval\
#  --pretrain_config configs.vgg-triplet_pancreas\
#  --config configs.unet_pancreas\
#  --freeze_encoder\
#  --path_encoder /home/gumougeot/all/codes/python/3dnucleus/logs/20220908-151317-vgg-triplet_pancreas/model/vgg-triplet_pancreas_best.pth\
#  --dir_in /home/gumougeot/all/codes/python/3dnucleus/data/pancreas/imagesTs_small\
#  --dir_out /mnt/52547A99547A8011/data/preds\
#  --dir_lab /home/gumougeot/all/codes/python/3dnucleus/data/pancreas/labelsTs_small
