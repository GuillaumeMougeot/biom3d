#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

# python pred.py -n seg -b logs/20220602-225456-unet-retrain-genesis-21 -i data/pancreas/tif_test_img -o data/pancreas/pred_genesis_1

python pred.py\
 --name seg\
 --bui_dir logs/20220909-152755-unet_mine-chromo\
 --dir_in /home/gumougeot/all/codes/python/3dnucleus/data/chromo/to_pred/raw\
 --dir_out /home/gumougeot/all/codes/python/3dnucleus/data/chromo/is_pred

# python pred.py -n seg_patch -b logs/20220705-102047-unet_patch-pancreas -i data/pancreas/imagesTs_small -o data/pancreas/pred/20220705-102047-unet_patch-pancreas

# python pred.py -n seg -b logs/20220721-173648-unet_mine-pancreas_21 -i data/pancreas/imagesTs_small -o /mnt/52547A99547A8011/data/preds/20220721-173648-unet_mine-pancreas_21
# python pred.py -n seg -b logs/20220728-215528-unet_mine-pancreas_21 -i data/pancreas/imagesTs_small -o /mnt/52547A99547A8011/data/preds/20220728-215528-unet_mine-pancreas_21


# python pred.py -n seg_eval\
#  -b logs/20220802-095821-unet_mine-pancreas_21\
#  -i data/pancreas/imagesTs_small\
#  -o /mnt/52547A99547A8011/data/preds/\
#  -a data/pancreas/labelsTs_small

# python pred.py -n seg_eval\
#  --bui_dir logs/20220819-113730-unet_mine-lung\
#  --dir_in data/lung/imagesTs\
#  --dir_out /mnt/52547A99547A8011/data/preds/\
#  --dir_lab data/lung/labelsTs