#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

python -m biom3d.pred\
 --name seg_eval\
 --bui_dir logs/20230425-162133-unet_btcv\
 --dir_in data/btcv/Testing_small/img\
 --dir_out data/btcv/Testing_small/preds\
 --dir_lab data/btcv/Testing_small/label