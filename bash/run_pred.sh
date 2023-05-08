#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

python -m biom3d.pred\
 --name seg_eval\
 --bui_dir logs/20230502-143157-unet_default\
 --dir_in data/msd/Task01_BrainTumour/imagesTr_test\
 --dir_out data/msd/Task01_BrainTumour/preds\
 --dir_lab data/msd/Task01_BrainTumour/labelsTr_test

# python -m biom3d.pred\
#  --name seg\
#  --bui_dir logs/20230427-170753-unet_default\
#  --dir_in data/btcv/Testing_official/img\
#  --dir_out data/btcv/Testing_official/preds