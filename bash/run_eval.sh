#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

# python -m biom3d.eval\
#  --dir_pred data/msd/Task01_BrainTumour/preds/20230502-143157-unet_default\
#  --dir_lab data/msd/Task01_BrainTumour/labelsTr_test\
#  --num_classes 3

# python -m biom3d.eval\
#  --dir_pred data/msd/Task07_Pancreas/preds/20230523-105736-unet_default\
#  --dir_lab data/msd/Task07_Pancreas/labelsTr_test\
#  --num_classes 2

# python -m biom3d.eval\
#  --dir_pred data/btcv/Testing_small/preds/20230522-182916-unet_default\
#  --dir_lab data/btcv/Testing_small/label\
#  --num_classes 13

python -m biom3d.eval\
 --dir_pred data/nucleus/official/test/preds/20230908-202124-nucleus_official_fold4\
 --dir_lab data/nucleus/official/test/msk\
 --num_classes 1

# python -m biom3d.eval\
#  --dir_pred data/mito/test/pred/20230203-091249-unet_mito\
#  --dir_lab data/mito/test/msk\
#  --num_classes 1