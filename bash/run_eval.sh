#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

# python -m biom3d.eval\
#  --dir_pred data/msd/Task01_BrainTumour/preds/20230502-143157-unet_default\
#  --dir_lab data/msd/Task01_BrainTumour/labelsTr_test\
#  --num_classes 3

# python -m biom3d.eval\
#  --dir_pred data/msd/Task07_Pancreas/preds/nnunet/preds\
#  --dir_lab data/msd/Task07_Pancreas/labelsTr_test\
#  --num_classes 2

python -m biom3d.eval\
 --dir_pred data/nucleus/official/test/preds/nnunet\
 --dir_lab data/nucleus/official/test/msk\
 --num_classes 1