#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

# python -m biom3d.pred\
#  --name seg_eval\
#  --log logs/20230531-092023-unet_lung\
#  --dir_in data/msd/Task06_Lung/imagesTr_test\
#  --dir_out data/msd/Task06_Lung/preds\
#  --dir_lab data/msd/Task06_Lung/labelsTr_test

# python -m biom3d.pred\
#  --name seg_eval\
#  --log logs/20230524-182512-unet_brain\
#  --dir_in data/msd/Task01_BrainTumour/imagesTr_test\
#  --dir_out data/msd/Task01_BrainTumour/preds\
#  --dir_lab data/msd/Task01_BrainTumour/labelsTr_test

# python -m biom3d.pred\
#  --name seg\
#  --log logs/20230427-170753-unet_default\
#  --dir_in data/btcv/Testing_official/img\
#  --dir_out data/btcv/Testing_official/preds

# python -m biom3d.pred\
#  --name seg_eval\
#  --log logs/20230522-182916-unet_default logs/20230425-162133-unet_btcv\
#  --dir_in data/btcv/Testing_small/img\
#  --dir_out data/btcv/Testing_small/preds\
#  --dir_lab data/btcv/Testing_small/label

# python -m biom3d.pred\
#  --name seg_eval_single\
#  --log logs/20230514-182230-unet_lung\
#  --dir_in data/msd/Task06_Lung/imagesTr_test/lung_051.nii.gz\
#  --dir_out data/msd/Task06_Lung/preds/20230514-182230-unet_lung/lung_051.nii.gz\
#  --dir_lab data/msd/Task06_Lung/labelsTr_test/lung_051.nii.gz

# python -m biom3d.pred\
#  --name seg\
#  --log logs/20230518-092836-unet_chromo_48h24-48hL\
#  --dir_in data/nucleus/to_pred/tiny_data\
#  --dir_out data/nucleus/preds/tiny_data

# python -m biom3d.pred\
#  --name seg\
#  --log logs/20230127-144947-unet_nadine\
#  --dir_in data/to_pred\
#  --dir_out data/pred

python -m biom3d.pred\
 --name seg_eval\
 --log logs/20230907-002954-nucleus_official_fold1 logs/20230906-101825-nucleus_official_fold0\
 --dir_in data/nucleus/official/test/img\
 --dir_out data/nucleus/official/test/preds\
 --dir_lab data/nucleus/official/test/msk