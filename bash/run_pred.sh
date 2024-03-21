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
#  --log logs/20231003-herve_cynthia\
#  --dir_in data/herve/img\
#  --dir_out data/herve/preds

# python -m biom3d.pred\
#  --name seg_eval\
#  --log logs/20230908-202124-nucleus_official_fold4 logs/20230908-060422-nucleus_official_fold3 logs/20230907-155249-nucleus_official_fold2 logs/20230907-002954-nucleus_official_fold1 logs/20230906-101825-nucleus_official_fold0\
#  --dir_in data/nucleus/official/test/img\
#  --dir_out data/nucleus/official/test/preds\
#  --dir_lab data/nucleus/official/test/msk

# python -m biom3d.pred\
#  --name seg_eval\
#  --log logs/20230927-110033-exp1_supervised_baseline_pancreas_fold0\
#  --dir_in data/pancreas/imagesTs_small\
#  --dir_out data/pancreas/preds\
#  --dir_lab data/pancreas/labelsTs_small

# nucleus
# python -m biom3d.pred\
#  --name seg\
#  --log logs/20230914-015946-chromo_dry_fold0 logs/20230830-120829-chromo_48h24hL_fold0 logs/20230831-211753-chromo_48h72hL_fold0\
#  --dir_in data/nucleus/to_pred/raw_selection_24h_cot\
#  --dir_out data/nucleus/preds/raw_selection_24h_cot

# python -m biom3d.pred\
#  --name seg\
#  --log logs/20230914-015946-chromo_dry_fold0 logs/20230830-120829-chromo_48h24hL_fold0 logs/20230831-211753-chromo_48h72hL_fold0\
#  --dir_in "data/nucleus/to_pred/raw_selection_48h_cot A-C-D-E"\
#  --dir_out "data/nucleus/preds/raw_selection_48h_cot A-C-D-E"

# python -m biom3d.pred\
#  --name seg\
#  --log logs/20230914-015946-chromo_dry_fold0 logs/20230830-120829-chromo_48h24hL_fold0 logs/20230831-211753-chromo_48h72hL_fold0\
#  --dir_in data/nucleus/to_pred/raw_selection_48h24hL_cot\
#  --dir_out data/nucleus/preds/raw_selection_48h24hL_cot

# python -m biom3d.pred\
#  --name seg\
#  --log logs/20230914-015946-chromo_dry_fold0 logs/20230830-120829-chromo_48h24hL_fold0 logs/20230831-211753-chromo_48h72hL_fold0\
#  --dir_in data/nucleus/to_pred/raw_selection_48h48hL_cot\
#  --dir_out data/nucleus/preds/raw_selection_48h48hL_cot

# python -m biom3d.pred\
#  --name seg\
#  --log logs/20230914-015946-chromo_dry_fold0 logs/20230830-120829-chromo_48h24hL_fold0 logs/20230831-211753-chromo_48h72hL_fold0\
#  --dir_in "data/nucleus/to_pred/raw_selection_48h72hL_cot A-B-C"\
#  --dir_out "data/nucleus/preds/raw_selection_48h72hL_cot A-B-C"

# python -m biom3d.pred\
#  --name seg\
#  --log logs/20230914-015946-chromo_dry_fold0 logs/20230830-120829-chromo_48h24hL_fold0 logs/20230831-211753-chromo_48h72hL_fold0\
#  --dir_in "data/nucleus/to_pred/raw_selection_dry_cot A-C-D-E"\
#  --dir_out "data/nucleus/preds/raw_selection_dry_cot A-C-D-E"

# droso-herve
# python -m biom3d.pred\
#  --name seg\
#  --log logs/20231128-104500-cynthia\
#  --dir_in "data/herve/cynthia"\
#  --dir_out "data/herve/preds"

# nucleus aline bug fix
# python -m biom3d.pred\
#  --name seg\
#  --log logs/20231127-165609-nucleus_official_fold0_fine-tuned_for_ubp5_fold0\
#  --dir_in "data/nucleus/aline_bug/img"\
#  --dir_out "data/nucleus/aline_bug/preds"

python -m biom3d.pred\
    --name seg\
    --log logs/20240219-100225-reims_full_fold0\
    --dir_in data/reims/big_stack/img\
    --dir_out data/reims/big_stack/preds\

# python -m biom3d.pred\
#     --name seg_eval\
#     --log logs/20240218-072550-reims_fold0\
#     --dir_in data/reims/test_match/img\
#     --dir_out data/reims/test_match/preds\
#     --dir_lab data/reims/test_match/msk

# python -m biom3d.pred\
#     --name seg_single\
#     --log logs/20240218-072550-reims_fold0\
#     --dir_in data/reims/big_stack/img/1024.tif\
#     --dir_out data/reims/big_stack/preds/1024.tif
