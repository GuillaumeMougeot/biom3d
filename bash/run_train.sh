#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

# train
# python -m biom3d.train --config configs/20230509-181824-segpatch_lung.py
# python -m biom3d.train --log logs/20230501-153638-unet_default
# python -m biom3d.train --config configs/20230524-181439-unet_brain.py
# python -m biom3d.train --log logs/20230908-112859-nucleus_official_fold2
# python -m biom3d.train --config configs/20230831-114941-nucleus_official_fold2.py
# python -m biom3d.train --log logs/20231103-100735-nucleus_official_fold2
# python -m biom3d.train --config configs/20230831-114941-nucleus_official_fold3.py
# python -m biom3d.train --config configs/20230926-104900-triplet_pancreas_exp20.py
# python -m biom3d.train --config configs/20230517-121730-unet_chromo.py
# python -m biom3d.train --log logs/20230605-181034-unet_chromo_48h24-48hL
python -m biom3d.train --config configs/20240319-093546-reims_large_full.py
# python -m biom3d.train --config configs/20240219-100035-reims_full.py

# fine-tuning
# python -m biom3d.train\
#  --log logs/20230522-182916-unet_default\
#  --config configs/20230522-182916-config_default.py

# train and eval
# python -m biom3d.train\
#     --name seg_pred_eval\
#     --config configs/20240217-191923-reims.py\
#     --dir_in data/reims/test_match/img\
#     --dir_out data/reims/test_match/preds\
#     --dir_lab data/reims/test_match/msk


