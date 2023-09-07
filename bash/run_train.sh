#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

# python -m biom3d.train --config configs/20230509-181824-segpatch_lung.py
# python -m biom3d.train --log logs/20230501-153638-unet_default
# python -m biom3d.train --config configs/20230524-181439-unet_brain.py
# python -m biom3d.train --log logs/20230524-182512-unet_brain
python -m biom3d.train --config configs/20230831-114941-nucleus_official_fold2.py
python -m biom3d.train --config configs/20230831-114941-nucleus_official_fold3.py
python -m biom3d.train --config configs/20230831-114941-nucleus_official_fold4.py
# python -m biom3d.train --config configs/20230517-121730-unet_chromo.py
# python -m biom3d.train --log logs/20230605-181034-unet_chromo_48h24-48hL

# fine-tuning
# python -m biom3d.train\
#  --log logs/20230522-182916-unet_default\
#  --config configs/20230522-182916-config_default.py


