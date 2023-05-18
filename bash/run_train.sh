#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

# python -m biom3d.train --config configs/20230413-unet_btcv_sempatch.py
# python -m biom3d.train --log logs/20230501-153638-unet_default
python -m biom3d.train --config configs/20230509-181824-segpatch_lung_nnunet.py
# python -m biom3d.train --config configs/20230515-182606-unet_nucleus.py
# python -m biom3d.train --config configs/20230517-121730-unet_chromo.py
