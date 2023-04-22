#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

python -m biom3d.train --config configs/20230413-unet_btcv.py
# python -m biom3d.train --config configs/20230421-141639-config_default.py