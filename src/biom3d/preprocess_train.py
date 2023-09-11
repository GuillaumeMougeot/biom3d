#---------------------------------------------------------------------------
# "All-in-one" command!
# Subsequently run the preprocessing and the training.
#---------------------------------------------------------------------------

import argparse
import os 

from biom3d.preprocess import Preprocessing, auto_config_preprocess
from biom3d.auto_config import auto_config, data_fingerprint
from biom3d.utils import load_python_config, save_python_config
from biom3d.builder import Builder

def preprocess_train(
        img_dir,
        msk_dir,
        num_classes,
        config_dir="configs/",
        base_config=None,
        ct_norm=False,
        desc="unet",
        max_dim=128,
        num_epochs=1000,
        ):
    # preprocessing
    config_path = auto_config_preprocess(
        img_dir=img_dir, 
        msk_dir=msk_dir, 
        num_classes=num_classes, 
        config_dir=config_dir, 
        base_config=base_config, 
        ct_norm=ct_norm,
        desc=desc, 
        max_dim=max_dim,
        num_epochs=num_epochs,
    )

    # training
    builder = Builder(config=config_path)
    builder.run_training()
    print("Training done!")
    return builder

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Let's do it all-at-once! Subsequent preprocessing and training.")
    parser.add_argument("--img_dir", type=str,
        help="Path of the images directory")
    parser.add_argument("--msk_dir", type=str, default=None,
        help="(default=None) Path to the masks/labels directory")
    parser.add_argument("--num_classes", type=int, default=1,
        help="(default=1) Number of classes (types of objects) in the dataset. The background is not included.")
    parser.add_argument("--max_dim", type=int, default=128,
        help="(default=128) max_dim^3 determines the maximum size of patch for auto-config.")
    parser.add_argument("--num_epochs", type=int, default=1000,
        help="(default=1000) Number of epochs for the training.")
    parser.add_argument("--config_dir", type=str, default='configs/',
        help="(default=\'configs/\') Configuration folder to save the auto-configuration.")
    parser.add_argument("--base_config", type=str, default=None,
        help="(default=None) Optional. Path to an existing configuration file which will be updated with the preprocessed values.")
    parser.add_argument("--desc", type=str, default='unet_default',
        help="(default=unet_default) Optional. A name used to describe the model.")
    parser.add_argument("--ct_norm", default=False,  action='store_true', dest='ct_norm',
        help="(default=False) Whether to use CT-Scan normalization routine (cf. nnUNet).") 
    args = parser.parse_args()

    preprocess_train(
        img_dir=args.img_dir,
        msk_dir=args.msk_dir,
        num_classes=args.num_classes,
        config_dir=args.config_dir,
        base_config=args.base_config,
        ct_norm=args.ct_norm,
        desc=args.desc,
        max_dim=args.max_dim,
        num_epochs=args.num_epochs,
    )