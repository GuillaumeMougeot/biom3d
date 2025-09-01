"""
"All-in-one" command.

Subsequently run the preprocessing and the training.
"""

import argparse
from biom3d.preprocess import auto_config_preprocess
from biom3d.builder import Builder

def preprocess_train(
        img_path:str,
        msk_path:str,
        num_classes:str,
        config_dir:str="configs/",
        base_config:str|None=None,
        ct_norm:bool=False,
        desc:str="unet",
        max_dim:int=128,
        num_epochs:int=1000,
        is_2d:bool=False,
        )->Builder:
    """
    Preprocess images and masks, then launch training with given configuration.

    This function automates preprocessing configuration creation and runs the training process.

    Parameters
    ----------
    img_path : str
        Path to the collection conating images.
    msk_path : str
        Path to the collection conating masks.
    num_classes : int
        Number of classes for segmentation, background not included.
    config_dir : str,default="configs/"
        Directory where preprocessing configurations are saved.
    base_config : str or None, optional
        Path to a base configuration file to start from.
    ct_norm : bool, default=False
        Whether to apply CT normalization during preprocessing.
    desc : str, default="unet"
        Model name.
    max_dim : int, default=128
        Maximum dimension size used in preprocessing.
    num_epochs : int, default=1000
        Number of epochs for training.
    is_2d : bool, default=False
        Whether to treat the input data as 2D slices.

    Returns
    -------
    biom3d.builder.Builder
        The Builder instance that was used to run training, which contains training details and results.
    """
    # preprocessing
    config_path = auto_config_preprocess(
        img_path=img_path, 
        msk_path=msk_path, 
        num_classes=num_classes, 
        config_dir=config_dir, 
        base_config=base_config, 
        ct_norm=ct_norm,
        desc=desc, 
        max_dim=max_dim,
        num_epochs=num_epochs,
        is_2d=is_2d
    )

    # training
    builder = Builder(config=config_path)
    builder.run_training()
    print("Training done!")
    return builder

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Let's do it all-at-once! Subsequent preprocessing and training.")
    parser.add_argument("--img_path","--img_dir",dest="img_path", type=str,required=True,
        help="Path of the images collection")
    parser.add_argument("--msk_path","--msk_dir",dest="msk_path", type=str, default=None,
        help="(default=None) Path to the masks/labels collection")
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
    parser.add_argument("--is_2d", default=False, dest='is_2d',
        help="(default=False) Whether the image is 2d.")

    args = parser.parse_args()

    preprocess_train(
        img_path=args.img_path,
        msk_path=args.msk_path,
        num_classes=args.num_classes,
        config_dir=args.config_dir,
        base_config=args.base_config,
        ct_norm=args.ct_norm,
        desc=args.desc,
        max_dim=args.max_dim,
        num_epochs=args.num_epochs,
        is_2d=args.is_2d,
    )