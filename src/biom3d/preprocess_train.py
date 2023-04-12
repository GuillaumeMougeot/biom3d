#---------------------------------------------------------------------------
# "All-in-one" command!
# Subsequently run the preprocessing and the training.
#---------------------------------------------------------------------------

import argparse
from biom3d.preprocess import Preprocessing
from biom3d.auto_config import auto_config, save_auto_config
from biom3d.utils import load_python_config
from biom3d.builder import Builder

def preprocess_train(img_dir, msk_dir, num_classes, config_dir):
    # preprocessing
    p=Preprocessing(
        img_dir=img_dir,
        msk_dir=msk_dir,
        num_classes=num_classes+1,
        remove_bg=False,
        use_tif=False,
    )
    p.run()

    # auto-config
    batch, aug_patch, patch, pool = auto_config(img_dir=p.img_dir)

    # save auto-config
    config_path = save_auto_config(
        config_dir=config_dir,
        img_dir=p.img_outdir,
        msk_dir=p.msk_outdir,
        num_classes=num_classes,
        batch_size=batch,
        aug_patch_size=aug_patch,
        patch_size=patch,
        num_pools=pool
    )

    # training
    cfg = load_python_config(config_path)
    builder = Builder(config=cfg,path=None)
    builder.run_training()
    print("Training done!")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Let's do it all-at-once! Subsequent preprocessing and training.")
    parser.add_argument("--img_dir", type=str,
        help="Path of the images directory")
    parser.add_argument("--msk_dir", type=str, default=None,
        help="(default=None) Path to the masks/labels directory")
    parser.add_argument("--num_classes", type=int, default=1,
        help="(default=1) Number of classes (types of objects) in the dataset. The background is not included.")
    parser.add_argument("--config_dir", type=str, default='configs/',
        help="(default=\'configs/\') Configuration folder to save the auto-configuration.")
    args = parser.parse_args()

    preprocess_train(
        img_dir=args.img_dir,
        msk_dir=args.msk_dir,
        num_classes=args.num_classes,
        config_dir=args.config_dir
    )