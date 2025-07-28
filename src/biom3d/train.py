#---------------------------------------------------------------------------
# Main code: the run the dataset and the model
#---------------------------------------------------------------------------

import argparse
import os
import numpy as np
from biom3d.builder import Builder
from biom3d.utils import load_python_config
from biom3d.eval import eval


#---------------------------------------------------------------------------
# utils

def train(config=None, path=None): 
    builder = Builder(config=config, path=path)
    builder.run_training()
    print("Training done!")
    return builder

#---------------------------------------------------------------------------
# main unet segmentation

def main_seg_pred_eval(
    config_path=None,
    path=None,
    path_in=None,
    path_out=None,
    path_lab=None,
    freeze_encoder=False,
    ):
    """
    do 3 tasks:
    - train the model
    - compute predictions on the test set (path_in) and store the results in path_out
    - evaluate the prediction stored in path_out and print the result
    """
    # train
    print("Start training")
    builder_train = Builder(config=config_path,path=path)
    if freeze_encoder:
        builder_train.model.freeze_encoder()
    builder_train.run_training()

    del builder_train

    # pred
    if path_in is not None and path_out is not None:
        print("Start inference")
        builder_pred = Builder(
            config=config_path,
            path=path,
            training=False)

        out = builder_pred.run_prediction_folder(path_in=path_in, path_out=path_out, return_logit=False)
        print("Inference done!")

        if path_lab is not None:
            # eval
            print("Start evaluation")
            eval(path_lab,out,builder_pred.config.NUM_CLASSES+1)        


#---------------------------------------------------------------------------
# self-supervised training


def main_pretrain_seg_pred_eval(
    pretrain_config=None,
    train_config=None,
    log=None, # TODO
    path_encoder=None,
    freeze_encoder=False,
    model_encoder=False, # if it is a model encoder (UNet) or just an encoder
    path_in=None,
    path_out=None,
    path_lab=None,
    ):
    """
    do 4 tasks:
    - pretrain the model/encoder
    - train the model
    - compute predictions on the test set (path_in) and store the results in path_out
    - evaluate the prediction stored in path_out and print the result
    """
    # pretraining
    print("Start pretraining")
    builder = Builder(config=pretrain_config,path=log)
    if path_encoder is None:
        builder.run_training()
    print("Pretraining is done!")

    # train
    print("Start training")
    cfg = load_python_config(train_config)

    if path_encoder and not model_encoder:
        cfg.MODEL.kwargs.encoder_ckpt = path_encoder
    elif path_encoder and model_encoder:
        cfg.MODEL.kwargs.model_ckpt = path_encoder
    elif not model_encoder and os.path.exists(os.path.join(builder.model_dir,builder.config.DESC+"_best.pth")):
        cfg.MODEL.kwargs.encoder_ckpt = os.path.join(builder.model_dir,builder.config.DESC+"_best.pth")
    elif model_encoder and os.path.exists(os.path.join(builder.model_dir,builder.config.DESC+"_best.pth")):
        cfg.MODEL.kwargs.model_ckpt = os.path.join(builder.model_dir,builder.config.DESC+"_best.pth")
    builder_train = Builder(config=cfg,path=None)
    if freeze_encoder:
        builder_train.model.freeze_encoder()
    builder_train.run_training()

    train_base_dir=builder_train.base_dir
    del builder_train
    print("Training done!")

    # pred
    if path_in is not None and path_out is not None:
        print("Start inference")
        builder_pred = Builder(
            config=cfg,
            path=train_base_dir, 
            training=False)

        out = builder_pred.run_prediction_folder(path_in=path_in, path_out=path_out, return_logit=False)
        print("Inference done!")


        if path_lab is not None:
            # eval
            print("Start evaluation")
            eval(path_lab,out,builder_pred.config.NUM_CLASSES+1) 

#---------------------------------------------------------------------------

if __name__=='__main__':

    # methods names 
    valid_names = {
        "train":                  train,
        "seg_pred_eval":          main_seg_pred_eval,
        "pretrain_seg_pred_eval": main_pretrain_seg_pred_eval,
        # "seg_monai":            lambda log: _main(cfg=config_unet_monai, log=log),
        # "seg_patch":            main_seg_patch,
        # "single":               main_single,
        # "genesis":              lambda log: _main(cfg=config_genesis, log=log),
        # "unet_genesis":         lambda log: _main(cfg=config_unet_genesis, log=log),
        # "denoiseg":             lambda log: _main(cfg=config_denoiseg, log=log),
        # "triplet":              lambda log: _main(cfg=config_triplet, log=log),
        # "arcface":              lambda log: _main(cfg=config_arcface, log=log),
        # "adverse":              lambda log: _main(cfg=config_adverse, log=log),
        # "unet_triplet":         main_unet_triplet,
        # "cotrain":              main_cotrain,
        # "cotrain_and_single":   main_cotrain_and_single,
    }

    # parser
    parser = argparse.ArgumentParser(description="Main training file.")
    parser.add_argument("-n", "--name", type=str, default="train",
        help="Name of the tested method. Valid names: {}".format(valid_names.keys()))
    parser.add_argument("-c", "--config", type=str, default=None,
        help="Name of the python or yaml configuration file")
    parser.add_argument("-pc", "--pretrain_config", type=str, default=None,
        help="Name of the python or yaml configuration file for the pretraining")
    parser.add_argument("-l", "--log", type=str, default=None,
        help="Name of the log folder. Used to resume model training")
    parser.add_argument("-le","--path_encoder", type=str, default=None,
        help="Path for the encoder model (.pth). Used to load the encoder")
    parser.add_argument("-me", "--model_encoder", default=False,  action='store_true', dest='model_encoder',
        help="Whether the encoder is a model encoder or a simple encoder.") 
    parser.add_argument("-fr", "--freeze_encoder", default=False,  action='store_true', dest='freeze_encoder',
        help="Whether to freeze or not the encoder.") 
    parser.add_argument("-i", "--path_in","--dir_in",dest="path_in", type=str, default=None,
        help="Path to the input image collection")
    parser.add_argument("-o", "--path_out","--dir_out",dest="path_out", type=str, default=None,
        help="Path to the output prediction collection")  
    parser.add_argument("-a", "--path_lab","--dir_lab",dest="path_lab", type=str, default=None,
        help="Path to the label image collection")  
    args = parser.parse_args()

    # run the method
    if args.name=="seg_pred_eval":
        valid_names[args.name](
            config_path=args.config,
            path=args.log,
            path_in=args.path_in,
            path_out=args.path_out,
            path_lab=args.path_lab,
            freeze_encoder=args.freeze_encoder,
            )
    elif args.name=='pretrain_seg_pred_eval':
        valid_names[args.name](
            pretrain_config=args.pretrain_config,
            train_config=args.config,
            log=args.log,
            path_in=args.path_in,
            path_encoder=args.path_encoder,
            model_encoder=args.model_encoder,
            freeze_encoder=args.freeze_encoder,
            path_out=args.path_out,
            path_lab=args.path_lab,
            )
    else:
        train(config=args.config, path=args.log)

#---------------------------------------------------------------------------
