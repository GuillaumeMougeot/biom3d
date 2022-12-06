#---------------------------------------------------------------------------
# Main code: the run the dataset and the model
#---------------------------------------------------------------------------

import argparse
# from importlib.resources import path
import os
import numpy as np
# from telegram_send import send
from importlib import import_module

from biom3d.builder import Builder
from biom3d.utils import load_config, abs_listdir, versus_one, dice


#---------------------------------------------------------------------------
# utils

def train(config=None, log=None): 
    cfg = None if not config else import_module(config).CONFIG
    builder = Builder(config=cfg,path=log)
    builder.run_training()
    print("Training done!")

def train_yaml(config=None, log=None): 
    cfg = None if not config else load_config(config)
    builder = Builder(config=cfg,path=log)
    builder.run_training()
    print("Training done!")

#---------------------------------------------------------------------------
# main unet segmentation

def main_seg_pred_eval(
    config=None,
    log=None,
    dir_in=None,
    dir_out=None,
    dir_lab=None,
    freeze_encoder=False,
    ):
    """
    do 3 tasks:
    - train the model
    - compute predictions on the test set (dir_in) and store the results in dir_out
    - evaluate the prediction stored in dir_out and print the result
    """
    # train
    print("Start training")
    cfg = import_module(config).CONFIG
    builder_train = Builder(config=cfg,path=log)
    if freeze_encoder:
        builder_train.model.freeze_encoder()
    builder_train.run_training()
    # builder_train.run_training_ddp()

    train_base_dir=builder_train.base_dir
    del builder_train
    print("Training done!")

    # pred
    if dir_in is not None and dir_out is not None:
        print("Start inference")
        builder_pred = Builder(
            config=cfg,
            path=train_base_dir, 
            training=False)

        dir_out = os.path.join(dir_out,os.path.split(train_base_dir)[-1]) # name the prediction folder with the model folder name
        builder_pred.run_prediction_folder(dir_in=dir_in, dir_out=dir_out, return_logit=False)
        print("Inference done!")


        if dir_lab is not None:
            # eval
            print("Start evaluation")
            paths_lab = [dir_lab, dir_out]
            list_abs = [sorted(abs_listdir(p)) for p in paths_lab]
            assert sum([len(t) for t in list_abs])%len(list_abs)==0, "[Error] Not the same number of labels and predictions! {}".format([len(t) for t in list_abs])

            results = []
            for idx in range(len(list_abs[0])):
                print("Metric computation for:", list_abs[1][idx])
                results += [versus_one(
                    fct=dice, 
                    in_path=list_abs[1][idx], 
                    tg_path=list_abs[0][idx], 
                    num_classes=cfg.NUM_CLASSES if cfg.USE_SOFTMAX else (cfg.NUM_CLASSES+1), 
                    single_class=None)]
                print("Metric result:", print(results[-1]))
            print("Evaluation done! Average result:", np.mean(results))
            # send(messages=["Evaluation done of model {}! Average result: {}".format(dir_out, np.mean(results))])
        


#---------------------------------------------------------------------------
# self-supervised training


def main_pretrain_seg_pred_eval(
    pretrain_config=None,
    train_config=None,
    log=None, # TODO
    path_encoder=None,
    freeze_encoder=False,
    model_encoder=False, # if it is a model encoder (UNet) or just an encoder
    dir_in=None,
    dir_out=None,
    dir_lab=None,
    # random_encoder=False,
    ):
    """
    do 4 tasks:
    - pretrain the model/encoder
    - train the model
    - compute predictions on the test set (dir_in) and store the results in dir_out
    - evaluate the prediction stored in dir_out and print the result
    """
    # pretraining
    print("Start pretraining")
    cfg = import_module(pretrain_config).CONFIG
    builder = Builder(config=cfg,path=log)
    if path_encoder is None:
        builder.run_training()
    print("Pretraining is done!")

    # train
    print("Start training")
    cfg = import_module(train_config).CONFIG
    # cfg.MODEL.kwargs.encoder_ckpt = os.path.join(builder.model_dir,builder.config.DESC+"_best.pth")

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
    # builder_train.run_training_ddp()

    train_base_dir=builder_train.base_dir
    del builder_train
    print("Training done!")

    # pred
    if dir_in is not None and dir_out is not None:
        print("Start inference")
        builder_pred = Builder(
            config=cfg,
            path=train_base_dir, 
            training=False)

        dir_out = os.path.join(dir_out,os.path.split(train_base_dir)[-1]) # name the prediction folder with the model folder name
        builder_pred.run_prediction_folder(dir_in=dir_in, dir_out=dir_out, return_logit=False)
        print("Inference done!")


        if dir_lab is not None:
            # eval
            print("Start evaluation")
            paths_lab = [dir_lab, dir_out]
            list_abs = [sorted(abs_listdir(p)) for p in paths_lab]
            assert sum([len(t) for t in list_abs])%len(list_abs)==0, "[Error] Not the same number of labels and predictions! {}".format([len(t) for t in list_abs])

            results = []
            for idx in range(len(list_abs[0])):
                print("Metric computation for:", list_abs[1][idx])
                results += [versus_one(
                    fct=dice, 
                    in_path=list_abs[1][idx], 
                    tg_path=list_abs[0][idx], 
                    num_classes=cfg.NUM_CLASSES if cfg.USE_SOFTMAX else (cfg.NUM_CLASSES+1), 
                    single_class=None)]
                print("Metric result:", print(results[-1]))
            print("Evaluation done! Average result:", np.mean(results))
            # send(messages=["Evaluation done of model {}! Average result: {}".format(dir_out, np.mean(results))])

#---------------------------------------------------------------------------

# import configs.config_unet_monai as config_unet_monai
# import configs.config_genesis as config_genesis
# import configs.config_unet_genesis as config_unet_genesis
# import configs.config_unet_denoiseg as config_denoiseg
# import configs.config_triplet as config_triplet
# import configs.config_arcface as config_arcface
# import configs.config_unet_adverse as config_adverse

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
    parser.add_argument("-c", "--config", type=str, default="configs.unet_pancreas",
        help="Name of the python configuration file (full path and without the .py)")
    parser.add_argument("-pc", "--pretrain_config", type=str, default="configs.vgg-dino_pancreas",
        help="Name of the python configuration file for the pretraining (full path and without the .py)")
    parser.add_argument("--config_yaml", type=str, default=None,
        help="Name of the configuration file stored in yaml format (full path and without the .py)")
    parser.add_argument("-l", "--log", type=str, default=None,
        help="Name of the log folder. Used to resume model training")
    parser.add_argument("-le","--path_encoder", type=str, default=None,
        help="Path for the encoder model (.pth). Used to load the encoder")
    parser.add_argument("-me", "--model_encoder", default=False,  action='store_true', dest='model_encoder',
        help="Whether the encoder is a model encoder or a simple encoder.") 
    parser.add_argument("-fr", "--freeze_encoder", default=False,  action='store_true', dest='freeze_encoder',
        help="Whether to freeze or not the encoder.") 
    parser.add_argument("-i", "--dir_in", type=str, default="/home/gumougeot/all/codes/python/3dnucleus/data/pancreas/imagesTs_small",
        help="Path to the input image directory")
    parser.add_argument("-o", "--dir_out", type=str, default="/mnt/52547A99547A8011/data/preds",
        help="Path to the output prediction directory")  
    parser.add_argument("-a", "--dir_lab", type=str, default="/home/gumougeot/all/codes/python/3dnucleus/data/pancreas/labelsTs_small",
        help="Path to the input image directory")  
    args = parser.parse_args()

    # run the method
    if args.name=="seg_pred_eval":
        valid_names[args.name](
            config=args.config,
            log=args.log,
            dir_in=args.dir_in,
            dir_out=args.dir_out,
            dir_lab=args.dir_lab,
            freeze_encoder=args.freeze_encoder,
            )
    elif args.name=='pretrain_seg_pred_eval':
        valid_names[args.name](
            pretrain_config=args.pretrain_config,
            train_config=args.config,
            log=args.log,
            dir_in=args.dir_in,
            path_encoder=args.path_encoder,
            model_encoder=args.model_encoder,
            freeze_encoder=args.freeze_encoder,
            dir_out=args.dir_out,
            dir_lab=args.dir_lab,
            )
    elif args.config_yaml is not None:
        train_yaml(config=args.config_yaml, log=args.log)
    else:
        train(config=args.config, log=args.log)

#---------------------------------------------------------------------------
