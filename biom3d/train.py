#---------------------------------------------------------------------------
# Main code: the run the dataset and the model
#---------------------------------------------------------------------------

import argparse
# from importlib.resources import path
import os
import numpy as np
# from telegram_send import send
from importlib import import_module

from builder import Builder
# from utils import Dict 
import utils

#---------------------------------------------------------------------------
# utils

def train(config=None, log=None): 
    cfg = import_module(config).CONFIG
    builder = Builder(config=cfg,path=log)
    builder.run_training()

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
            list_abs = [sorted(utils.abs_listdir(p)) for p in paths_lab]
            assert sum([len(t) for t in list_abs])%len(list_abs)==0, "[Error] Not the same number of labels and predictions! {}".format([len(t) for t in list_abs])

            results = []
            for idx in range(len(list_abs[0])):
                print("Metric computation for:", list_abs[1][idx])
                results += [utils.versus_one(
                    fct=utils.dice, 
                    in_path=list_abs[1][idx], 
                    tg_path=list_abs[0][idx], 
                    num_classes=cfg.NUM_CLASSES if cfg.USE_SOFTMAX else (cfg.NUM_CLASSES+1), 
                    single_class=None)]
                print("Metric result:", print(results[-1]))
            print("Evaluation done! Average result:", np.mean(results))
            # send(messages=["Evaluation done of model {}! Average result: {}".format(dir_out, np.mean(results))])
        
#---------------------------------------------------------------------------

if __name__=='__main__':

    # methods names 
    valid_names = {
        "train":                  train,
        "seg_pred_eval":          main_seg_pred_eval,
    }

    # parser
    parser = argparse.ArgumentParser(description="Main training file.")
    parser.add_argument("-n", "--name", type=str, default="train",
        help="Name of the tested method. Valid names: {}".format(valid_names.keys()))
    parser.add_argument("-c", "--config", type=str, default="configs.unet_pancreas",
        help="Name of the python configuration file (full path and without the .py)")
    parser.add_argument("-l", "--log", type=str, default=None,
        help="Name of the log folder. Used to resume model training")
    parser.add_argument("-fr", "--freeze_encoder", default=False,  action='store_true', dest='freeze_encoder',
        help="Whether to freeze or not the encoder.") 
    parser.add_argument("-i", "--dir_in", type=str, default="",
        help="Path to the input image directory")
    parser.add_argument("-o", "--dir_out", type=str, default="",
        help="Path to the output prediction directory")  
    parser.add_argument("-a", "--dir_lab", type=str, default="",
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
    else:
        valid_names[args.name](config=args.config, log=args.log)

#---------------------------------------------------------------------------
