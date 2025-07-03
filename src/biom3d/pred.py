#---------------------------------------------------------------------------
# Main code: run predictions
#---------------------------------------------------------------------------

import os
import argparse
import pathlib
import numpy as np

from biom3d.builder import Builder
from biom3d.utils import versus_one, dice, DataHandlerFactory
from biom3d.eval import eval

#---------------------------------------------------------------------------
# prediction base fonction

def pred_single(log, img_path,out_path):
    """Prediction on a single image.
    """
    if not isinstance(log,list): log=str(log)
    builder = Builder(config=None,path=log, training=False)
    handler = DataHandlerFactory.get(
            img_path,
            output=out_path,
            img_path = img_path,
            img_outdir = out_path,
        )
    img = builder.run_prediction_single(handler, return_logit=False)

    _,metadata = handler.load(handler.images[0])
    handler.save(handler.images[0], img, metadata)
    return builder.config.NUM_CLASSES+1 # for pred_seg_eval_single

def pred(log, dir_in, dir_out):
    """Prediction on a folder of images.
    """
    if not isinstance(log,list): log=str(log)
    dir_in=str(dir_in)
    dir_out=str(dir_out)

    dir_out = os.path.join(dir_out,os.path.split(log[0] if isinstance(log,list) else log)[-1]) # name the prediction folder with the model folder name
    builder = Builder(config=None,path=log, training=False)
    builder.run_prediction_folder(dir_in=dir_in, dir_out=dir_out, return_logit=False)
    return dir_out

def pred_multiple(log, dir_in, dir_out):
    """Prediction a folder of folders of images.
    """
    list_dir_in = [os.path.join(dir_in, e) for e in os.listdir(dir_in)]
    list_dir_out = [os.path.join(dir_out, e) for e in os.listdir(dir_in)]
    LOG_PATH = log

    for i in range(len(list_dir_in)):
        dir_in = list_dir_in[i]
        dir_out = list_dir_out[i]

        builder = Builder(config=None,path=LOG_PATH, training=False)
        builder.run_prediction_folder(dir_in=dir_in, dir_out=dir_out, return_logit=False)

#---------------------------------------------------------------------------
# main unet segmentation
def pred_seg(log=pathlib.Path.home(), dir_in=pathlib.Path.home(), dir_out=pathlib.Path.home()):
    pred(log, dir_in, dir_out)

def pred_seg_eval(log=pathlib.Path.home(), dir_in=pathlib.Path.home(), dir_out=pathlib.Path.home(), dir_lab=None, eval_only=False):
    print("Start inference")
    builder_pred = Builder(
        config=None,
        path=log, 
        training=False)

    dir_out = os.path.join(dir_out,os.path.split(log[0] if isinstance(log,list) else log)[-1]) # name the prediction folder with the last model folder name
    if not eval_only:
        builder_pred.run_prediction_folder(dir_in=dir_in, dir_out=dir_out, return_logit=False) # run the predictions
    print("Inference done!")


    if dir_lab is not None:
        if isinstance(builder_pred.config,list):
            num_classes = builder_pred.config[0].NUM_CLASSES+1
        else:
            num_classes = builder_pred.config.NUM_CLASSES+1
        # eval
        eval(dir_lab,dir_out,num_classes=num_classes,single_class=None)

def pred_seg_eval_single(log, img_path, out_path, msk_path):
    handler1 = DataHandlerFactory.get(
        dir_lab,
        read_only=True,
        img_path = dir_lab,
    )
    handler2 = DataHandlerFactory.get(
        dir_out,
        read_only=True,
        img_path = dir_out,
    )
    print("Run prediction for:", img_path)
    num_classes = pred_single(log, img_path, out_path)
    print("Done! Prediction saved in:", out_path)
    print("Metric computation with mask:", msk_path)
    dice_score = versus_one(fct=dice, in_path=out_path, tg_path=msk_path, num_classes=num_classes)
    print("Metric result:", dice_score)

#---------------------------------------------------------------------------

if __name__=='__main__':

    # methods names 
    valid_names = {
        "seg": pred_seg,
        "seg_eval": pred_seg_eval,
        "seg_multiple": pred_multiple,
        "seg_single": pred_single,
        "seg_eval_single": pred_seg_eval_single,
        # "seg_patch": pred_seg_patch,
        # "seg_patch_multi": pred_seg_patch_multi,
        # "single": pred_single,
        # "triplet": main_triplet,
        # "arcface": main_arcface,
        # "unet_triplet": main_unet_triplet,
        # "cotrain": main_cotrain,
        # "cotrain_and_single": main_cotrain_and_single
    }

    # parser
    parser = argparse.ArgumentParser(description="Main training file.")
    parser.add_argument("-n", "--name", type=str, default="seg",
        help="Name of the tested method. Valid names: {}".format(valid_names.keys()))
    parser.add_argument("-l", "--log", type=str, nargs='+',
        help="Path of the builder directory/directiories. You can pass several paths to make a prediction using several models.")
    parser.add_argument("-i", "--dir_in", type=str,
        help="Path to the input image directory")
    parser.add_argument("-o", "--dir_out", type=str,
        help="Path to the output prediction directory")
    parser.add_argument("-a", "--dir_lab", type=str, default=None,
        help="Path to the input image directory") 
    parser.add_argument("-e", "--eval_only", default=False,  action='store_true', dest='eval_only',
        help="Do only the evaluation and skip the prediction (predictions must have been done already.)") 
    args = parser.parse_args()

    if isinstance(args.log,list) and len(args.log)==1:
        args.log = args.log[0]

    # run the method
    assert args.name in valid_names.keys(), "[Error] Name of the method must be one of {}".format(valid_names.keys())
    if args.log is None:
        valid_names[args.name].show(run=True)
    else:
        if args.name=="seg_eval":
            valid_names[args.name](args.log, args.dir_in, args.dir_out, args.dir_lab, args.eval_only)
        elif args.name=="seg_eval_single":
            valid_names[args.name](args.log, args.dir_in, args.dir_out, args.dir_lab)
        else:
            valid_names[args.name](args.log, args.dir_in, args.dir_out)

#---------------------------------------------------------------------------