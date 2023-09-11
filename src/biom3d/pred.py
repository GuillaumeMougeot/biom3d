#---------------------------------------------------------------------------
# Main code: run predictions
#---------------------------------------------------------------------------

import os
import argparse
import pathlib
import numpy as np
# from telegram_send import send
# from magicgui import magicgui

from biom3d.builder import Builder
from biom3d.utils import abs_listdir, versus_one, dice, adaptive_imread, adaptive_imsave

#---------------------------------------------------------------------------
# prediction base fonction

def pred_single(log, img_path, out_path):
    """Prediction on a single image.
    """
    if type(log)!=list: log=str(log)
    builder = Builder(config=None,path=log, training=False)
    img = builder.run_prediction_single(img_path, return_logit=False)

    metadata = adaptive_imread(img_path)[1]
    adaptive_imsave(out_path, img, metadata)
    return builder.config.NUM_CLASSES+1 # for pred_seg_eval_single

def pred(log, dir_in, dir_out):
    """Prediction on a folder of images.
    """
    if type(log)!=list: log=str(log)
    dir_in=str(dir_in)
    dir_out=str(dir_out)

    dir_out = os.path.join(dir_out,os.path.split(log[0] if type(log)==list else log)[-1]) # name the prediction folder with the model folder name
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

# import configs.config_unet as config_unet

# @magicgui(call_button="predict")
def pred_seg(log=pathlib.Path.home(), dir_in=pathlib.Path.home(), dir_out=pathlib.Path.home()):
    pred(log, dir_in, dir_out)

def pred_seg_eval(log=pathlib.Path.home(), dir_in=pathlib.Path.home(), dir_out=pathlib.Path.home(), dir_lab=None, eval_only=False):
    print("Start inference")
    builder_pred = Builder(
        config=None,
        path=log, 
        training=False)

    dir_out = os.path.join(dir_out,os.path.split(log[0] if type(log)==list else log)[-1]) # name the prediction folder with the last model folder name
    if not eval_only:
        builder_pred.run_prediction_folder(dir_in=dir_in, dir_out=dir_out, return_logit=False) # run the predictions
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
            if type(builder_pred.config)==list:
                num_classes = builder_pred.config[0].NUM_CLASSES+1
            else:
                num_classes = builder_pred.config.NUM_CLASSES+1
            results += [versus_one(
                fct=dice, 
                in_path=list_abs[1][idx], 
                tg_path=list_abs[0][idx], 
                # num_classes=2, 
                # single_class=-1,
                num_classes=num_classes, 
                single_class=None,
                )]
            print("Metric result:", results[-1])
        print("Evaluation done! Average result:", np.mean(results))
        # send(messages=["Evaluation done of model {}! Average result: {}".format(dir_out, np.mean(results))])

def pred_seg_eval_single(log, img_path, out_path, msk_path):
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

    if type(args.log)==list and len(args.log)==1:
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