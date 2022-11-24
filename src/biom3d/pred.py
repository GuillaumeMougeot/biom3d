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
from biom3d.utils import abs_listdir, versus_one, dice

#---------------------------------------------------------------------------
# prediction base fonction

def pred(bui_dir, dir_in, dir_out):
    bui_dir=str(bui_dir)
    dir_in=str(dir_in)
    dir_out=str(dir_out)
    LOG_PATH = bui_dir

    builder = Builder(config=None,path=LOG_PATH, training=False)
    builder.run_prediction_folder(dir_in=dir_in, dir_out=dir_out, return_logit=False)

def pred_multiple(bui_dir, dir_in, dir_out):
    """
    predict a folder of folders
    """
    list_dir_in = [dir_in + "/" + e for e in os.listdir(dir_in)]
    list_dir_out = [dir_out + "/" + e for e in os.listdir(dir_in)]
    LOG_PATH = bui_dir

    for i in range(len(list_dir_in)):
        dir_in = list_dir_in[i]
        dir_out = list_dir_out[i]

        builder = Builder(config=None,path=LOG_PATH, training=False)
        builder.run_prediction_folder(dir_in=dir_in, dir_out=dir_out, return_logit=False)

#---------------------------------------------------------------------------
# main unet segmentation

# import configs.config_unet as config_unet

# @magicgui(call_button="predict")
def pred_seg(bui_dir=pathlib.Path.home(), dir_in=pathlib.Path.home(), dir_out=pathlib.Path.home()):
    pred(bui_dir, dir_in, dir_out)

def pred_seg_eval(bui_dir=pathlib.Path.home(), dir_in=pathlib.Path.home(), dir_out=pathlib.Path.home(), dir_lab=None, eval_only=False):
    print("Start inference")
    builder_pred = Builder(
        config=None,
        path=bui_dir, 
        training=False)

    dir_out = os.path.join(dir_out,os.path.split(bui_dir)[-1]) # name the prediction folder with the model folder name
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
            results += [versus_one(
                fct=dice, 
                in_path=list_abs[1][idx], 
                tg_path=list_abs[0][idx], 
                # num_classes=2, 
                # single_class=-1,
                num_classes=builder_pred.config.NUM_CLASSES if builder_pred.config.USE_SOFTMAX else (builder_pred.config.NUM_CLASSES+1), 
                single_class=None,
                )]
            print("Metric result:", print(results[-1]))
        print("Evaluation done! Average result:", np.mean(results))
        # send(messages=["Evaluation done of model {}! Average result: {}".format(dir_out, np.mean(results))])

#---------------------------------------------------------------------------

if __name__=='__main__':

    # methods names 
    valid_names = {
        "seg": pred_seg,
        "seg_eval": pred_seg_eval,
        "seg_multiple": pred_multiple,
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
    parser.add_argument("-b", "--bui_dir", type=str,
        help="Path of the builder directory")
    parser.add_argument("-i", "--dir_in", type=str,
        help="Path to the input image directory")
    parser.add_argument("-o", "--dir_out", type=str,
        help="Path to the output prediction directory")
    parser.add_argument("-a", "--dir_lab", type=str, default=None,
        help="Path to the input image directory") 
    parser.add_argument("-e", "--eval_only", default=False,  action='store_true', dest='eval_only',
        help="Do only the evaluation and skip the prediction (predictions must have been done already.)") 
    args = parser.parse_args()

    # run the method
    # valid_names[args.name](
    #     bui_dir=args.bui_dir,
    #     dir_in=args.dir_in,
    #     dir_out=args.dir_out,
    # )
    if args.bui_dir is None:
        valid_names[args.name].show(run=True)
    else:
        if args.dir_lab is not None:
            valid_names[args.name](args.bui_dir, args.dir_in, args.dir_out, args.dir_lab, args.eval_only)
        else:
            valid_names[args.name](args.bui_dir, args.dir_in, args.dir_out)

#---------------------------------------------------------------------------