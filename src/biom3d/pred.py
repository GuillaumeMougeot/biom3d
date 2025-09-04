"""
Main module for predictions.

This module contains generic predictions functions:

- pred_single
- pred
- pred multiple

And interface predictions functions made for CLI:

- pred_seg
- pred_seg_eval
- pred_seg_eval_single

"""

import os
import argparse
import pathlib
from typing import Optional
from biom3d.builder import Builder
from biom3d.utils import deprecated, versus_one, dice, DataHandlerFactory
from biom3d.eval import eval

#---------------------------------------------------------------------------
# prediction base fonction
def pred_single(log:str|list[str], 
                img_path:str,
                out_path:str,
                skip_preprocessing:bool=False,
                )->tuple[int,str]:
    """
    Predict segmentation or classification on a single image.

    Parameters:
    -----------
    log : str or list of str
        Path to the model/log directory or configuration.
    img_path : str
        Path to the input image file.
    out_path : str
        Directory where the prediction output will be saved.
    skip_preprocessing : bool, default=False
        If True, skips preprocessing step.

    Returns:
    --------
    num_classes: int
        Number of classes + 1 (the background)
    path_out: str
        Path to the saved mask output.
    """
    if not isinstance(log,list): log=str(log)
    builder = Builder(config=None,path=log, training=False)
    handler = DataHandlerFactory.get(
            img_path,
            output=out_path,
            msk_outpath = out_path,
            model_name = builder.config[-1].DESC if isinstance(builder.config,list) else builder.config.DESC,
        )
    img = builder.run_prediction_single(handler, return_logit=False,skip_preprocessing=skip_preprocessing)
    handler.save(handler.images[0], img,"pred")
    return builder.config.NUM_CLASSES+1,handler.msk_outpath  # for pred_seg_eval_single

def pred(log:str|list[str], 
         path_in:str, 
         path_out:str,
         skip_preprocessing:bool=False,
         )->str:
    """
    Predict on all images in a collecion.

    Parameters:
    -----------
    log : str or list of string
        Path to the model/log directory or configuration.
    path_in : str
        Path to collection containing input images.
    path_out : str
        Path to collection to save prediction outputs.
    skip_preprocessing : bool, default=False
        If True, skips preprocessing step.

    Returns:
    --------
    str
        Path to the output directory containing predictions.
    """
    if not isinstance(log,list): log=str(log)
    path_in=str(path_in)
    path_out=str(path_out)

    path_out = os.path.join(path_out,os.path.split(log[0] if isinstance(log,list) else log)[-1]) # name the prediction folder with the last model folder name

    builder = Builder(config=None,path=log, training=False)
    path_out = builder.run_prediction_folder(path_in=path_in, path_out=path_out, return_logit=False,skip_preprocessing=skip_preprocessing)
    return path_out

@deprecated("This method is no longer used as it is the default behaviour of DataHandlers.")
def pred_multiple(log:str|list[str], 
         path_in:str, 
         path_out:str,
         skip_preprocessing:bool=False,
         )->str:
    """
    Predict on multiple folders of images. DEPRECATED.

    This method is deprecated because the default behavior of DataHandlers 
    now supports multiple folder prediction.

    Parameters:
    -----------
    Same as pred()

    Returns:
    --------
    Same as pred()
    """
    return pred(log,path_in,path_out,skip_preprocessing=skip_preprocessing)

#---------------------------------------------------------------------------
# main unet segmentation interface
def pred_seg(log:pathlib.Path|str|list[str]=pathlib.Path.home(), 
             path_in:pathlib.Path | str =pathlib.Path.home(), 
             path_out:pathlib.Path | str =pathlib.Path.home(),
             skip_preprocessing:bool=False
             )->None:
    """
    Run prediction on a folder of images using default paths.

    Parameters:
    -----------
    log : pathlib.Path, str or list of str, default=home directory
        Path to the model or log directory.
    path_in : pathlib.Path or str, default=home directory
        Path to collection containing images.
    path_out : pathlib.Path or str, default=home directory
        Path to collection where predictions will be saved.
    skip_preprocessing : bool, default=False
        If True, skips preprocessing step.

    Returns
    -------
    None
    """
    pred(log, path_in, path_out,skip_preprocessing=skip_preprocessing)

# TODO remove eval only, we have a module for that
def pred_seg_eval(log:pathlib.Path|str|list[str]=pathlib.Path.home(),
                  path_in:pathlib.Path | str =pathlib.Path.home(), 
                  path_out:pathlib.Path | str =pathlib.Path.home(), 
                  path_lab:Optional[pathlib.Path | str]=None, 
                  eval_only:bool=False,
                  skip_preprocessing:bool=False
                  )->None:
    """
    Run prediction on a folder of images and optionally evaluate segmentation (with dice).

    Parameters:
    -----------
    log : pathlib.Path, str or list of str, default=home directory
        Path to the model or log directory.
    path_in : pathlib.Path or str, default=home directory
        Path to collection containing images.
    path_out : pathlib.Path or str, default=home directory
        Path to collection where predictions will be saved.
    path_lab : pathlib.Path or str, optional
        Path to collection containing ground-truth label masks for evaluation.
    eval_only : bool, default=False
        If True, skips prediction and runs evaluation only.
    skip_preprocessing : bool, default=False
        If True, skips preprocessing step.

    Returns
    -------
    None
    """
    print("Start inference")
    builder_pred = Builder(
        config=None,
        path=log, 
        training=False)

    path_out = os.path.join(path_out,os.path.split(log[0] if isinstance(log,list) else log)[-1]) # name the prediction folder with the last model folder name

    if not eval_only:
        path_out = builder_pred.run_prediction_folder(path_in=path_in, path_out=path_out, return_logit=False,skip_preprocessing=skip_preprocessing) # run the predictions
    print("Inference done!")


    if path_lab is not None:
        if isinstance(builder_pred.config,list):
            num_classes = builder_pred.config[0].NUM_CLASSES
        else:
            num_classes = builder_pred.config.NUM_CLASSES
        # eval
        eval(path_lab,path_out,num_classes=num_classes)

def pred_seg_eval_single(log:str|list[str], 
                         img_path:str, 
                         out_path:str, 
                         msk_path:str,
                         skip_preprocessing:bool=False
                         )->None:
    """
    Run prediction on a single image and compute evaluation metric (dice) against mask.

    Parameters:
    -----------
    log : str or list of str
        Path to the model or log directory.
    img_path : str
        Path to the input image file.
    out_path : str
        Directory where prediction output will be saved.
    msk_path : str
        Path to the ground-truth mask for evaluation.
    skip_preprocessing : bool, default=False
        If True, skips preprocessing step.

    Returns:
    --------
    None
    """
    print("Run prediction for:", img_path)
    num_classes,out = pred_single(log, img_path, out_path,skip_preprocessing=skip_preprocessing)
    print("Done! Prediction saved in:", out)
    handler1 = DataHandlerFactory.get(
        out,
        read_only=True,
        eval="pred",
    )
    handler2 = DataHandlerFactory.get(
        msk_path,
        read_only=True,
        eval="label",
    )
    print("Metric computation with mask:", msk_path)
    dice_score = versus_one(fct=dice, input_img=handler1.load(handler1.images[0])[0], target_img=handler2.load(handler2.images[0])[0], num_classes=num_classes)
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
    parser.add_argument("-l", "--log", type=str, nargs='+',required=True,
        help="Path of the builder directory/directiories. You can pass several paths to make a prediction using several models.")
    parser.add_argument("-i", "--path_in","--dir_in",dest="path_in",type=str,required=True,
        help="Path to the input image collection")
    parser.add_argument("-o", "--path_out","--dir_out",dest="path_out", type=str,required=True,
        help="Path to the output prediction collection")
    parser.add_argument("-a", "--path_lab","--dir_lab",dest="path_lab", type=str, default=None,
        help="Path to the input label collection") 
    parser.add_argument("-e", "--eval_only", default=False,  action='store_true', dest='eval_only',
        help="Do only the evaluation and skip the prediction (predictions must have been done already.)") 
    parser.add_argument("--skip_preprocessing", default=False, action='store_true',dest="skip_preprocessing",
        help="(default=False) Skip preprocessing, it assume the preprocessing has already be done and can crash otherwise")
    args = parser.parse_args()

    if isinstance(args.log,list) and len(args.log)==1:
        args.log = args.log[0]

    # run the method
    assert args.name in valid_names.keys(), "[Error] Name of the method must be one of {}".format(valid_names.keys())
    if args.log is None:
        valid_names[args.name].show(run=True)
    else:
        if args.name=="seg_eval":
            valid_names[args.name](args.log, 
                                   args.path_in, 
                                   args.path_out, 
                                   args.path_lab, 
                                   args.eval_only,
                                   skip_preprocessing=args.skip_preprocessing)
        elif args.name=="seg_eval_single":
            valid_names[args.name](args.log, 
                                   args.path_in, 
                                   args.path_out, 
                                   args.path_lab,
                                   skip_preprocessing=args.skip_preprocessing)
        else:
            valid_names[args.name](args.log, 
                                   args.path_in, 
                                   args.path_out,
                                   skip_preprocessing=args.skip_preprocessing)

#---------------------------------------------------------------------------