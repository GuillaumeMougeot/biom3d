#---------------------------------------------------------------------------
# Main code: run predictions
#---------------------------------------------------------------------------

import argparse
import pathlib
from biom3d.builder import Builder
from biom3d.utils import deprecated, versus_one, dice, DataHandlerFactory
from biom3d.eval import eval

#---------------------------------------------------------------------------
# prediction base fonction

def pred_single(log, img_path,out_path,is_2d = False,skip_preprocessing=False):
    """Prediction on a single image.
    """
    if not isinstance(log,list): log=str(log)
    builder = Builder(config=None,path=log, training=False)
    handler = DataHandlerFactory.get(
            img_path,
            output=out_path,
            img_path = img_path,
            msk_outpath = out_path,
            model_name = builder.config[-1].DESC if isinstance(builder.config,list) else builder.config.DESC,
        )
    img = builder.run_prediction_single(handler, return_logit=False,is_2d=is_2d,skip_preprocessing=skip_preprocessing)
    handler.save(handler.images[0], img,"pred")
    return builder.config.NUM_CLASSES+1,handler.msk_outpath  # for pred_seg_eval_single

def pred(log, path_in, path_out,is_2d = False,skip_preprocessing=False):
    """Prediction on a folder of images.
    """
    if not isinstance(log,list): log=str(log)
    path_in=str(path_in)
    path_out=str(path_out)

    builder = Builder(config=None,path=log, training=False)
    path_out = builder.run_prediction_folder(path_in=path_in, path_out=path_out, return_logit=False,is_2d=is_2d,skip_preprocessing=skip_preprocessing)
    return path_out

@deprecated("This method is no longer used as it is the default behaviour of DataHandlers.")
def pred_multiple(log, path_in, path_out,is_2d = False,skip_preprocessing=False):
    """Prediction a folder of folders of images.
    """
    return pred(log,path_in,path_out,is_2d=is_2d,skip_preprocessing=skip_preprocessing)

#---------------------------------------------------------------------------
# main unet segmentation
def pred_seg(log=pathlib.Path.home(), path_in=pathlib.Path.home(), path_out=pathlib.Path.home(),is_2d = False,skip_preprocessing=False):
    pred(log, path_in, path_out,is_2d=is_2d,skip_preprocessing=skip_preprocessing)

# TODO remove eval only, we have a module for that
def pred_seg_eval(log=pathlib.Path.home(), path_in=pathlib.Path.home(), path_out=pathlib.Path.home(), path_lab=None, eval_only=False,is_2d = False,skip_preprocessing=False):
    print("Start inference")
    builder_pred = Builder(
        config=None,
        path=log, 
        training=False)
    out = path_out
    if not eval_only:
        out = builder_pred.run_prediction_folder(path_in=path_in, path_out=path_out, return_logit=False,is_2d=is_2d,skip_preprocessing=skip_preprocessing) # run the predictions
    print("Inference done!")


    if path_lab is not None:
        if isinstance(builder_pred.config,list):
            num_classes = builder_pred.config[0].NUM_CLASSES+1
        else:
            num_classes = builder_pred.config.NUM_CLASSES+1
        # eval
        eval(path_lab,out,num_classes=num_classes)

def pred_seg_eval_single(log, img_path, out_path, msk_path,is_2d = False,skip_preprocessing=False):
    print("Run prediction for:", img_path)
    num_classes,out = pred_single(log, img_path, out_path,is_2d=is_2d,skip_preprocessing=skip_preprocessing)
    print("Done! Prediction saved in:", out_path)
    handler1 = DataHandlerFactory.get(
        out_path,
        read_only=True,
        img_path = out,
        eval="pred",
    )
    handler2 = DataHandlerFactory.get(
        msk_path,
        read_only=True,
        img_path = msk_path,
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
    parser.add_argument("--is_2d", default=False, dest="is_2d",
        help="(default=False) Whether the image is 2d.")
    parser.add_argument("--skip_preprocessing", default=False, action='store_true',dest="skip_prepprocessing",
        help="(default=False) Skip preprocessing")
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
                                   is_2d=args.is_2d,
                                   skip_preprocessing=args.skip_preprocessing)
        elif args.name=="seg_eval_single":
            valid_names[args.name](args.log, 
                                   args.path_in, 
                                   args.path_out, 
                                   args.path_lab,
                                   is_2d = args.is_2d,
                                   skip_preprocessing=args.skip_preprocessing)
        else:
            valid_names[args.name](args.log, 
                                   args.path_in, 
                                   args.path_out,
                                   is_2d = args.is_2d,
                                   skip_preprocessing=args.skip_preprocessing)

#---------------------------------------------------------------------------