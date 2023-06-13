#---------------------------------------------------------------------------
# A register for all the existing methods
# Aim of this module: 
# - to gather all required imports in a single file
# - to use it in colaboration with a config file
#---------------------------------------------------------------------------

from biom3d.utils import Dict

#---------------------------------------------------------------------------
# dataset register

from biom3d.datasets.semseg_patch_fast import SemSeg3DPatchFast
from biom3d.datasets.semseg_torchio import TorchioDataset

datasets = Dict(
    SegPatchFast    =Dict(fct=SemSeg3DPatchFast, kwargs=Dict()),
    Torchio         =Dict(fct=TorchioDataset, kwargs=Dict()),
)

try:
    from biom3d.datasets.semseg_batchgen import MTBatchGenDataLoader
    datasets.BatchGen = Dict(fct=MTBatchGenDataLoader, kwargs=Dict())
except:
    pass

#---------------------------------------------------------------------------
# model register

from biom3d.models.unet3d_vgg_deep import UNet
from biom3d.models.encoder_vgg import VGGEncoder, EncoderBlock

models = Dict(
    UNet3DVGGDeep   =Dict(fct=UNet, kwargs=Dict()),
    VGG3D           =Dict(fct=VGGEncoder, kwargs=Dict(block=EncoderBlock, use_head=True)),
)

#---------------------------------------------------------------------------
# metric register

import biom3d.metrics as mt

metrics = Dict(
    Dice        =Dict(fct=mt.Dice, kwargs=Dict()),
    DiceBCE     =Dict(fct=mt.DiceBCE, kwargs=Dict()),
    DiceCEnnUNet=Dict(fct=mt.DC_and_CE_loss, kwargs=Dict(soft_dice_kwargs={'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, ce_kwargs={})),
    IoU         =Dict(fct=mt.IoU, kwargs=Dict()),
    MSE         =Dict(fct=mt.MSE, kwargs=Dict()),
    CE          =Dict(fct=mt.CrossEntropy, kwargs=Dict()),
    DeepMSE     =Dict(fct=mt.DeepMetric, kwargs=Dict(metric=mt.MSE)),
    DeepDiceBCE =Dict(fct=mt.DeepMetric, kwargs=Dict(metric=mt.DiceBCE)),
)

#---------------------------------------------------------------------------
# trainer register

from biom3d.trainers import (
    seg_train, 
    seg_validate,
    seg_patch_validate,
    seg_patch_train,
)

trainers = Dict(
    SegTrain        =Dict(fct=seg_train, kwargs=Dict()),
    SegVal          =Dict(fct=seg_validate, kwargs=Dict()),
    SegPatchTrain   =Dict(fct=seg_patch_train, kwargs=Dict()),
    SegPatchVal     =Dict(fct=seg_patch_validate, kwargs=Dict()),
)

#---------------------------------------------------------------------------
# Preprocessor and predictor register
# We register preprocessors here because they are needed to preprocess
# data before prediction.
# Preprocessor must correspond to the one used to preprocess data
# before training.

from biom3d.preprocess import seg_preprocessor

preprocessors = Dict(
    Seg = Dict(fct=seg_preprocessor, kwargs=Dict())
)

from biom3d.predictors import (
    seg_predict,
    seg_predict_patch_2,
)

predictors = Dict(
    Seg = Dict(fct=seg_predict, kwargs=Dict()),
    SegPatch = Dict(fct=seg_predict_patch_2, kwargs=Dict()),
)

from biom3d.predictors import seg_postprocessing

postprocessors = Dict(
    Seg = Dict(fct=seg_postprocessing, kwargs=Dict())
)

#---------------------------------------------------------------------------
# Display utils

# import argparse

# if __name__=="__main__":
#     parser = argparse.ArgumentParser(description="Register of all modules in biom3d.")
#     parser.add_argument("--one", type=str, default=None,
#         help="Print one specific documentation of one module given its category:name in the register.")  
#     parser.add_argument("--all", default=False,  action='store_true', dest='all',
#         help="Print the documentation of all the modules.")
#     args = parser.parse_args()

#     all_cat = [datasets, models, metrics, trainers, preprocessors, predictors, postprocessors]

#     if args.all:
#         for cat in all_cat:
#             for k,v in cat.items():
#                 print("Module:", k)
#                 if v['fct'].__doc__ is not None:
#                     print(v['fct'].__doc__)
#                 print("")
#     elif args.one is not None:
#         # check if it is a valid category
        

#         # check if it is a valid name
#         idx_valid=-1
#         for i,cat in enumerate(all_cat):
#             if args.one in cat.keys():
#                 idx_valid = i
#         if idx_valid<0:
#             print("[Error] Not a correct name:", args.one)
#             raise RuntimeError
#         else:
#             help(all_cat[idx_valid][args.one]['fct'])


#---------------------------------------------------------------------------
