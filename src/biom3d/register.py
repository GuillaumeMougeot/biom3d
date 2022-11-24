#---------------------------------------------------------------------------
# A register for all the existing methods
# Aim of this module: 
# - to gather all required imports in a single file
# - to use it in colaboration with a config file
#---------------------------------------------------------------------------

from stat import FILE_ATTRIBUTE_NO_SCRUB_DATA
from utils import Dict

#---------------------------------------------------------------------------
# dataset register

# import datasets.datasets_old as datasets_old
# from datasets.arcface import ArcFace
# from datasets.triplet import Triplet
# from datasets.triplet_seg import TripletSeg
# from datasets.semseg import SemSeg3D
# from datasets.semseg_fast import SemSeg3DFast
# from datasets.semseg_patch import SemSeg3DPatchTrain, SemSeg3DPatchVal
from datasets.semseg_patch_fast import SemSeg3DPatchFast
# from datasets.model_genesis import Genesis
# from datasets.denoiseg import DenoiSeg
# from datasets.adversarial import Adversarial
# from datasets.dino import Dino
# from datasets.arcface import ArcFace

datasets = Dict(
    # Seg             =Dict(fct=SemSeg3D, kwargs=Dict()),
    # SegFast         =Dict(fct=SemSeg3DFast, kwargs=Dict()),
    # SingleNucleus   =Dict(fct=datasets_old.Nucleus3DSingle, kwargs=Dict()),
    # Triplet         =Dict(fct=Triplet, kwargs=Dict()),
    # TripletSeg      =Dict(fct=TripletSeg, kwargs=Dict()),
    # ArcFace         =Dict(fct=ArcFace, kwargs=Dict()),
    # CoTrain         =Dict(fct=datasets_old.CoTrain, kwargs=Dict()),
    # SegPatchTrain   =Dict(fct=SemSeg3DPatchTrain, kwargs=Dict()),
    # SegPatchVal     =Dict(fct=SemSeg3DPatchVal, kwargs=Dict()),
    SegPatchFast    =Dict(fct=SemSeg3DPatchFast, kwargs=Dict()),
    # Genesis         =Dict(fct=Genesis, kwargs=Dict()),
    # DenoiSeg        =Dict(fct=DenoiSeg, kwargs=Dict()),
    # Adversarial     =Dict(fct=Adversarial, kwargs=Dict()),
    # Dino            =Dict(fct=Dino, kwargs=Dict()),
    # ArcFace         =Dict(fct=ArcFace, kwargs=Dict()),
)


from utils import (
    # skimage_imread,
    sitk_imread,
)

imread = Dict(
    # Skimage =Dict(fct=skimage_imread, kwargs=Dict()),
    Sitk    =Dict(fct=sitk_imread, kwargs=Dict()),
)

#---------------------------------------------------------------------------
# model register

from models.unet3d_vgg_deep import UNet
from models.encoder_vgg import VGGEncoder, EncoderBlock

models = Dict(
    UNet3DVGGDeep   =Dict(fct=UNet, kwargs=Dict()),
    VGG3D           =Dict(fct=VGGEncoder, kwargs=Dict(block=EncoderBlock)),
)

#---------------------------------------------------------------------------
# metric register

import metrics as mt

metrics = Dict(
    Dice        =Dict(fct=mt.Dice, kwargs=Dict()),
    DiceBCE     =Dict(fct=mt.DiceBCE, kwargs=Dict()),
    IoU         =Dict(fct=mt.IoU, kwargs=Dict()),
    # DeepMSE     =Dict(fct=mt.DeepMetric, kwargs=Dict(metric=mt.MSE)),
    DeepDiceBCE =Dict(fct=mt.DeepMetric, kwargs=Dict(metric=mt.DiceBCE)),
)

#---------------------------------------------------------------------------
# trainer register

from trainers import (
    # denoiseg_train,
    # denoiseg_val,
    # dino_train,
    seg_train, 
    seg_validate,
    # triplet_seg_train, 
    # triplet_train,
    # triplet_val,
    # arcface_train,
    # arcface_val,
    # adverse_train,
    # cotrain_train,
    # cotrain_validate,
    # seg_patch_validate,
    # seg_patch_train,
)

trainers = Dict(
    SegTrain        =Dict(fct=seg_train, kwargs=Dict()),
    SegVal          =Dict(fct=seg_validate, kwargs=Dict()),
    # SegPatchTrain   =Dict(fct=seg_patch_train, kwargs=Dict()),
    # SegPatchVal     =Dict(fct=seg_patch_validate, kwargs=Dict()),
    # TripletTrain    =Dict(fct=triplet_train, kwargs=Dict()),
    # TripletVal      =Dict(fct=triplet_val, kwargs=Dict()),
    # TripletSegTrain =Dict(fct=triplet_seg_train, kwargs=Dict()),
    # ArcFaceTrain    =Dict(fct=arcface_train, kwargs=Dict()),
    # ArcFaceVal      =Dict(fct=arcface_val, kwargs=Dict()),
    # AdverseTrain    =Dict(fct=adverse_train, kwargs=Dict()),
    # CoTrainTrain    =Dict(fct=cotrain_train,  kwargs=Dict()),
    # CoTrainVal      =Dict(fct=cotrain_validate,  kwargs=Dict()),
    # DenoiSegTrain   =Dict(fct=denoiseg_train, kwargs=Dict()),
    # DenoiSegVal     =Dict(fct=denoiseg_val, kwargs=Dict()),
    # DinoTrain       =Dict(fct=dino_train, kwargs=Dict()),
)

#---------------------------------------------------------------------------
# predictor register

from predictors import (
    # seg_predict,
    seg_predict_patch,
)

predictors = Dict(
    # Seg = Dict(fct=seg_predict, kwargs=Dict()),
    SegPatch = Dict(fct=seg_predict_patch, kwargs=Dict()),
)

#---------------------------------------------------------------------------
