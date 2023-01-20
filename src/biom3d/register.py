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

datasets = Dict(
    SegPatchFast    =Dict(fct=SemSeg3DPatchFast, kwargs=Dict()),
)

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
    IoU         =Dict(fct=mt.IoU, kwargs=Dict()),
    # IoUChannel  =Dict(fct=mt.IoUChannel, kwargs=Dict()),
    MSE         =Dict(fct=mt.MSE, kwargs=Dict()),
    CE          =Dict(fct=mt.CrossEntropy, kwargs=Dict()),
    DeepMSE     =Dict(fct=mt.DeepMetric, kwargs=Dict(metric=mt.MSE)),
    DeepDiceBCE =Dict(fct=mt.DeepMetric, kwargs=Dict(metric=mt.DiceBCE)),
    Triplet     =Dict(fct=mt.Triplet, kwargs=Dict()),
    TripletSeg  =Dict(fct=mt.TripletDiceBCE, kwargs=Dict()),
    # NNTriplet   =Dict(fct=mt.NNTriplet, kwargs=Dict()),
    ArcFace     =Dict(fct=mt.ArcFace, kwargs=Dict()),
    FocalLoss   =Dict(fct=mt.FocalLoss, kwargs=Dict()),
    CoTrain     =Dict(fct=mt.CoTrain, kwargs=Dict()),
    DenoiSeg    =Dict(fct=mt.DenoiSeg, kwargs=Dict()),
    Adversarial =Dict(fct=mt.Adversarial, kwargs=Dict()),
    DiceAdverse =Dict(fct=mt.DiceAdversarial, kwargs=Dict()),
    Dino        =Dict(fct=mt.DINOLoss, kwargs=Dict()),
    KLDiv       =Dict(fct=mt.KLDiv, kwargs=Dict()),
    Entropy     =Dict(fct=mt.Entropy, kwargs=Dict()),
    HRDiceBCE   =Dict(fct=mt.HRDiceBCE, kwargs=Dict()),
)

#---------------------------------------------------------------------------
# trainer register

from biom3d.trainers import (
    denoiseg_train,
    denoiseg_val,
    dino_train,
    seg_train, 
    seg_validate,
    triplet_seg_train, 
    triplet_train,
    triplet_val,
    arcface_train,
    arcface_val,
    adverse_train,
    cotrain_train,
    cotrain_validate,
    seg_patch_validate,
    seg_patch_train,
)

trainers = Dict(
    SegTrain        =Dict(fct=seg_train, kwargs=Dict()),
    SegVal          =Dict(fct=seg_validate, kwargs=Dict()),
    SegPatchTrain   =Dict(fct=seg_patch_train, kwargs=Dict()),
    SegPatchVal     =Dict(fct=seg_patch_validate, kwargs=Dict()),
    TripletTrain    =Dict(fct=triplet_train, kwargs=Dict()),
    TripletVal      =Dict(fct=triplet_val, kwargs=Dict()),
    TripletSegTrain =Dict(fct=triplet_seg_train, kwargs=Dict()),
    ArcFaceTrain    =Dict(fct=arcface_train, kwargs=Dict()),
    ArcFaceVal      =Dict(fct=arcface_val, kwargs=Dict()),
    AdverseTrain    =Dict(fct=adverse_train, kwargs=Dict()),
    CoTrainTrain    =Dict(fct=cotrain_train,  kwargs=Dict()),
    CoTrainVal      =Dict(fct=cotrain_validate,  kwargs=Dict()),
    DenoiSegTrain   =Dict(fct=denoiseg_train, kwargs=Dict()),
    DenoiSegVal     =Dict(fct=denoiseg_val, kwargs=Dict()),
    DinoTrain       =Dict(fct=dino_train, kwargs=Dict()),
)

#---------------------------------------------------------------------------
# predictor register

from biom3d.predictors import (
    seg_predict,
    seg_predict_patch,
)

predictors = Dict(
    Seg = Dict(fct=seg_predict, kwargs=Dict()),
    SegPatch = Dict(fct=seg_predict_patch, kwargs=Dict()),
)

#---------------------------------------------------------------------------
