#---------------------------------------------------------------------------
# A register for all the existing methods
# Aim of this module: 
# - to gather all required imports in a single file
# - to use it in colaboration with a config file
#---------------------------------------------------------------------------

from biom3d.utils import AttrDict

#---------------------------------------------------------------------------
# dataset register

from biom3d.datasets.semseg_patch_fast import SemSeg3DPatchFast
from biom3d.datasets.semseg_torchio import TorchioDataset

datasets = AttrDict(
    SegPatchFast    =AttrDict(fct=SemSeg3DPatchFast, kwargs=AttrDict()),
    Torchio         =AttrDict(fct=TorchioDataset, kwargs=AttrDict()),
)

try:
    # Batchgen use nnUnet batchgenerator that may not be installed (it is not a dependency), do pip install batchgenerators
    from biom3d.datasets.semseg_batchgen import MTBatchGenDataLoader
    datasets.BatchGen = AttrDict(fct=MTBatchGenDataLoader, kwargs=AttrDict())
except:
    pass

#---------------------------------------------------------------------------
# model register

from biom3d.models.encoder_vgg import VGGEncoder, EncoderBlock
from biom3d.models.unet3d_vgg_deep import UNet
from biom3d.models.encoder_efficientnet3d import EfficientNet3D
from biom3d.models.unet3d_eff import EffUNet
from monai.networks import nets

models = AttrDict(
    VGG3D           =AttrDict(fct=VGGEncoder, kwargs=AttrDict(block=EncoderBlock, use_head=True)),
    UNet3DVGGDeep   =AttrDict(fct=UNet, kwargs=AttrDict()),
    Eff3D           =AttrDict(fct=EfficientNet3D.from_name, kwargs=AttrDict()),
    EffUNet         =AttrDict(fct=EffUNet, kwargs=AttrDict()),
    SwinUNETR       =AttrDict(fct=nets.SwinUNETR, kwargs=AttrDict()),
)

#---------------------------------------------------------------------------
# metric register

import biom3d.metrics as mt

metrics = AttrDict(
    Dice        =AttrDict(fct=mt.Dice, kwargs=AttrDict()),
    DiceBCE     =AttrDict(fct=mt.DiceBCE, kwargs=AttrDict()),
    DiceCEnnUNet=AttrDict(fct=mt.DC_and_CE_loss, kwargs=AttrDict(soft_dice_kwargs={'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, ce_kwargs={})),
    IoU         =AttrDict(fct=mt.IoU, kwargs=AttrDict()),
    MSE         =AttrDict(fct=mt.MSE, kwargs=AttrDict()),
    CE          =AttrDict(fct=mt.CrossEntropy, kwargs=AttrDict()),
    DeepMSE     =AttrDict(fct=mt.DeepMetric, kwargs=AttrDict(metric=mt.MSE)),
    DeepDiceBCE =AttrDict(fct=mt.DeepMetric, kwargs=AttrDict(metric=mt.DiceBCE)),
)

#---------------------------------------------------------------------------
# trainer register

from biom3d.trainers import (
    seg_train, 
    seg_validate,
    seg_patch_validate,
    seg_patch_train,
)

trainers = AttrDict(
    SegTrain        =AttrDict(fct=seg_train, kwargs=AttrDict()),
    SegVal          =AttrDict(fct=seg_validate, kwargs=AttrDict()),
    SegPatchTrain   =AttrDict(fct=seg_patch_train, kwargs=AttrDict()),
    SegPatchVal     =AttrDict(fct=seg_patch_validate, kwargs=AttrDict()),
)

#---------------------------------------------------------------------------
# Preprocessor and predictor register
# We register preprocessors here because they are needed to preprocess
# data before prediction.
# Preprocessor must correspond to the one used to preprocess data
# before training.

from biom3d.preprocess import seg_preprocessor

preprocessors = AttrDict(
    Seg = AttrDict(fct=seg_preprocessor, kwargs=AttrDict())
)

from biom3d.predictors import (
    seg_predict,
    seg_predict_patch_2,
)

predictors = AttrDict(
    Seg = AttrDict(fct=seg_predict, kwargs=AttrDict()),
    SegPatch = AttrDict(fct=seg_predict_patch_2, kwargs=AttrDict()),
)

from biom3d.predictors import seg_postprocessing

postprocessors = AttrDict(
    Seg = AttrDict(fct=seg_postprocessing, kwargs=AttrDict())
)

