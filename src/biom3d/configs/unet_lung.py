#---------------------------------------------------------------------------
# configuration file
#---------------------------------------------------------------------------

############################################################################
# handy class for dictionary

from pickle import FALSE


class Dict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

############################################################################

#---------------------------------------------------------------------------
# Logs configs

CSV_DIR = 'data/lung/folds_lung.csv'
# CSV_DIR = 'data/nucleus/folds_sophie.csv'
# CSV_DIR = None

# IMG_DIR = 'data/chromo/tif_img'
IMG_DIR = 'data/lung/tif_imagesTr'
# IMG_DIR = 'data/pancreas/tif_train_img'
# IMG_DIR = 'data/pancreas/tif_train_img_17'
# IMG_DIR = 'data/aline/img_resampled'
# IMG_DIR = 'data/nucleus/images_resampled'
# IMG_DIR = 'data/nucleus/images_resampled_sophie'
# IMG_DIR = 'data/nucleus/images_manual'

# MSK_DIR = 'data/chromo/tif_msk'
MSK_DIR = 'data/lung/tif_labelsTr'
# MSK_DIR = 'data/pancreas/tif_train_img_labels'
# MSK_DIR = 'data/pancreas/tif_train_img_labels_17'
# MSK_DIR = 'data/aline/msk_resampled'
# MSK_DIR = 'data/nucleus/masks_resampled_sophie'
# MSK_DIR = 'data/nucleus/masks_resampled'
# MSK_DIR = 'data/nucleus/masks_manual'

# PT_PATH = 'data/pancreas/data_pancreas.pt'
LOG_DIR = 'logs/'

DESC = 'unet_mine-lung'
# DESC = 'unet_mine-chromo_21'

#---------------------------------------------------------------------------
# training configs

SAVE_BEST = True # whether we save also the best model 

NB_EPOCHS = 1000
BATCH_SIZE = 2
LR_START = 1e-2 # comment if need to reload learning rate after training interruption
# LR_MILESTONES = [100, NB_EPOCHS//2, NB_EPOCHS-100]
# LR_T_MAX = NB_EPOCHS
WEIGHT_DECAY = 3e-5

USE_DEEP_SUPERVISION = False
# NUM_POOLS = [3,5,5]
# NUM_POOLS = [3,4,4]
NUM_POOLS = [4,5,5]
# NUM_CLASSES=3
NUM_CLASSES=1
USE_SOFTMAX=False # carefule num_classes changes if use_softmax is enabled/disabled

PATCH_SIZE = [96,160,160]
# PATCH_SIZE = [48,256,256]
# PATCH_SIZE = [64,320,320]
# PATCH_SIZE = [53,266,266]
# PATCH_SIZE = [56,64,64]

# AUG_PATCH_SIZE = [112,192,192]
AUG_PATCH_SIZE = [191, 257, 219]
# AUG_PATCH_SIZE = [56,288,288]

MEDIAN_SPACING=[ 0.79882801, 0.79882801, 1.24499428]

#---------------------------------------------------------------------------
# callback setup

SAVE_MODEL_EVERY_EPOCH = 1
USE_IMAGE_CLBK = True
VAL_EVERY_EPOCH = SAVE_MODEL_EVERY_EPOCH
SAVE_IMAGE_EVERY_EPOCH = SAVE_MODEL_EVERY_EPOCH
USE_FG_CLBK = True

#---------------------------------------------------------------------------
# dataset configs

# TRAIN_DATASET = Dict(
#     fct="Seg",
#     kwargs=Dict(
#         img_dir     =IMG_DIR,
#         msk_dir     =MSK_DIR,  
#         folds_csv   =CSV_DIR,
#         fold        = 0,
#         val_split   = 0.36,
#         train       =True,
#         use_onehot  =False,
#     )
# )



# TRAIN_DATASET = Dict(
#     fct="SegFast",
#     kwargs=Dict(
#         pt_path     =PT_PATH,
#         val_split   = 0.2,
#         train       =True,
#     )
# )

TRAIN_DATASET = Dict(
    fct="SegPatchFast",
    kwargs=Dict(
        img_dir    = IMG_DIR,
        msk_dir    = MSK_DIR, 
        batch_size = BATCH_SIZE, 
        patch_size = PATCH_SIZE,
        nbof_steps = 250,
        folds_csv  = CSV_DIR, 
        fold       = 0, 
        val_split  = 0.20,
        train      = True,
        use_aug    = True,
        aug_patch_size = AUG_PATCH_SIZE,
        use_softmax  = USE_SOFTMAX,
        fg_rate    = 0.33, # use the foreground scheduler
    )
)

TRAIN_DATALOADER_KWARGS = Dict(
    batch_size  =BATCH_SIZE, 
    drop_last   =False, 
    shuffle     =True, 
    num_workers =4, 
    pin_memory  =True,
    # persistent_workers=True,
)          


# TRAIN_DATALOADER = Dict(
#     fct="SegPatchFast",
#     kwargs = Dict(
#         img_dir    = IMG_DIR,
#         msk_dir    = MSK_DIR, 
#         batch_size = BATCH_SIZE, 
#         patch_size = [40,224,224],
#         nbof_steps = 250,
#         folds_csv  = CSV_DIR, 
#         fold       = 0, 
#         val_split  = 0.25,
#         train      = True,
#         use_aug    = False,
#     )
# )


# VAL_DATASET = Dict(
#     fct="Seg",
#     kwargs = Dict(
#         img_dir     =IMG_DIR,
#         msk_dir     =MSK_DIR,  
#         folds_csv   =CSV_DIR,
#         fold        = 0,
#         val_split   = 0.36,
#         train       =False,
#         use_onehot  =False,
#     )
# )

# VAL_DATASET = Dict(
#     fct="SegFast",
#     kwargs = Dict(
#         pt_path     =PT_PATH,
#         val_split   = 0.2,
#         train       =False,
#     )
# )

VAL_DATASET = Dict(
    fct="SegPatchFast",
    kwargs = Dict(
        img_dir    = IMG_DIR,
        msk_dir    = MSK_DIR, 
        batch_size = BATCH_SIZE, 
        patch_size = PATCH_SIZE,
        nbof_steps = 50,
        folds_csv  = CSV_DIR, 
        fold       = 0, 
        val_split  = 0.20,
        train      = False,
        use_aug    = False,
        aug_patch_size = AUG_PATCH_SIZE,
        use_softmax  = USE_SOFTMAX,
        fg_rate    = 0.33,
    )
)

VAL_DATALOADER_KWARGS = Dict(
    batch_size  =2, # TODO: change it in the final version
    drop_last   =False, 
    shuffle     =False, 
    num_workers =4,  
    pin_memory  =True,
    # persistent_workers=True,
)


# VAL_DATALOADER = Dict(
#     fct="SegPatchFast",
#     kwargs = Dict(
#         img_dir    = IMG_DIR,
#         msk_dir    = MSK_DIR, 
#         batch_size = BATCH_SIZE, 
#         patch_size = [40,224,224],
#         nbof_steps = 50,
#         folds_csv  = CSV_DIR, 
#         fold       = 0, 
#         val_split  = 0.25,
#         train      = False,
#         use_aug    = False,
#     )
# )

#---------------------------------------------------------------------------
# model configs

MODEL = Dict(
    fct="UNet3DVGGDeep", # from the register
    kwargs = Dict(
        num_pools=NUM_POOLS,
        num_classes=NUM_CLASSES,
        factor = 32,
        use_deep=USE_DEEP_SUPERVISION,
    )
)



# MODEL = Dict(
#     fct="FPN", # from the register
#     kwargs = Dict(
#         # pyramid={                       # size  fmaps
#         # 0: ['_bn0'],                    # 64    32
#         # 1: ['_blocks', '0', '_bn2'],    # 32    16
#         # 2: ['_blocks', '2', '_bn2'],    # 16    24
#         # 3: ['_blocks', '4', '_bn2'],    # 8     40
#         # 4: ['_blocks', '10', '_bn2'],   # 4     112
#         # 5: ['_blocks', '15', '_bn2']    # 2     320
#         # },
#         pyramid={                       # efficientnet b4
#         0: ['_bn0'],                    
#         1: ['_blocks', '1', '_bn2'],   
#         2: ['_blocks', '5', '_bn2'],  
#         3: ['_blocks', '9', '_bn2'],    
#         4: ['_blocks', '21', '_bn2'],  
#         # 5: ['_blocks', '31', '_bn2']  
#         },
#         in_dim=1,
#         out_dim=1,
#         num_filters=1,
#     )
# )


# MODEL = Dict(
#     fct="BasicUNetMonai",
#     kwargs = Dict(out_channels=2),
# )

#---------------------------------------------------------------------------
# loss configs

# TRAIN_LOSS = Dict(
#     fct="DeepDiceBCE",
#     kwargs = Dict(
#         milestones=[200],
#         alphas=[1/5, 0],
#         name="train_loss",
#     )
# )

# TRAIN_LOSS = Dict(
#     fct="DeepDiceBCE",
#     kwargs = Dict(
#         alphas=[0, 0, 0.06667, 0.1333, 0.2667, 0.5333],
#         name="train_loss",
#     )
# )

TRAIN_LOSS = Dict(
    fct="DiceBCE",
    kwargs = Dict(name="train_loss", use_softmax=USE_SOFTMAX)
)

VAL_LOSS = Dict(
    fct="DiceBCE",
    kwargs = Dict(name="val_loss", use_softmax=USE_SOFTMAX)
)

#---------------------------------------------------------------------------
# metrics configs

TRAIN_METRICS = Dict(
    train_iou=Dict(
        fct="IoU",
        kwargs = Dict(name="train_iou", use_softmax=USE_SOFTMAX)),
    train_dice=Dict(
        fct="Dice",
        kwargs=Dict(name="train_dice", use_softmax=USE_SOFTMAX)),
)

VAL_METRICS = Dict(
    val_iou=Dict(
        fct="IoU",
        kwargs = Dict(name="val_iou", use_softmax=USE_SOFTMAX)),
    val_dice=Dict(
        fct="Dice",
        kwargs=Dict(name="val_dice", use_softmax=USE_SOFTMAX)),
)

#---------------------------------------------------------------------------
# trainers configs

TRAINER = Dict(
    fct="SegTrain",
    kwargs=Dict(),
)

VALIDATER = Dict(
    fct="SegVal",
    kwargs=Dict(),
)

#---------------------------------------------------------------------------
# predictors configs

# PREDICTOR = Dict(
#     fct="SegFast",
#     kwargs=Dict(input_shape=[40,224,224]),
# )
PREDICTOR = Dict(
    fct="SegPatch",
    kwargs=Dict(patch_size=PATCH_SIZE, tta=True, median_spacing=MEDIAN_SPACING, use_softmax=USE_SOFTMAX),
)

############################################################################
# end of config file
# do not write anything in or below this field

CONFIG = Dict(**globals().copy()) # stores all variables in one Dict

to_remove = ['__name__', '__doc__', '__package__',
    '__loader__' , '__spec__', '__annotations__',
    '__builtins__', '__file__', '__cached__', 'Dict']

for k in to_remove: 
    if (k in CONFIG.keys()): del CONFIG[k] 

############################################################################
