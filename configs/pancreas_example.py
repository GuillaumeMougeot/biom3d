#---------------------------------------------------------------------------
# Configuration file. 
# Copy it and edit it as much as you want! :) 
#
# FORMATTING DETAILS:
# Each key and values must be written with the format: KEY = value
# Do not forget the spaces before and after the '=' symbol! This is 
# especially import for the global variables, such as BATCH_SIZE or
# PATCH_SIZE.
#---------------------------------------------------------------------------

############################################################################
# handy class for dictionary

class Dict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

############################################################################

#---------------------------------------------------------------------------
# Dataset builder-parameters
# EDIT THE FOLLOWING PARAMATERS WITH YOUR OWN DATASETS PARAMETERS

# Folder where pre-processed images are stored
IMG_DIR = '/home/gumougeot/all/codes/python/biom3d/data/pancreas/imagesTs_tiny_out'
# Folder where pre-processed masks are stored
MSK_DIR = '/home/gumougeot/all/codes/python/biom3d/data/pancreas/labelsTs_tiny_out'
# (optional) path to the .csv file storing "filename,hold_out,fold", where:
# "filename" is the image name,
# "hold_out" is either 0 (training image) or 1 (testing image),
# "fold" (non-negative integer) indicates the k-th fold, 
# by default fold 0 of the training image (hold_out=0) is the validation set.
# CSV_DIR = 'data/pancreas/folds_pancreas.csv'

# CSV_DIR can be set to None, in which case the validation set will be
# automatically chosen from the training set (20% of the training images/masks)
CSV_DIR = None 

# model name
DESC = 'unet_default'

# number of classes of objects
# the background does not count, so the minimum is 1 (the max is 255)
NUM_CLASSES = 2
#---------------------------------------------------------------------------
# Auto-config builder-parameters
# PASTE AUTO-CONFIG RESULTS HERE

# batch size
BATCH_SIZE = 2
# patch size passed to the model
PATCH_SIZE = [40, 224, 224]
# larger patch size used prior rotation augmentation to avoid "empty" corners.
AUG_PATCH_SIZE = [56, 288, 288]
# number of pooling done in the UNet
NUM_POOLS = [3, 5, 5]
# median spacing is used only during prediction to normalize the output images
# it is commented here because we did not noticed any improvement yet
# MEDIAN_SPACING=[0.79492199, 0.79492199, 2.5]
MEDIAN_SPACING = []

#---------------------------------------------------------------------------
# Advanced paramaters (can be left as such) 
# training configs

# whether to store also the best model 
SAVE_BEST = True 

# number of epochs
# the number of epochs can be reduced for small training set (e.g. a set of 10 images/masks of 128x128x64)
NB_EPOCHS = 1000

# optimizer paramaters
LR_START = 1e-2 # comment if need to reload learning rate after training interruption
WEIGHT_DECAY = 3e-5

# whether to use deep-supervision loss:
# a loss is placed at each stage of the UNet model
USE_DEEP_SUPERVISION = False

# whether to use softmax loss instead of sigmoid
# should not be set to True if object classes are overlapping in the masks
USE_SOFTMAX = True 

# training loop parameters
USE_FP16 = True
NUM_WORKERS = 6
PIN_MEMORY = True

#---------------------------------------------------------------------------
# callback setup (can be left as such) 
# callbacks are routines that execute periodically during the training loop

# folder where the training logs will be stored, including:
# - model .pth files (state_dict)
# - image snapshots of model training (only if USE_IMAGE_CLBK is True)
# - logs with this configuration stored in .yaml format and tensorboard logs
LOG_DIR = 'logs/'

SAVE_MODEL_EVERY_EPOCH = 1
USE_IMAGE_CLBK = True
VAL_EVERY_EPOCH = SAVE_MODEL_EVERY_EPOCH
SAVE_IMAGE_EVERY_EPOCH = SAVE_MODEL_EVERY_EPOCH
USE_FG_CLBK = True

#---------------------------------------------------------------------------
# dataset configs

TRAIN_DATASET = Dict(
    fct="Torchio",
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
    )
)

TRAIN_DATALOADER_KWARGS = Dict(
    batch_size  = BATCH_SIZE, 
    drop_last   = True, 
    shuffle     = True, 
    num_workers = NUM_WORKERS, 
    pin_memory  = PIN_MEMORY,
)          

VAL_DATASET = Dict(
    fct="Torchio",
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
        use_softmax  = USE_SOFTMAX,
        fg_rate    = 0.33,
    )
)

VAL_DATALOADER_KWARGS = Dict(
    batch_size  = BATCH_SIZE, # TODO: change it in the final version
    drop_last   = False, 
    shuffle     = False, 
    num_workers = NUM_WORKERS, 
    pin_memory  = PIN_MEMORY,
)

#---------------------------------------------------------------------------
# model configs

MODEL = Dict(
    fct="UNet3DVGGDeep", # from the register
    kwargs = Dict(
        num_pools=NUM_POOLS,
        num_classes=NUM_CLASSES if not USE_SOFTMAX else NUM_CLASSES+1,
        factor = 32,
        use_deep=USE_DEEP_SUPERVISION,
    )
)

#---------------------------------------------------------------------------
# loss configs

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

PREDICTOR = Dict(
    fct="SegPatch",
    kwargs=Dict(patch_size=PATCH_SIZE, tta=True, median_spacing=MEDIAN_SPACING, use_softmax=USE_SOFTMAX, keep_biggest_only=False),
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
