import copy
import os

from box import Box

C = Box()
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
C.AUTO_LR_FIND = True

C.CHECKPOINT = Box()
C.CHECKPOINT.NAME = '{epoch}-{val_loss:.2f}-{val_mcc:.2f}'
C.CHECKPOINT.MONITOR = 'val_mcc'
C.CHECKPOINT.MONITOR_MODE = 'max'

C.EARLY_STOPPING = Box()
C.EARLY_STOPPING.min_delta = 0.1
C.EARLY_STOPPING.patience = 10
C.EARLY_STOPPING.verbose = True

# -----------------------------------------------------------------------------
# TRAINER
# -----------------------------------------------------------------------------
C.TRAINER = Box()
C.TRAINER.accumulate_grad_batches = 1

# -----------------------------------------------------------------------------
# AUGMENTATION
# -----------------------------------------------------------------------------
C.AUGMENTATION = ''

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
C.MODEL = Box()
C.MODEL.DEVICE = "cuda"
# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
C.MODEL.BACKBONE = Box()
C.MODEL.BACKBONE.WEIGHT = ""
C.MODEL.BACKBONE.NAME = "resnet18"
C.MODEL.BACKBONE.FREEZE = True
C.MODEL.BACKBONE.USE_PRETRAINED = True
C.MODEL.BACKBONE.UNFREEZE_LAYERS = [-1]

C.MODEL.HEAD = Box()
C.MODEL.HEAD.NFEATURES = 2048

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
C.INPUT = Box()
# crop size
C.INPUT.CENTER_CROP_SIZE = [640, 640]
# Values to be used for image normalization
C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
C.DATASETS = Box()
# List of the dataset names for training, as present in paths_catalog.py
C.DATASETS.TRAIN = Box(data_name="cloglossv0d0_overfit", split="train")

# List of the dataset names for testing, as present in paths_catalog.py
C.DATASETS.TEST = Box(data_name="cloglossv0d0_overfit", split="validation")
C.DATASETS.NUM_CLASSES = 2

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
C.DATALOADER = Box()
C.DATALOADER.SHUFFLE = True
# Number of data loading threads
C.DATALOADER.NUM_WORKERS = 16
C.DATALOADER.CLASS_BALANCING = False

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
C.SOLVER = Box()

C.SOLVER.LOSS = Box()
C.SOLVER.LOSS.NAME = "CrossEntropyLoss"
C.SOLVER.LOSS.PARAMS = {}

C.SOLVER.OPTIMIZER = Box()
C.SOLVER.OPTIMIZER.NAME = 'Adam'
C.SOLVER.OPTIMIZER.PARAMS = {}

C.SOLVER.SCHEDULER = Box()
C.SOLVER.SCHEDULER.NAME = "OneCycleLR"
C.SOLVER.SCHEDULER.PARAMS = Box(total_steps=10000)
C.SOLVER.SCHEDULER_META = Box(interval='step', monitor='val_acc')
C.SOLVER.IMS_PER_BATCH = 32
C.SOLVER.DEFAULT_LR = 0.0001

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
C.TEST = Box()
C.TEST.EXPECTED_RESULTS = []
C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
C.TEST.IMS_PER_BATCH = 8


def get():
    return copy.deepcopy(C)
