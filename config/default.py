from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.BATCH_SIZE = 2
_C.INPUT_SIZE = 640

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'RetinaFace_VarGNetv2'
_C.MODEL.OUT_CHANNEL = 256

_C.MODEL.ANCHOR = CN()
_C.MODEL.ANCHOR.MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
_C.MODEL.ANCHOR.STEPS = [8, 16, 32]
_C.MODEL.ANCHOR.MATCH_THRESH = 0.45
_C.MODEL.ANCHOR.IGNORE_THRESH = 0.3
_C.MODEL.ANCHOR.VARIANCES = [0.1, 0.2]
_C.MODEL.ANCHOR.CLIP = False

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'VarGNetv2'
_C.MODEL.BACKBONE.BATCHNORM_MOMENTUM = 0.9
_C.MODEL.BACKBONE.BATCHNORM_EPSILON = 2e-5
_C.MODEL.BACKBONE.FIX_GAMMA = False
_C.MODEL.BACKBONE.USE_GLOBAL_STATS = False
_C.MODEL.BACKBONE.WORKSPACE = 512
_C.MODEL.BACKBONE.ACTIVATION_TYPE = 'lrelu'
_C.MODEL.BACKBONE.USE_SEBLOCK = True
_C.MODEL.BACKBONE.SE_RATIO = 4
_C.MODEL.BACKBONE.GROUP_BASE = 8

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = '/Users/mater/Documents/projects/SegNet/cityscapes'
_C.DATASET.DATASET = 'cityscapes_panoptic'
# _C.DATASET.TRAIN_PATH = 'train'
# _C.DATASET.TEST_PATH = 'test'
# _C.DATASET.LENGTH = 12880
# _C.DATASET.USE_BIN = True
# _C.DATASET.USE_FLIP = True
# _C.DATASET.USE_DISTORT = True

_C.DATASET.NUM_CLASSES = 19
_C.DATASET.TRAIN_SPLIT = 'train'
_C.DATASET.TEST_SPLIT = 'val'
_C.DATASET.CROP_SIZE = (512, 1024)
_C.DATASET.MIRROR = True
_C.DATASET.MIN_SCALE = 0.5
_C.DATASET.MAX_SCALE = 2.0
_C.DATASET.SCALE_STEP_SIZE = 0.1
_C.DATASET.MEAN = (0.485, 0.456, 0.406)
_C.DATASET.STD = (0.229, 0.224, 0.225)
_C.DATASET.SEMANTIC_ONLY = False
_C.DATASET.IGNORE_STUFF_IN_OFFSET = True
_C.DATASET.SMALL_INSTANCE_AREA = 0
_C.DATASET.SMALL_INSTANCE_WEIGHT = 1

_C.DATASET.MIN_RESIZE_VALUE = -1
_C.DATASET.MAX_RESIZE_VALUE = -1
_C.DATASET.RESIZE_FACTOR = -1

# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCH = 100
_C.TRAIN.INIT_LR = 1e-2
_C.TRAIN.LR_DECAY_EPOCH = [50, 68]
_C.TRAIN.LR_RATE = 0.1
_C.TRAIN.WARMUP_EPOCH = 5
_C.TRAIN.MIN_LR = 1e-3
_C.TRAIN.WEIGHTS_DECAY = 5e-4
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.PRETRAIN = True
_C.TRAIN.SAVE_STEPS = 2000
_C.TRAIN.CHECKPOINT_DIR = './checkpoints/'
_C.TRAIN.IMS_PER_BATCH = 2
_C.TRAIN.MAX_ITER = 1000
_C.TRAIN.RESUME = False
_C.TRAIN.AMP = False

# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()

_C.DATALOADER.SAMPLER_TRAIN = 'TrainingSampler'
_C.DATALOADER.TRAIN_SHUFFLE = True
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.WEIGHT_DECAY = 0.0001
# Weight decay of norm layers.
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0
# Bias.
_C.SOLVER.BIAS_LR_FACTOR = 2.0
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.OPTIMIZER = 'sgd'
_C.SOLVER.ADAM_BETAS = (0.9, 0.999)
_C.SOLVER.ADAM_EPS = 1e-08

_C.SOLVER.LR_SCHEDULER_NAME = 'WarmupPolyLR'
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000, )
_C.SOLVER.GAMMA = 0.1

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.POLY_LR_POWER = 0.9
_C.SOLVER.POLY_LR_CONSTANT_ENDING = 0

_C.SOLVER.CLIP_GRADIENTS = CN()
_C.SOLVER.CLIP_GRADIENTS.ENABLED = False
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
