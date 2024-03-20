from yacs.config import CfgNode as CN


_C = CN()

_C.ACCELERATOR = CN()
_C.ACCELERATOR.SPLIT_BATCHES = False
_C.ACCELERATOR.ACCUMULATION_STEPS = 1
_C.ACCELERATOR.MIXED_PRECISION = 'no'
_C.ACCELERATOR.LOG_WITH = 'wandb'

_C.OPTIMIZER = CN()
_C.OPTIMIZER.LEARNING_RATE = 1e-4
_C.OPTIMIZER.ADAM_BETA1 = 0.9
_C.OPTIMIZER.ADAM_BETA2 = 0.999
_C.OPTIMIZER.ADAM_WEIGHT_DECAY = 0
_C.OPTIMIZER.ADAM_EPSILON = 1e-8

_C.LORA = CN()
_C.LORA.RANK = 2
_C.LORA.HIDDEN_SIZE = 8
_C.LORA.MAX_GRADIENT_NORM = 0.25

_C.SCHEDULER = CN()
_C.SCHEDULER.NAME = 'constant'
_C.SCHEDULER.WARMUP_STEPS = 0

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 8
_C.SYSTEM.NUM_WORKERS = 4
_C.SYSTEM.RANDOM_SEED = 2204

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 2
_C.TRAIN.DATALOADER_NUM_WORKERS = 4
_C.TRAIN.BATCH_SIZE = 32

_C.EXPERIMENT = CN()
_C.EXPERIMENT.PROJECT_NAME = 'Example_project'
_C.EXPERIMENT.NAME = 'example_experiment'
_C.EXPERIMENT.LORA_WEIGHTS_OUTPUT_DIR = '../lora_weights'
_C.EXPERIMENT.DATASET_TRAIN_PATH = '../data/train'
_C.EXPERIMENT.PROMPTS_FILE_NAME = 'prompt-gt.csv'

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()