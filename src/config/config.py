from yacs.config import CfgNode as CN


_C = CN()

_C.ACCELERATOR = CN()
_C.ACCELERATOR.SPLIT_BATCHES = False
_C.ACCELERATOR.ACCUMULATION_STEPS = 4
_C.ACCELERATOR.MIXED_PRECISION = 'no'
_C.ACCELERATOR.LOG_WITH = 'wandb'

_C.OPTIMIZER = CN()
_C.OPTIMIZER.LEARNING_RATE = 1e-4
_C.OPTIMIZER.ADAM_BETA1 = 0.9
_C.OPTIMIZER.ADAM_BETA2 = 0.999
_C.OPTIMIZER.ADAM_WEIGHT_DECAY = 0
_C.OPTIMIZER.ADAM_EPSILON = 1e-8

_C.LORA = CN()
_C.LORA.RANK = 128
_C.LORA.ALPHA = 128
_C.LORA.HIDDEN_SIZE = 2048
_C.LORA.MAX_GRADIENT_NORM = 1

_C.SCHEDULER = CN()
_C.SCHEDULER.NAME = 'constant'
_C.SCHEDULER.WARMUP_STEPS = 0
_C.SCHEDULER.PREDICTION_TYPE = 'epsilon'
_C.SCHEDULER.SNR_GAMMA = -1

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 3
_C.SYSTEM.NUM_WORKERS = 4
_C.SYSTEM.RANDOM_SEED = 2204
_C.SYSTEM.CONFIG_PATH = '/home/mvchaychuk/Diploma_HSE_ImageCLEFmed_2024_VQA_Image_Synthesis/src/config/experiments/'

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 2
_C.TRAIN.VALIDATION_EPOCHS = 1
_C.TRAIN.DATALOADER_NUM_WORKERS = 2
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.IMAGE_RESOLUTION = 512

_C.EXPERIMENT = CN()
_C.EXPERIMENT.PROJECT_NAME = 'Diploma_HSE_ImageCLEFmed_2024_VQA_Image_Synthesis'
_C.EXPERIMENT.NAME = 'Prior_Experiment_1'
_C.EXPERIMENT.LORA_PRIOR_WEIGHTS_OUTPUT_DIR = '/home/mvchaychuk/Diploma_HSE_ImageCLEFmed_2024_VQA_Image_Synthesis/lora_weights_prior'
_C.EXPERIMENT.LORA_DECODER_WEIGHTS_OUTPUT_DIR = '/home/mvchaychuk/Diploma_HSE_ImageCLEFmed_2024_VQA_Image_Synthesis/lora_weights_decoder'
_C.EXPERIMENT.KANDINSKY3_LORA_WEIGHTS_OUTPUT_DIR = '/home/mvchaychuk/Diploma_HSE_ImageCLEFmed_2024_VQA_Image_Synthesis/kandinky3_lora_weights'
_C.EXPERIMENT.DATASET_TRAIN_PATH = '/home/mvchaychuk/Diploma_HSE_ImageCLEFmed_2024_VQA_Image_Synthesis/data/train'
_C.EXPERIMENT.PROMPTS_FILE_NAME = 'prompt-gt.csv'
_C.EXPERIMENT.DECODER_PATH = '/home/mvchaychuk/kandinsky-2-2-decoder'
_C.EXPERIMENT.PRIOR_PATH = '/home/mvchaychuk/kandinsky-2-2-prior'
_C.EXPERIMENT.KANDINSKY3_PATH = '/home/mvchaychuk/kandinsky-3'
_C.EXPERIMENT.VALIDATION_PROMPT = "generate an image containing a polyp"

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()