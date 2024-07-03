CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

PRETRAIN = 'pretrain'
FINETUNE = 'finetune'
INFERENCE = 'inference'
STAGES = [PRETRAIN, FINETUNE, INFERENCE]

FT_LOCAL = 'finetune-local'
FT_GLOBAL = 'finetune-global'
FT_NOGATE = 'finetune-nogate'
IFC_LOCAL = 'inference-local'
IFC_GLOBAL = 'inference-global'
IFC_NOGATE = 'inference-nogate'
IFC_FIX = 'inference-fix'

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

N_INDICATOR_TOKEN = 1
N_IMAGE_TOKEN = 576
# N_IMAGE_TOKEN = 729