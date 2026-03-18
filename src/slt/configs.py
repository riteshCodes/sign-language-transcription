import os

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"  # Local
# Common parent folders
BASE_DATA_DIR = os.path.join("data", "How2Sign", "sentence_level")
RGB_FRONT_CLIPS_DIR = os.path.join("rgb_front", "raw_videos")
TEXT_REALIGNED_DIR = os.path.join("text", "en", "raw_text", "re_aligned")

# Data Folder Paths (Sentence-level Videos)
# TRAIN_VIDEO_DIR = os.path.join("data", "raw")  # Local data
VIDEO_DIR = os.path.join(BASE_DATA_DIR, 
                               "train", 
                               RGB_FRONT_CLIPS_DIR)

# CSV Path (Target Sentences)
# TRAIN_CSV_PATH = os.path.join("data", "how2sign_realigned_val.csv")  # LOCAL
TRAIN_CSV_PATH = os.path.join(BASE_DATA_DIR, 
                              "train", 
                              TEXT_REALIGNED_DIR, 
                              "how2sign_realigned_train_subsampled.csv")
VALIDATION_CSV_PATH = os.path.join(BASE_DATA_DIR, 
                                   "val",
                                   TEXT_REALIGNED_DIR, 
                                   "how2sign_realigned_val_subsampled.csv")
TEST_CSV_PATH = VALIDATION_CSV_PATH

# Model checkpoint path
CHECKPOINTS_PATH = os.path.join("checkpoints")


# Training Configurations
# Size and Length should be divisible by 10 (Since we are using batch features extraction based on 10 multiples)
# Maximum number of frames to include per video for feature extraction
# Using FPS = 4 (4*60*1 -> 1 minute video)
MAX_FRAME_SIZE = 360
MAX_TOKEN_LENGTH = 64  # Maximum number of tokens to include in decoder
FRAMES_PER_SECOND = 6  # How2Sign dataset videos around 20-24 FPS


DINOV2_DIM = 1024
LR = 1e-4 # 1e-3 is ideal for SmolLM2 and 1e-4 for BART
ETA_MIN = 1e-7 # 1e-5 for SmolLM2 and 1e-7 for BART
WARM_UP_STEPS = 500
EPOCHS = 10

PATIENCE = 6 # Early stopping patience
LOGGING_STEP = 25 # Logging Step

BLEU_N_GRAM = 4 # N-gram for BLEU metric

DINOv2_FEATURE_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536
}
