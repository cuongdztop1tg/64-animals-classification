import torch
import torch.nn as nn

from torchmetrics.classification import MulticlassAccuracy

# Train configs
RANDOM_SEED = 42
EPOCHS = 150
LR = 1e-3
BATCH_SIZE = 32
NUM_CLASSES = 64
IMG_SIZE = (128, 128)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# result path
RESULT_PATH = "./models"

# data paths
RAW_DATA_PATH = "./data/image"
PROCESSED_DATA_PATH = "./data/processed"
