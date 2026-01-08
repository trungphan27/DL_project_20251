
import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(BASE_DIR)

# Dataset paths for CelebA-HQ 256x256
DATASET_DIR = os.path.join(DATA_ROOT, "celeba_hq_256", "without_mask")  
MASKED_DIR = os.path.join(DATA_ROOT, "celeba_hq_256", "with_mask")      

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULT_DIR = os.path.join(BASE_DIR, "results")

# Hyperparameters
IMG_SIZE = 256          
BATCH_SIZE = 8         
NUM_EPOCHS = 100        
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002 
BETAS = (0.5, 0.999)
NUM_WORKERS = 2

# Loss Weights 
LAMBDA_L1 = 100
LAMBDA_VGG = 10 
LAMBDA_ADV = 1

# Dataset Split
TRAIN_RATIO = 0.9  

LOAD_MODEL = True 
SAVE_EVERY = 1     

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
