"""
Configuration file for Offroad Semantic Segmentation Hackathon
Optimized for maximum IoU and generalization to unseen desert environments
"""

import torch
from pathlib import Path

class Config:
    # Paths
    DATA_ROOT = Path("./data")  # Adjust to your data location
    TRAIN_IMAGE_DIR = DATA_ROOT / "Train" / "images"
    TRAIN_MASK_DIR = DATA_ROOT / "Train" / "masks"
    VAL_IMAGE_DIR = DATA_ROOT / "Val" / "images"
    VAL_MASK_DIR = DATA_ROOT / "Val" / "masks"
    TEST_IMAGE_DIR = DATA_ROOT / "testImages"
    
    # Output directories
    OUTPUT_DIR = Path("./runs")
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
    
    # Model architecture
    # Using SegFormer for better generalization and handling domain shift
    # SegFormer has shown superior performance on diverse datasets with less overfitting
    MODEL_NAME = "segformer"  # Options: "segformer", "deeplabv3plus"
    ENCODER_NAME = "mit_b3"  # For SegFormer: mit_b0 to mit_b5
    # For DeepLabV3+: "resnet50", "resnet101"
    ENCODER_WEIGHTS = "imagenet"
    
    # Dataset
    NUM_CLASSES = 11  # Adjust based on your actual number of classes
    CLASS_NAMES = [
        "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", 
        "Ground Clutter", "Flowers", "Logs", "Rocks", 
        "Landscape", "Sky", "Background"
    ]
    
    # Image size
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    
    # Training hyperparameters
    BATCH_SIZE = 8  # Adjust based on GPU memory
    NUM_EPOCHS = 150
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Loss weights
    CE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5
    
    # Optimizer
    OPTIMIZER = "adamw"  # Options: "adam", "adamw"
    
    # Learning rate scheduler
    SCHEDULER = "cosine"  # Options: "cosine", "plateau"
    LR_MIN = 1e-6  # For cosine annealing
    PATIENCE = 10  # For ReduceLROnPlateau
    FACTOR = 0.5  # For ReduceLROnPlateau
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 25
    
    # Mixed precision training
    USE_AMP = True
    
    # Data augmentation strength
    # Strong augmentation for better generalization
    AUG_PROB = 0.8
    ROTATION_LIMIT = 15
    BRIGHTNESS_LIMIT = 0.2
    CONTRAST_LIMIT = 0.2
    SATURATION_LIMIT = 0.2
    HUE_LIMIT = 0.1
    SCALE_LIMIT = 0.2
    NOISE_VARIANCE = 0.01
    
    # Test-time augmentation
    USE_TTA = True
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Random seed for reproducibility
    SEED = 42
    
    # Validation frequency
    VAL_EVERY_N_EPOCHS = 1
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
