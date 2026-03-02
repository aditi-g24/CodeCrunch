"""
Configuration file for Offroad Semantic Segmentation
Competition-grade settings optimized for generalization
"""

import torch
import os

class Config:
    # ==================== PATHS ====================
    # Adjust these paths for Google Colab
    TRAIN_IMG_DIR = '/content/Train/images'
    TRAIN_MASK_DIR = '/content/Train/masks'
    VAL_IMG_DIR = '/content/Val/images'
    VAL_MASK_DIR = '/content/Val/masks'
    TEST_IMG_DIR = '/content/testImages'
    
    # Output directories
    CHECKPOINT_DIR = '/content/checkpoints'
    RUNS_DIR = '/content/runs'
    PREDICTIONS_DIR = '/content/predictions'
    
    # ==================== MODEL ====================
    # Architecture choice: SegFormer is preferred for better generalization
    # and global context modeling, crucial for domain shift scenarios
    MODEL_NAME = 'segformer'  # Options: 'segformer', 'deeplabv3plus'
    
    # SegFormer settings
    SEGFORMER_ENCODER = 'mit_b3'  # Options: mit_b0 to mit_b5 (b3 is good balance)
    
    # DeepLabV3+ settings (if using deeplabv3plus)
    DEEPLABV3_BACKBONE = 'resnet101'  # Options: resnet50, resnet101
    
    # ==================== CLASSES ====================
    NUM_CLASSES = 11  # Background + 10 semantic classes
    CLASS_NAMES = [
        'Background',
        'Trees',
        'Lush_Bushes',
        'Dry_Grass',
        'Dry_Bushes',
        'Ground_Clutter',
        'Flowers',
        'Logs',
        'Rocks',
        'Landscape',
        'Sky'
    ]
    
    # Color map for visualization (RGB)
    CLASS_COLORS = [
        [0, 0, 0],        # Background - Black
        [34, 139, 34],    # Trees - Forest Green
        [0, 255, 0],      # Lush_Bushes - Lime
        [255, 215, 0],    # Dry_Grass - Gold
        [210, 180, 140],  # Dry_Bushes - Tan
        [165, 42, 42],    # Ground_Clutter - Brown
        [255, 20, 147],   # Flowers - Deep Pink
        [139, 69, 19],    # Logs - Saddle Brown
        [128, 128, 128],  # Rocks - Gray
        [244, 164, 96],   # Landscape - Sandy Brown
        [135, 206, 235]   # Sky - Sky Blue
    ]
    
    # ==================== TRAINING ====================
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.1  # As requested
    WEIGHT_DECAY = 1e-4
    
    # Image settings
    IMAGE_SIZE = (512, 512)  # (height, width)
    
    # Mixed precision training
    USE_AMP = True
    
    # ==================== AUGMENTATION ====================
    # Strong augmentation for better generalization
    AUG_PROB = 0.5
    ROTATION_LIMIT = 15
    BRIGHTNESS_LIMIT = 0.2
    CONTRAST_LIMIT = 0.2
    SATURATION_LIMIT = 0.2
    HUE_LIMIT = 0.1
    SCALE_LIMIT = 0.15
    GAUSSIAN_NOISE_VAR = (10.0, 50.0)
    
    # ==================== OPTIMIZER & SCHEDULER ====================
    OPTIMIZER = 'adamw'  # Options: 'adam', 'adamw', 'sgd'
    
    # Scheduler settings
    SCHEDULER = 'cosine'  # Options: 'cosine', 'reducelr'
    T_MAX = NUM_EPOCHS  # For CosineAnnealing
    ETA_MIN = 1e-6  # Minimum learning rate
    
    # ReduceLROnPlateau settings
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    
    # ==================== EARLY STOPPING ====================
    EARLY_STOPPING_PATIENCE = 15
    MIN_DELTA = 1e-4
    
    # ==================== LOSS FUNCTION ====================
    # Hybrid loss: 0.5 * CrossEntropy + 0.5 * Dice Loss
    CE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5
    USE_CLASS_WEIGHTS = True  # Computed from training set
    DICE_SMOOTH = 1.0
    
    # ==================== TEST-TIME AUGMENTATION ====================
    USE_TTA = True  # Horizontal flip averaging
    
    # ==================== MISC ====================
    NUM_WORKERS = 2  # DataLoader workers
    PIN_MEMORY = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_PREDICTIONS = True
    
    @staticmethod
    def create_dirs():
        """Create necessary directories"""
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.RUNS_DIR, exist_ok=True)
        os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)
    
    @staticmethod
    def print_config():
        """Print configuration"""
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"Model: {Config.MODEL_NAME}")
        print(f"Device: {Config.DEVICE}")
        print(f"Image Size: {Config.IMAGE_SIZE}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Epochs: {Config.NUM_EPOCHS}")
        print(f"Classes: {Config.NUM_CLASSES}")
        print(f"Use AMP: {Config.USE_AMP}")
        print(f"Use TTA: {Config.USE_TTA}")
        print("=" * 60)
