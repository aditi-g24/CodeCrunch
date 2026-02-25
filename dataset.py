"""
Dataset module with strong augmentation pipeline for generalization
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional, List
import random

from config import Config


class OffroadSegmentationDataset(Dataset):
    """Dataset for offroad semantic segmentation"""
    
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Optional[Path] = None,
        transform: Optional[A.Compose] = None,
        is_test: bool = False
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.is_test = is_test
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob("*.png")) + 
                                 list(self.image_dir.glob("*.jpg")) +
                                 list(self.image_dir.glob("*.jpeg")))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        print(f"Found {len(self.image_files)} images in {self.image_dir}")
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.is_test:
            # Test mode: only apply normalization and convert to tensor
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            return image, str(img_path.name)
        
        # Load mask
        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            # Try with different extension
            mask_path = self.mask_dir / (img_path.stem + ".png")
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Convert mask to long tensor for CrossEntropyLoss
        mask = torch.from_numpy(mask).long()
        
        return image, mask


def get_train_transform() -> A.Compose:
    """
    Strong augmentation pipeline for training
    Optimized for generalization to unseen desert environments
    """
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=Config.ROTATION_LIMIT, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=Config.SCALE_LIMIT,
            rotate_limit=Config.ROTATION_LIMIT,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        
        # Color augmentations - critical for domain adaptation
        A.OneOf([
            A.ColorJitter(
                brightness=Config.BRIGHTNESS_LIMIT,
                contrast=Config.CONTRAST_LIMIT,
                saturation=Config.SATURATION_LIMIT,
                hue=Config.HUE_LIMIT,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(Config.HUE_LIMIT * 180),
                sat_shift_limit=int(Config.SATURATION_LIMIT * 100),
                val_shift_limit=int(Config.BRIGHTNESS_LIMIT * 100),
                p=1.0
            ),
        ], p=Config.AUG_PROB),
        
        # Additional color variations
        A.RandomBrightnessContrast(
            brightness_limit=Config.BRIGHTNESS_LIMIT,
            contrast_limit=Config.CONTRAST_LIMIT,
            p=0.5
        ),
        
        # Lighting variations (important for desert scenes)
        A.OneOf([
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.RandomToneCurve(scale=0.1, p=1.0),
        ], p=0.3),
        
        # Noise and blur for robustness
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        
        # Weather augmentations (useful for desert environments)
        A.RandomShadow(p=0.2),
        
        # Normalization and conversion
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


def get_val_transform() -> A.Compose:
    """Validation transform - only normalization"""
    return A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


def get_test_transform() -> A.Compose:
    """Test transform - only normalization"""
    return A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


def compute_class_weights(train_loader: DataLoader, num_classes: int) -> torch.Tensor:
    """
    Compute class weights based on pixel frequency in training set
    This helps handle class imbalance common in segmentation tasks
    """
    print("Computing class weights from training data...")
    class_counts = torch.zeros(num_classes, dtype=torch.float64)
    
    for _, masks in train_loader:
        for class_idx in range(num_classes):
            class_counts[class_idx] += (masks == class_idx).sum().item()
    
    # Avoid division by zero
    class_counts = class_counts + 1e-6
    
    # Compute inverse frequency weights
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print("Class weights:")
    for i, (name, weight) in enumerate(zip(Config.CLASS_NAMES[:num_classes], class_weights)):
        print(f"  {name}: {weight:.4f} (pixels: {int(class_counts[i])})")
    
    return class_weights.float()


def get_dataloaders(
    compute_weights: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor]]:
    """
    Create train and validation dataloaders
    
    Returns:
        train_loader, val_loader, class_weights
    """
    # Create datasets
    train_dataset = OffroadSegmentationDataset(
        image_dir=Config.TRAIN_IMAGE_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        transform=get_train_transform(),
        is_test=False
    )
    
    val_dataset = OffroadSegmentationDataset(
        image_dir=Config.VAL_IMAGE_DIR,
        mask_dir=Config.VAL_MASK_DIR,
        transform=get_val_transform(),
        is_test=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    # Compute class weights
    class_weights = None
    if compute_weights:
        class_weights = compute_class_weights(train_loader, Config.NUM_CLASSES)
    
    return train_loader, val_loader, class_weights


def get_test_dataloader() -> DataLoader:
    """Create test dataloader"""
    test_dataset = OffroadSegmentationDataset(
        image_dir=Config.TEST_IMAGE_DIR,
        mask_dir=None,
        transform=get_test_transform(),
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for TTA
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    return test_loader


def set_seed(seed: int = Config.SEED):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
