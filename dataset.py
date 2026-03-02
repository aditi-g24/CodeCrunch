"""
Dataset module for Offroad Semantic Segmentation
Implements strong augmentation pipeline for generalization
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config


class OffroadDataset(Dataset):
    """
    Custom dataset for offroad semantic segmentation
    Supports strong augmentation for domain shift robustness
    """
    
    def __init__(self, img_dir, mask_dir, transform=None):
        """
        Args:
            img_dir: Directory containing images
            mask_dir: Directory containing masks
            transform: Albumentations transform pipeline
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get all image filenames
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Loaded {len(self.images)} images from {img_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load mask
        # Assuming mask has same name as image (adjust extension if needed)
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            # Ensure mask is single channel
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
        else:
            # If mask doesn't exist, create dummy mask (for test set)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert mask to long tensor
        mask = mask.long()
        
        return image, mask, img_name


class TestDataset(Dataset):
    """
    Dataset for test images (no masks)
    """
    
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f"Loaded {len(self.images)} test images from {img_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Store original size for later
        original_size = (image.shape[0], image.shape[1])
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, img_name, original_size


def get_train_transform():
    """
    Strong augmentation pipeline for training
    Optimized for generalization to unseen desert environments
    """
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=Config.ROTATION_LIMIT, p=Config.AUG_PROB, border_mode=0),
        A.RandomScale(scale_limit=Config.SCALE_LIMIT, p=Config.AUG_PROB),
        
        # Color augmentations (critical for domain shift)
        A.ColorJitter(
            brightness=Config.BRIGHTNESS_LIMIT,
            contrast=Config.CONTRAST_LIMIT,
            saturation=Config.SATURATION_LIMIT,
            hue=Config.HUE_LIMIT,
            p=Config.AUG_PROB
        ),
        
        # Noise augmentation
        A.GaussNoise(var_limit=Config.GAUSSIAN_NOISE_VAR, p=0.3),
        
        # Resize to target size
        A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        ToTensorV2()
    ])


def get_val_transform():
    """
    Validation transform (no augmentation, only resize and normalize)
    """
    return A.Compose([
        A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def compute_class_weights(train_loader, num_classes):
    """
    Compute class weights based on pixel frequency in training set
    Inverse frequency weighting to handle class imbalance
    
    Args:
        train_loader: Training data loader
        num_classes: Number of classes
    
    Returns:
        torch.Tensor: Class weights
    """
    print("Computing class weights from training set...")
    
    class_counts = torch.zeros(num_classes, dtype=torch.float64)
    total_pixels = 0
    
    for _, masks, _ in train_loader:
        for mask in masks:
            # Count pixels for each class
            for c in range(num_classes):
                class_counts[c] += (mask == c).sum().item()
            total_pixels += mask.numel()
    
    # Compute weights: inverse of frequency
    class_frequencies = class_counts / total_pixels
    
    # Avoid division by zero
    class_frequencies = torch.clamp(class_frequencies, min=1e-8)
    
    # Inverse frequency
    class_weights = 1.0 / class_frequencies
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print("\nClass distribution and weights:")
    print("-" * 60)
    for i in range(num_classes):
        class_name = Config.CLASS_NAMES[i] if i < len(Config.CLASS_NAMES) else f"Class {i}"
        print(f"{class_name:20s}: {class_frequencies[i].item()*100:6.2f}% | Weight: {class_weights[i].item():.4f}")
    print("-" * 60)
    
    return class_weights.float()


def get_data_loaders():
    """
    Create train and validation data loaders
    
    Returns:
        train_loader, val_loader, class_weights
    """
    # Create datasets
    train_dataset = OffroadDataset(
        img_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        transform=get_train_transform()
    )
    
    val_dataset = OffroadDataset(
        img_dir=Config.VAL_IMG_DIR,
        mask_dir=Config.VAL_MASK_DIR,
        transform=get_val_transform()
    )
    
    # Create data loaders
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
    if Config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_loader, Config.NUM_CLASSES)
    else:
        class_weights = None
    
    return train_loader, val_loader, class_weights


def get_test_loader():
    """
    Create test data loader
    """
    test_dataset = TestDataset(
        img_dir=Config.TEST_IMG_DIR,
        transform=get_val_transform()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for TTA
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    return test_loader
