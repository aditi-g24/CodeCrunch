"""
Model architectures for semantic segmentation
Implements SegFormer and DeepLabV3+ with pretrained backbones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import segmentation_models_pytorch as smp

from config import Config


def get_model(num_classes: int, device: str = 'cpu') -> nn.Module:
    """
    Factory function to create the segmentation model
    
    Using SegFormer by default for better generalization:
    - SegFormer uses hierarchical Transformer encoder (MiT)
    - Better at capturing global context
    - More robust to domain shift than pure CNNs
    - Efficient and accurate
    
    Alternative: DeepLabV3+ with ResNet backbone
    - Strong CNN baseline
    - ASPP module for multi-scale features
    - Good performance but may overfit more on limited data
    
    Args:
        num_classes: Number of segmentation classes
        device: Device to load model on
    
    Returns:
        model: Segmentation model
    """
    
    if Config.MODEL_NAME.lower() == "segformer":
        # SegFormer - Recommended for better generalization
        # MiT (Mix Transformer) backbone with hierarchical features
        # Papers: "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
        
        model = smp.SegFormer(
            encoder_name=Config.ENCODER_NAME,  # mit_b0 to mit_b5
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=3,
            classes=num_classes,
        )
        
        print(f"Created SegFormer model with {Config.ENCODER_NAME} encoder")
        print(f"SegFormer chosen for: better global context, robustness to domain shift")
        
    elif Config.MODEL_NAME.lower() == "deeplabv3plus":
        # DeepLabV3+ - Strong CNN baseline
        # Uses ASPP (Atrous Spatial Pyramid Pooling) for multi-scale features
        # Good baseline but may require more regularization
        
        model = smp.DeepLabV3Plus(
            encoder_name=Config.ENCODER_NAME,  # resnet50, resnet101, etc.
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=3,
            classes=num_classes,
        )
        
        print(f"Created DeepLabV3+ model with {Config.ENCODER_NAME} encoder")
        print(f"DeepLabV3+ uses ASPP for multi-scale feature extraction")
        
    else:
        raise ValueError(f"Unknown model: {Config.MODEL_NAME}")
    
    # Move model to device
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for better generalization
    Can be used during inference for improved predictions
    """
    
    def __init__(self, models: list):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average predictions from all models"""
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(F.softmax(pred, dim=1))
        
        # Average probabilities
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        return ensemble_pred


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = 'cpu'
) -> nn.Module:
    """
    Load model from checkpoint
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        checkpoint_path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
