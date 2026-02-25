"""
Hybrid loss function combining CrossEntropy and Dice Loss
Optimized for semantic segmentation with class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation
    Handles class imbalance better than pure CrossEntropy
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) class indices
        
        Returns:
            Dice loss value
        """
        num_classes = predictions.size(1)
        
        # Convert logits to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Flatten spatial dimensions
        predictions = predictions.reshape(predictions.size(0), predictions.size(1), -1)  # (B, C, H*W)
        targets_one_hot = targets_one_hot.reshape(targets_one_hot.size(0), targets_one_hot.size(1), -1)  # (B, C, H*W)
        
        # Compute Dice coefficient for each class
        intersection = (predictions * targets_one_hot).sum(dim=2)  # (B, C)
        union = predictions.sum(dim=2) + targets_one_hot.sum(dim=2)  # (B, C)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)
        
        # Compute loss (1 - Dice)
        dice_loss = 1.0 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class HybridLoss(nn.Module):
    """
    Hybrid loss combining CrossEntropy and Dice Loss
    
    Loss = ce_weight * CrossEntropy + dice_weight * DiceLoss
    
    This combination leverages:
    - CrossEntropy: pixel-wise classification accuracy
    - Dice Loss: region overlap and class imbalance handling
    """
    
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ):
        super(HybridLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # CrossEntropy with class weights
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )
        
        # Dice loss
        self.dice_loss = DiceLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) class indices
        
        Returns:
            Combined loss value
        """
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        
        total_loss = self.ce_weight * ce + self.dice_weight * dice
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Alternative to CrossEntropy for hard examples
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) class indices
        
        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Get probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Factory function to create the loss function
    
    Args:
        class_weights: Class weights for handling imbalance
    
    Returns:
        Loss module
    """
    from config import Config
    
    loss_fn = HybridLoss(
        ce_weight=Config.CE_WEIGHT,
        dice_weight=Config.DICE_WEIGHT,
        class_weights=class_weights
    )
    
    return loss_fn
