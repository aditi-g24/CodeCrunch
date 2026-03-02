"""
Loss functions for Offroad Semantic Segmentation
Implements hybrid loss: CrossEntropy + Dice Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation
    Optimizes directly for IoU-like metric
    """
    
    def __init__(self, smooth=1.0, num_classes=None):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            num_classes: Number of classes (if None, inferred from input)
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) class indices
        
        Returns:
            Dice loss value
        """
        # Get number of classes
        num_classes = predictions.shape[1] if self.num_classes is None else self.num_classes
        
        # Convert predictions to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Flatten spatial dimensions
        predictions = predictions.contiguous().view(predictions.size(0), predictions.size(1), -1)  # (B, C, H*W)
        targets_one_hot = targets_one_hot.contiguous().view(targets_one_hot.size(0), targets_one_hot.size(1), -1)  # (B, C, H*W)
        
        # Compute Dice coefficient per class
        intersection = (predictions * targets_one_hot).sum(dim=2)  # (B, C)
        union = predictions.sum(dim=2) + targets_one_hot.sum(dim=2)  # (B, C)
        
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)
        
        # Average over batch and classes
        dice_loss = 1.0 - dice_coeff.mean()
        
        return dice_loss


class HybridLoss(nn.Module):
    """
    Hybrid loss combining CrossEntropy and Dice Loss
    Formula: 0.5 * CrossEntropy + 0.5 * Dice Loss
    
    This combination provides:
    - CrossEntropy: Strong per-pixel classification signal
    - Dice Loss: Direct optimization of IoU-like metric
    """
    
    def __init__(self, class_weights=None, ce_weight=0.5, dice_weight=0.5, dice_smooth=1.0):
        """
        Args:
            class_weights: Tensor of class weights for CrossEntropy
            ce_weight: Weight for CrossEntropy loss
            dice_weight: Weight for Dice loss
            dice_smooth: Smoothing factor for Dice loss
        """
        super(HybridLoss, self).__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # CrossEntropy loss with optional class weights
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        # Dice loss
        self.dice_loss = DiceLoss(smooth=dice_smooth)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) class indices
        
        Returns:
            Combined loss value
        """
        # Compute individual losses
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        
        # Combine losses
        total_loss = self.ce_weight * ce + self.dice_weight * dice
        
        return total_loss, ce, dice


def get_loss_function(class_weights=None):
    """
    Factory function to create loss function
    
    Args:
        class_weights: Optional class weights for CrossEntropy
    
    Returns:
        Loss function instance
    """
    if class_weights is not None:
        class_weights = class_weights.to(Config.DEVICE)
    
    loss_fn = HybridLoss(
        class_weights=class_weights,
        ce_weight=Config.CE_WEIGHT,
        dice_weight=Config.DICE_WEIGHT,
        dice_smooth=Config.DICE_SMOOTH
    )
    
    return loss_fn


# Additional loss functions for experimentation

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Can be used as alternative to CrossEntropy
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) class indices
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss
    Useful for handling false positives and false negatives differently
    """
    
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        num_classes = predictions.shape[1]
        predictions = F.softmax(predictions, dim=1)
        
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        # True positives, false positives, false negatives
        tp = (predictions * targets_one_hot).sum(dim=2)
        fp = (predictions * (1 - targets_one_hot)).sum(dim=2)
        fn = ((1 - predictions) * targets_one_hot).sum(dim=2)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1.0 - tversky.mean()
