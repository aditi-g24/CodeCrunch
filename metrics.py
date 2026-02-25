"""
Comprehensive metrics for semantic segmentation evaluation
Includes per-class IoU, mean IoU, confusion matrix, precision, and recall
"""

import torch
import numpy as np
from typing import Tuple, Dict, List
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class SegmentationMetrics:
    """
    Comprehensive metrics for semantic segmentation
    """
    
    def __init__(self, num_classes: int, class_names: List[str], device: str = 'cpu'):
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device=self.device
        )
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix with new predictions
        
        Args:
            predictions: (B, C, H, W) logits or (B, H, W) class indices
            targets: (B, H, W) class indices
        """
        # Convert predictions to class indices if needed
        if predictions.dim() == 4:  # (B, C, H, W) logits
            predictions = torch.argmax(predictions, dim=1)  # (B, H, W)
        
        # Flatten
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Filter out ignore index if present
        valid_mask = (targets >= 0) & (targets < self.num_classes)
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        # Update confusion matrix
        for t, p in zip(targets, predictions):
            self.confusion_matrix[t.long(), p.long()] += 1
    
    def compute_iou(self) -> Tuple[torch.Tensor, float]:
        """
        Compute per-class IoU and mean IoU
        
        Returns:
            per_class_iou: (num_classes,) tensor
            mean_iou: float
        """
        # True Positives (diagonal elements)
        tp = torch.diag(self.confusion_matrix)
        
        # False Positives (column sum - TP)
        fp = self.confusion_matrix.sum(dim=0) - tp
        
        # False Negatives (row sum - TP)
        fn = self.confusion_matrix.sum(dim=1) - tp
        
        # IoU = TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn + 1e-6)
        
        # Mean IoU (excluding classes with no ground truth)
        valid_classes = (tp + fn) > 0
        mean_iou = iou[valid_classes].mean().item()
        
        return iou, mean_iou
    
    def compute_precision_recall(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-class precision and recall
        
        Returns:
            precision: (num_classes,) tensor
            recall: (num_classes,) tensor
        """
        # True Positives
        tp = torch.diag(self.confusion_matrix)
        
        # False Positives
        fp = self.confusion_matrix.sum(dim=0) - tp
        
        # False Negatives
        fn = self.confusion_matrix.sum(dim=1) - tp
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp + 1e-6)
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn + 1e-6)
        
        return precision, recall
    
    def compute_pixel_accuracy(self) -> float:
        """
        Compute overall pixel accuracy
        
        Returns:
            accuracy: float
        """
        correct = torch.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        accuracy = (correct / (total + 1e-6)).item()
        return accuracy
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get all metrics as a dictionary
        
        Returns:
            metrics: Dictionary of metric names and values
        """
        iou, mean_iou = self.compute_iou()
        precision, recall = self.compute_precision_recall()
        pixel_acc = self.compute_pixel_accuracy()
        
        metrics = {
            'mean_iou': mean_iou,
            'pixel_accuracy': pixel_acc,
        }
        
        # Add per-class metrics
        for i, name in enumerate(self.class_names):
            if i < len(iou):
                metrics[f'iou_{name}'] = iou[i].item()
                metrics[f'precision_{name}'] = precision[i].item()
                metrics[f'recall_{name}'] = recall[i].item()
        
        return metrics
    
    def print_metrics(self):
        """Print metrics in a formatted way"""
        iou, mean_iou = self.compute_iou()
        precision, recall = self.compute_precision_recall()
        pixel_acc = self.compute_pixel_accuracy()
        
        print("\n" + "="*80)
        print(f"{'SEGMENTATION METRICS':^80}")
        print("="*80)
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Pixel Accuracy: {pixel_acc:.4f}")
        print("\nPer-class metrics:")
        print(f"{'Class':<20} {'IoU':>10} {'Precision':>12} {'Recall':>10}")
        print("-"*80)
        
        for i, name in enumerate(self.class_names):
            if i < len(iou):
                print(f"{name:<20} {iou[i].item():>10.4f} {precision[i].item():>12.4f} {recall[i].item():>10.4f}")
        
        print("="*80 + "\n")
    
    def plot_confusion_matrix(self, save_path: Path):
        """
        Plot and save confusion matrix
        
        Args:
            save_path: Path to save the plot
        """
        cm = self.confusion_matrix.cpu().numpy()
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized,
            annot=False,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Frequency'}
        )
        plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")


def compute_iou_batch(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> Tuple[torch.Tensor, float]:
    """
    Compute IoU for a single batch (quick computation)
    
    Args:
        predictions: (B, C, H, W) logits or (B, H, W) class indices
        targets: (B, H, W) class indices
        num_classes: Number of classes
    
    Returns:
        per_class_iou: (num_classes,) tensor
        mean_iou: float
    """
    # Convert predictions to class indices if needed
    if predictions.dim() == 4:
        predictions = torch.argmax(predictions, dim=1)
    
    iou_list = []
    
    for cls in range(num_classes):
        pred_mask = (predictions == cls)
        target_mask = (targets == cls)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union == 0:
            iou_list.append(torch.tensor(0.0, device=predictions.device))
        else:
            iou_list.append(intersection / union)
    
    iou_tensor = torch.stack(iou_list)
    mean_iou = iou_tensor.mean().item()
    
    return iou_tensor, mean_iou


class MetricTracker:
    """
    Track metrics over training epochs
    """
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mean_iou': [],
            'val_pixel_acc': [],
            'learning_rate': []
        }
        self.best_mean_iou = 0.0
        self.best_epoch = 0
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update history with new metrics"""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        
        # Track best model
        if 'val_mean_iou' in metrics:
            if metrics['val_mean_iou'] > self.best_mean_iou:
                self.best_mean_iou = metrics['val_mean_iou']
                self.best_epoch = epoch
    
    def plot_history(self, save_path: Path):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        if 'train_loss' in self.history and 'val_loss' in self.history:
            axes[0, 0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
            axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Mean IoU
        if 'val_mean_iou' in self.history:
            axes[0, 1].plot(self.history['val_mean_iou'], label='Val Mean IoU', 
                           color='green', linewidth=2)
            axes[0, 1].axhline(y=self.best_mean_iou, color='r', linestyle='--', 
                              label=f'Best: {self.best_mean_iou:.4f}')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Mean IoU')
            axes[0, 1].set_title('Validation Mean IoU')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Pixel Accuracy
        if 'val_pixel_acc' in self.history:
            axes[1, 0].plot(self.history['val_pixel_acc'], label='Val Pixel Acc', 
                           color='purple', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Pixel Accuracy')
            axes[1, 0].set_title('Validation Pixel Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'learning_rate' in self.history:
            axes[1, 1].plot(self.history['learning_rate'], label='Learning Rate', 
                           color='orange', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training history saved to {save_path}")
