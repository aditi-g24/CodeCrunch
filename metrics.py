"""
Metrics module for Offroad Semantic Segmentation
Implements IoU, Precision, Recall, and Confusion Matrix
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from config import Config


class SegmentationMetrics:
    """
    Comprehensive metrics for semantic segmentation evaluation
    Computes per-class and mean IoU, Precision, Recall, and Confusion Matrix
    """
    
    def __init__(self, num_classes, class_names=None):
        """
        Args:
            num_classes: Number of classes
            class_names: List of class names for display
        """
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0
    
    def update(self, predictions, targets):
        """
        Update metrics with a batch of predictions and targets
        
        Args:
            predictions: (B, H, W) predicted class indices
            targets: (B, H, W) ground truth class indices
        """
        # Convert to numpy
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Flatten arrays
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Update confusion matrix
        # Filter out invalid labels (if any)
        mask = (targets >= 0) & (targets < self.num_classes)
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Compute confusion matrix for this batch
        cm = sklearn_confusion_matrix(
            targets,
            predictions,
            labels=np.arange(self.num_classes)
        )
        
        self.confusion_matrix += cm
        self.total_samples += len(targets)
    
    def get_confusion_matrix(self):
        """Get the confusion matrix"""
        return self.confusion_matrix
    
    def get_iou_per_class(self):
        """
        Compute IoU for each class
        
        Returns:
            numpy array of per-class IoU
        """
        # IoU = TP / (TP + FP + FN)
        # TP: diagonal of confusion matrix
        # FP: sum of column - TP
        # FN: sum of row - TP
        
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        # Avoid division by zero
        denominator = tp + fp + fn
        iou = np.zeros(self.num_classes)
        
        valid = denominator > 0
        iou[valid] = tp[valid] / denominator[valid]
        
        return iou
    
    def get_mean_iou(self):
        """
        Compute mean IoU across all classes
        Only includes classes that appear in the ground truth
        """
        iou_per_class = self.get_iou_per_class()
        
        # Only average over classes that appear in ground truth
        class_appears = self.confusion_matrix.sum(axis=1) > 0
        
        if class_appears.sum() > 0:
            mean_iou = iou_per_class[class_appears].mean()
        else:
            mean_iou = 0.0
        
        return mean_iou
    
    def get_precision_per_class(self):
        """
        Compute Precision for each class
        Precision = TP / (TP + FP)
        """
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        
        denominator = tp + fp
        precision = np.zeros(self.num_classes)
        
        valid = denominator > 0
        precision[valid] = tp[valid] / denominator[valid]
        
        return precision
    
    def get_recall_per_class(self):
        """
        Compute Recall for each class
        Recall = TP / (TP + FN)
        """
        tp = np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        denominator = tp + fn
        recall = np.zeros(self.num_classes)
        
        valid = denominator > 0
        recall[valid] = tp[valid] / denominator[valid]
        
        return recall
    
    def get_f1_per_class(self):
        """
        Compute F1 score for each class
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        precision = self.get_precision_per_class()
        recall = self.get_recall_per_class()
        
        f1 = np.zeros(self.num_classes)
        denominator = precision + recall
        
        valid = denominator > 0
        f1[valid] = 2 * (precision[valid] * recall[valid]) / denominator[valid]
        
        return f1
    
    def get_pixel_accuracy(self):
        """
        Compute overall pixel accuracy
        Accuracy = (TP for all classes) / Total pixels
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        
        if total > 0:
            return correct / total
        else:
            return 0.0
    
    def print_metrics(self):
        """Print comprehensive metrics report"""
        iou_per_class = self.get_iou_per_class()
        precision_per_class = self.get_precision_per_class()
        recall_per_class = self.get_recall_per_class()
        f1_per_class = self.get_f1_per_class()
        mean_iou = self.get_mean_iou()
        pixel_acc = self.get_pixel_accuracy()
        
        print("\n" + "=" * 80)
        print("SEGMENTATION METRICS")
        print("=" * 80)
        
        print(f"\nOverall Metrics:")
        print(f"  Mean IoU:         {mean_iou:.4f}")
        print(f"  Pixel Accuracy:   {pixel_acc:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print("-" * 80)
        print(f"{'Class':<20} {'IoU':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
        print("-" * 80)
        
        for i in range(self.num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
            # Only print if class appears in ground truth
            if self.confusion_matrix[i].sum() > 0:
                print(f"{class_name:<20} {iou_per_class[i]:>10.4f} {precision_per_class[i]:>12.4f} "
                      f"{recall_per_class[i]:>10.4f} {f1_per_class[i]:>10.4f}")
        
        print("-" * 80)
        print()
    
    def get_metrics_dict(self):
        """
        Return metrics as a dictionary
        Useful for logging and tracking
        """
        iou_per_class = self.get_iou_per_class()
        precision_per_class = self.get_precision_per_class()
        recall_per_class = self.get_recall_per_class()
        
        metrics = {
            'mean_iou': self.get_mean_iou(),
            'pixel_accuracy': self.get_pixel_accuracy(),
            'iou_per_class': iou_per_class,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class
        }
        
        # Add per-class metrics with names
        for i, class_name in enumerate(self.class_names):
            metrics[f'iou_{class_name}'] = iou_per_class[i]
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
        
        return metrics


def compute_iou_batch(predictions, targets, num_classes):
    """
    Fast computation of IoU for a single batch
    
    Args:
        predictions: (B, H, W) predicted class indices
        targets: (B, H, W) ground truth class indices
        num_classes: Number of classes
    
    Returns:
        Mean IoU for the batch
    """
    batch_size = predictions.shape[0]
    ious = []
    
    for b in range(batch_size):
        pred = predictions[b].flatten()
        target = targets[b].flatten()
        
        iou_per_class = []
        for c in range(num_classes):
            pred_c = (pred == c)
            target_c = (target == c)
            
            intersection = (pred_c & target_c).sum().item()
            union = (pred_c | target_c).sum().item()
            
            if union > 0:
                iou_per_class.append(intersection / union)
        
        if len(iou_per_class) > 0:
            ious.append(np.mean(iou_per_class))
    
    return np.mean(ious) if len(ious) > 0 else 0.0


def save_confusion_matrix(confusion_matrix, class_names, save_path):
    """
    Save confusion matrix as image
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the image
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-10)
    
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")
