"""
Training script for Offroad Semantic Segmentation Hackathon

Optimized for maximum IoU and generalization to unseen desert environments
Implements:
- SegFormer/DeepLabV3+ with pretrained backbone
- Hybrid loss (CrossEntropy + Dice)
- Class weights for imbalance
- Strong augmentation pipeline
- Mixed precision training
- Learning rate scheduling
- Early stopping
- Comprehensive metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from config import Config
from dataset import get_dataloaders, set_seed
from model import get_model, save_checkpoint
from loss import get_loss_function
from metrics import SegmentationMetrics, MetricTracker


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_metric: float) -> bool:
        """
        Check if training should stop
        
        Args:
            val_metric: Validation metric (higher is better)
        
        Returns:
            True if should stop, False otherwise
        """
        score = val_metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: GradScaler,
    epoch: int
) -> float:
    """
    Train for one epoch
    
    Returns:
        Average training loss
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if Config.USE_AMP and device != 'cpu':
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        # Update progress bar
        if batch_idx % Config.LOG_INTERVAL == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int,
    class_names: list
) -> tuple:
    """
    Validate the model
    
    Returns:
        avg_loss, metrics_dict, seg_metrics
    """
    model.eval()
    running_loss = 0.0
    
    # Initialize metrics
    seg_metrics = SegmentationMetrics(num_classes, class_names, device=device)
    
    pbar = tqdm(val_loader, desc="Validation")
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        running_loss += loss.item()
        
        # Update metrics
        seg_metrics.update(outputs, masks)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(val_loader)
    metrics_dict = seg_metrics.get_metrics()
    
    return avg_loss, metrics_dict, seg_metrics


def train():
    """Main training function"""
    
    # Create output directories
    Config.create_dirs()
    
    # Set random seed for reproducibility
    set_seed(Config.SEED)
    
    # Device
    device = Config.DEVICE
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Get dataloaders and class weights
    print("\n" + "="*80)
    print("Loading datasets...")
    print("="*80)
    train_loader, val_loader, class_weights = get_dataloaders(compute_weights=True)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    # Create model
    print("\n" + "="*80)
    print("Creating model...")
    print("="*80)
    model = get_model(num_classes=Config.NUM_CLASSES, device=device)
    
    # Loss function
    criterion = get_loss_function(class_weights=class_weights)
    print(f"\nLoss function: {Config.CE_WEIGHT} * CrossEntropy + {Config.DICE_WEIGHT} * Dice")
    
    # Optimizer
    if Config.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
    elif Config.OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {Config.OPTIMIZER}")
    
    print(f"Optimizer: {Config.OPTIMIZER}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    print(f"Weight decay: {Config.WEIGHT_DECAY}")
    
    # Learning rate scheduler
    if Config.SCHEDULER.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=Config.NUM_EPOCHS,
            eta_min=Config.LR_MIN
        )
        print(f"Scheduler: CosineAnnealingLR (min_lr={Config.LR_MIN})")
    elif Config.SCHEDULER.lower() == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=Config.FACTOR,
            patience=Config.PATIENCE,
            verbose=True
        )
        print(f"Scheduler: ReduceLROnPlateau (patience={Config.PATIENCE})")
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = GradScaler() if Config.USE_AMP and device.type == 'cuda' else None
    if Config.USE_AMP and device.type == 'cuda':
        print("Mixed precision training: Enabled")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)
    
    # Metric tracker
    tracker = MetricTracker()
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    print(f"Total epochs: {Config.NUM_EPOCHS}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("="*80 + "\n")
    
    best_mean_iou = 0.0
    
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}")
        print("-" * 80)
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if epoch % Config.VAL_EVERY_N_EPOCHS == 0:
            val_loss, val_metrics, seg_metrics = validate(
                model, val_loader, criterion, device,
                Config.NUM_CLASSES, Config.CLASS_NAMES
            )
            
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Mean IoU: {val_metrics['mean_iou']:.4f}")
            print(f"Val Pixel Acc: {val_metrics['pixel_accuracy']:.4f}")
            
            # Update tracker
            tracker.update(epoch, {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mean_iou': val_metrics['mean_iou'],
                'val_pixel_acc': val_metrics['pixel_accuracy'],
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Print detailed metrics
            seg_metrics.print_metrics()
            
            # Save best model
            if val_metrics['mean_iou'] > best_mean_iou:
                best_mean_iou = val_metrics['mean_iou']
                save_checkpoint(
                    model, optimizer, epoch, val_metrics,
                    Config.CHECKPOINT_DIR / 'best_model.pth',
                    scheduler
                )
                print(f"✓ New best model saved! Mean IoU: {best_mean_iou:.4f}")
                
                # Save confusion matrix
                seg_metrics.plot_confusion_matrix(
                    Config.LOG_DIR / 'confusion_matrix_best.png'
                )
            
            # Learning rate scheduler step
            if scheduler is not None:
                if Config.SCHEDULER.lower() == 'plateau':
                    scheduler.step(val_metrics['mean_iou'])
                else:
                    scheduler.step()
            
            # Early stopping check
            if early_stopping(val_metrics['mean_iou']):
                print("\n" + "="*80)
                print("Early stopping triggered!")
                print("="*80)
                break
        else:
            # Update tracker with train loss only
            tracker.update(epoch, {
                'train_loss': train_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Step scheduler if not plateau-based
            if scheduler is not None and Config.SCHEDULER.lower() != 'plateau':
                scheduler.step()
        
        # Save last checkpoint
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, {},
                Config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth',
                scheduler
            )
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best Mean IoU: {tracker.best_mean_iou:.4f} (Epoch {tracker.best_epoch})")
    print(f"Best model saved at: {Config.CHECKPOINT_DIR / 'best_model.pth'}")
    
    # Plot training history
    tracker.plot_history(Config.LOG_DIR / 'training_history.png')
    
    # Save training history as JSON
    history_path = Config.LOG_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        history_serializable = {}
        for key, values in tracker.history.items():
            history_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]
        json.dump(history_serializable, f, indent=4)
    print(f"Training history saved to: {history_path}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    train()
