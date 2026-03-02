"""
Training script for Offroad Semantic Segmentation
Competition-grade training pipeline with all optimizations
"""

import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import Config
from dataset import get_data_loaders
from model import get_model, print_model_info
from loss import get_loss_function
from metrics import SegmentationMetrics, save_confusion_matrix


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like IoU, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0


class Trainer:
    """
    Complete training pipeline with all features:
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Comprehensive logging
    """
    
    def __init__(self):
        # Set seed
        set_seed(Config.SEED)
        
        # Create directories
        Config.create_dirs()
        
        # Print configuration
        Config.print_config()
        
        # Get data loaders
        print("\n" + "=" * 60)
        print("Loading datasets...")
        print("=" * 60)
        self.train_loader, self.val_loader, self.class_weights = get_data_loaders()
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        
        # Create model
        print("\n" + "=" * 60)
        print("Creating model...")
        print("=" * 60)
        self.model = get_model(Config.NUM_CLASSES)
        self.model = self.model.to(Config.DEVICE)
        print_model_info(self.model)
        
        # Loss function
        self.criterion = get_loss_function(self.class_weights)
        
        # Optimizer
        self.optimizer = self.get_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self.get_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if Config.USE_AMP else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.MIN_DELTA,
            mode='max'
        )
        
        # Metrics
        self.train_metrics = SegmentationMetrics(Config.NUM_CLASSES, Config.CLASS_NAMES)
        self.val_metrics = SegmentationMetrics(Config.NUM_CLASSES, Config.CLASS_NAMES)
        
        # Training state
        self.current_epoch = 0
        self.best_val_iou = 0.0
        self.train_history = []
        self.val_history = []
    
    def get_optimizer(self):
        """Create optimizer"""
        if Config.OPTIMIZER == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY
            )
        elif Config.OPTIMIZER == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY
            )
        elif Config.OPTIMIZER == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=Config.LEARNING_RATE,
                momentum=0.9,
                weight_decay=Config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unknown optimizer: {Config.OPTIMIZER}")
        
        print(f"Optimizer: {Config.OPTIMIZER}")
        return optimizer
    
    def get_scheduler(self):
        """Create learning rate scheduler"""
        if Config.SCHEDULER == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=Config.T_MAX,
                eta_min=Config.ETA_MIN
            )
            print(f"Scheduler: CosineAnnealingLR (T_max={Config.T_MAX}, eta_min={Config.ETA_MIN})")
        elif Config.SCHEDULER == 'reducelr':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=Config.LR_FACTOR,
                patience=Config.LR_PATIENCE,
                verbose=True
            )
            print(f"Scheduler: ReduceLROnPlateau (patience={Config.LR_PATIENCE}, factor={Config.LR_FACTOR})")
        else:
            scheduler = None
            print("No scheduler")
        
        return scheduler
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        running_loss = 0.0
        running_ce_loss = 0.0
        running_dice_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{Config.NUM_EPOCHS} [Train]")
        
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if Config.USE_AMP:
                with autocast():
                    outputs = self.model(images)
                    loss, ce_loss, dice_loss = self.criterion(outputs, masks)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss, ce_loss, dice_loss = self.criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
            # Update metrics
            self.train_metrics.update(predictions, masks)
            
            # Update running losses
            running_loss += loss.item()
            running_ce_loss += ce_loss.item()
            running_dice_loss += dice_loss.item()
            
            # Update progress bar
            if (batch_idx + 1) % Config.LOG_INTERVAL == 0:
                avg_loss = running_loss / (batch_idx + 1)
                avg_ce = running_ce_loss / (batch_idx + 1)
                avg_dice = running_dice_loss / (batch_idx + 1)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ce': f'{avg_ce:.4f}',
                    'dice': f'{avg_dice:.4f}',
                    'lr': f'{current_lr:.6f}'
                })
        
        # Compute epoch metrics
        avg_loss = running_loss / len(self.train_loader)
        avg_ce_loss = running_ce_loss / len(self.train_loader)
        avg_dice_loss = running_dice_loss / len(self.train_loader)
        mean_iou = self.train_metrics.get_mean_iou()
        
        return {
            'loss': avg_loss,
            'ce_loss': avg_ce_loss,
            'dice_loss': avg_dice_loss,
            'mean_iou': mean_iou
        }
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        self.val_metrics.reset()
        
        running_loss = 0.0
        running_ce_loss = 0.0
        running_dice_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1}/{Config.NUM_EPOCHS} [Val]  ")
        
        for images, masks, _ in pbar:
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)
            
            # Forward pass
            outputs = self.model(images)
            loss, ce_loss, dice_loss = self.criterion(outputs, masks)
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
            # Update metrics
            self.val_metrics.update(predictions, masks)
            
            # Update running losses
            running_loss += loss.item()
            running_ce_loss += ce_loss.item()
            running_dice_loss += dice_loss.item()
        
        # Compute epoch metrics
        avg_loss = running_loss / len(self.val_loader)
        avg_ce_loss = running_ce_loss / len(self.val_loader)
        avg_dice_loss = running_dice_loss / len(self.val_loader)
        mean_iou = self.val_metrics.get_mean_iou()
        
        return {
            'loss': avg_loss,
            'ce_loss': avg_ce_loss,
            'dice_loss': avg_dice_loss,
            'mean_iou': mean_iou
        }
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_iou': self.best_val_iou,
            'config': {
                'model_name': Config.MODEL_NAME,
                'num_classes': Config.NUM_CLASSES,
                'image_size': Config.IMAGE_SIZE
            }
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(Config.CHECKPOINT_DIR, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(Config.CHECKPOINT_DIR, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with IoU: {self.best_val_iou:.4f}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(Config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train IoU: {train_metrics['mean_iou']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val IoU:   {val_metrics['mean_iou']:.4f}")
            
            # Learning rate scheduling
            if Config.SCHEDULER == 'cosine':
                self.scheduler.step()
            elif Config.SCHEDULER == 'reducelr':
                self.scheduler.step(val_metrics['mean_iou'])
            
            # Save checkpoint
            is_best = val_metrics['mean_iou'] > self.best_val_iou
            if is_best:
                self.best_val_iou = val_metrics['mean_iou']
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            self.early_stopping(val_metrics['mean_iou'])
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"Total time: {elapsed_time/3600:.2f} hours")
        print(f"Best validation IoU: {self.best_val_iou:.4f}")
        
        # Final validation with best model
        self.load_best_model()
        self.final_evaluation()
        
        # Save training history
        self.save_history()
    
    def load_best_model(self):
        """Load best model checkpoint"""
        best_path = os.path.join(Config.CHECKPOINT_DIR, 'best.pth')
        if os.path.exists(best_path):
            print("\nLoading best model for final evaluation...")
            checkpoint = torch.load(best_path, map_location=Config.DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("\nWarning: Best model checkpoint not found")
    
    @torch.no_grad()
    def final_evaluation(self):
        """Final evaluation on validation set"""
        print("\n" + "=" * 60)
        print("Final Evaluation on Validation Set")
        print("=" * 60)
        
        self.model.eval()
        self.val_metrics.reset()
        
        for images, masks, _ in tqdm(self.val_loader, desc="Evaluating"):
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)
            
            outputs = self.model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            self.val_metrics.update(predictions, masks)
        
        # Print comprehensive metrics
        self.val_metrics.print_metrics()
        
        # Save confusion matrix
        cm_path = os.path.join(Config.RUNS_DIR, 'confusion_matrix.png')
        save_confusion_matrix(
            self.val_metrics.get_confusion_matrix(),
            Config.CLASS_NAMES,
            cm_path
        )
    
    def save_history(self):
        """Save training history"""
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'best_val_iou': self.best_val_iou
        }
        
        history_path = os.path.join(Config.RUNS_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"\nTraining history saved to {history_path}")


def main():
    """Main function"""
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
