"""
Test/Inference script for Offroad Semantic Segmentation

Features:
- Load best trained model
- Test-Time Augmentation (TTA) for improved predictions
- Generate prediction masks
- Compute IoU if ground truth available
- Save predictions and visualizations
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from model import get_model, load_checkpoint
from dataset import get_test_dataloader, set_seed
from metrics import SegmentationMetrics


class TestTimeAugmentation:
    """
    Test-Time Augmentation for improved predictions
    Averages predictions over multiple augmented versions of the input
    """
    
    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = device
    
    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict with TTA
        
        Args:
            image: (1, C, H, W) input image
        
        Returns:
            prediction: (1, C, H, W) averaged predictions
        """
        predictions = []
        
        # Original image
        pred = self.model(image)
        predictions.append(F.softmax(pred, dim=1))
        
        # Horizontal flip
        image_flipped = torch.flip(image, dims=[3])
        pred_flipped = self.model(image_flipped)
        pred_flipped = torch.flip(pred_flipped, dims=[3])
        predictions.append(F.softmax(pred_flipped, dim=1))
        
        # Average predictions
        avg_pred = torch.stack(predictions).mean(dim=0)
        
        return avg_pred


def visualize_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    class_colors: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create visualization of prediction
    
    Args:
        image: (H, W, 3) RGB image
        prediction: (H, W) class indices
        ground_truth: (H, W) ground truth class indices (optional)
        class_colors: (num_classes, 3) RGB colors for each class
    
    Returns:
        visualization: Combined visualization
    """
    if class_colors is None:
        # Generate random colors for each class
        np.random.seed(42)
        class_colors = np.random.randint(0, 255, size=(Config.NUM_CLASSES, 3))
    
    # Create colored prediction mask
    pred_colored = class_colors[prediction]
    
    # Blend with original image
    alpha = 0.5
    blended = (alpha * image + (1 - alpha) * pred_colored).astype(np.uint8)
    
    if ground_truth is not None:
        # Create colored ground truth mask
        gt_colored = class_colors[ground_truth]
        gt_blended = (alpha * image + (1 - alpha) * gt_colored).astype(np.uint8)
        
        # Stack: original | prediction | ground truth
        visualization = np.hstack([image, blended, gt_blended])
    else:
        # Stack: original | prediction
        visualization = np.hstack([image, blended])
    
    return visualization


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """
    Denormalize image from tensor to numpy array
    
    Args:
        image: (C, H, W) normalized tensor
    
    Returns:
        image: (H, W, C) denormalized numpy array
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = std * image + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image


@torch.no_grad()
def test():
    """Main testing function"""
    
    # Set random seed
    set_seed(Config.SEED)
    
    # Device
    device = Config.DEVICE
    print(f"\nUsing device: {device}")
    
    # Create output directory
    Config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n" + "="*80)
    print("Loading model...")
    print("="*80)
    
    model = get_model(num_classes=Config.NUM_CLASSES, device=device)
    
    # Load best checkpoint
    checkpoint_path = Config.CHECKPOINT_DIR / 'best_model.pth'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = load_checkpoint(model, str(checkpoint_path), device=str(device))
    model.eval()
    
    print(f"Loaded model from: {checkpoint_path}")
    
    # Get test dataloader
    print("\n" + "="*80)
    print("Loading test data...")
    print("="*80)
    test_loader = get_test_dataloader()
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize TTA if enabled
    if Config.USE_TTA:
        print("Test-Time Augmentation: Enabled")
        tta = TestTimeAugmentation(model, device)
    else:
        print("Test-Time Augmentation: Disabled")
    
    # Generate class colors for visualization
    np.random.seed(42)
    class_colors = np.random.randint(0, 255, size=(Config.NUM_CLASSES, 3))
    
    # Create color map legend
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, color) in enumerate(zip(Config.CLASS_NAMES, class_colors)):
        ax.add_patch(plt.Rectangle((0, i), 1, 0.8, color=color/255.0))
        ax.text(1.2, i + 0.4, name, va='center', fontsize=12)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, len(Config.CLASS_NAMES))
    ax.axis('off')
    plt.title('Class Color Map', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Config.PREDICTIONS_DIR / 'color_legend.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Inference loop
    print("\n" + "="*80)
    print("Running inference...")
    print("="*80)
    
    pbar = tqdm(test_loader, desc="Testing")
    
    for batch_idx, (images, filenames) in enumerate(pbar):
        images = images.to(device)
        
        # Get predictions
        if Config.USE_TTA:
            predictions = tta.predict(images)
        else:
            outputs = model(images)
            predictions = F.softmax(outputs, dim=1)
        
        # Convert to class indices
        pred_masks = torch.argmax(predictions, dim=1)  # (B, H, W)
        
        # Process each image in batch
        for i in range(images.size(0)):
            filename = filenames[i] if isinstance(filenames, (list, tuple)) else filenames
            
            # Get image and prediction
            image_tensor = images[i]  # (C, H, W)
            pred_mask = pred_masks[i].cpu().numpy()  # (H, W)
            
            # Denormalize image for visualization
            image_np = denormalize_image(image_tensor)
            
            # Resize prediction to original size if needed
            # (Assuming original size is same as processed size for now)
            
            # Save prediction mask
            mask_path = Config.PREDICTIONS_DIR / f"pred_{filename}"
            cv2.imwrite(str(mask_path), pred_mask.astype(np.uint8))
            
            # Create and save visualization
            vis = visualize_prediction(image_np, pred_mask, class_colors=class_colors)
            vis_path = Config.PREDICTIONS_DIR / f"vis_{filename}"
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            
            pbar.set_postfix({'file': filename})
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETED")
    print("="*80)
    print(f"Predictions saved to: {Config.PREDICTIONS_DIR}")
    print(f"Total images processed: {len(test_loader.dataset)}")
    print("="*80 + "\n")


@torch.no_grad()
def evaluate_on_validation():
    """
    Evaluate the trained model on validation set
    Useful for final performance verification
    """
    
    # Set random seed
    set_seed(Config.SEED)
    
    # Device
    device = Config.DEVICE
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\n" + "="*80)
    print("Loading model for validation evaluation...")
    print("="*80)
    
    model = get_model(num_classes=Config.NUM_CLASSES, device=device)
    checkpoint_path = Config.CHECKPOINT_DIR / 'best_model.pth'
    model = load_checkpoint(model, str(checkpoint_path), device=str(device))
    model.eval()
    
    # Get validation dataloader
    from dataset import get_dataloaders
    _, val_loader, _ = get_dataloaders(compute_weights=False)
    
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize metrics
    seg_metrics = SegmentationMetrics(
        Config.NUM_CLASSES,
        Config.CLASS_NAMES,
        device=str(device)
    )
    
    # Evaluate
    print("\n" + "="*80)
    print("Evaluating on validation set...")
    print("="*80)
    
    pbar = tqdm(val_loader, desc="Validation")
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        if Config.USE_TTA:
            tta = TestTimeAugmentation(model, device)
            outputs = tta.predict(images)
        else:
            outputs = model(images)
        
        # Update metrics
        seg_metrics.update(outputs, masks)
    
    # Print final metrics
    seg_metrics.print_metrics()
    
    # Save confusion matrix
    seg_metrics.plot_confusion_matrix(
        Config.LOG_DIR / 'confusion_matrix_final.png'
    )
    
    # Get metrics
    metrics = seg_metrics.get_metrics()
    
    print("\n" + "="*80)
    print("FINAL VALIDATION METRICS")
    print("="*80)
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print("="*80 + "\n")
    
    return metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--eval':
        # Evaluate on validation set
        evaluate_on_validation()
    else:
        # Run inference on test set
        test()
