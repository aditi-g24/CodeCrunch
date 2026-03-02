"""
Test/Inference script for Offroad Semantic Segmentation
Includes Test-Time Augmentation (TTA) for improved predictions
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from dataset import get_test_loader, TestDataset
from model import get_model
from metrics import SegmentationMetrics, save_confusion_matrix


class Predictor:
    """
    Prediction pipeline with optional Test-Time Augmentation
    """
    
    def __init__(self, checkpoint_path):
        """
        Args:
            checkpoint_path: Path to model checkpoint
        """
        print("=" * 60)
        print("Initializing Predictor")
        print("=" * 60)
        
        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        
        # Create model
        self.model = get_model(Config.NUM_CLASSES)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(Config.DEVICE)
        self.model.eval()
        
        print(f"Model loaded successfully")
        print(f"Best validation IoU: {checkpoint.get('best_val_iou', 'N/A')}")
        print(f"Trained for {checkpoint.get('epoch', 'N/A')} epochs")
        print(f"Using TTA: {Config.USE_TTA}")
    
    @torch.no_grad()
    def predict_single(self, image):
        """
        Predict segmentation mask for a single image
        
        Args:
            image: (C, H, W) tensor
        
        Returns:
            predictions: (H, W) class indices
            probabilities: (C, H, W) class probabilities
        """
        # Add batch dimension
        image = image.unsqueeze(0).to(Config.DEVICE)
        
        # Forward pass
        outputs = self.model(image)
        
        # Upsample to input size if needed
        if outputs.shape[-2:] != image.shape[-2:]:
            outputs = F.interpolate(
                outputs,
                size=image.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Get probabilities and predictions
        probabilities = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        # Remove batch dimension
        predictions = predictions.squeeze(0)
        probabilities = probabilities.squeeze(0)
        
        return predictions, probabilities
    
    @torch.no_grad()
    def predict_with_tta(self, image):
        """
        Predict with Test-Time Augmentation
        Averages predictions from original and horizontally flipped image
        
        Args:
            image: (C, H, W) tensor
        
        Returns:
            predictions: (H, W) class indices
        """
        # Original prediction
        pred_original, prob_original = self.predict_single(image)
        
        if not Config.USE_TTA:
            return pred_original
        
        # Horizontal flip
        image_flipped = torch.flip(image, dims=[2])
        pred_flipped, prob_flipped = self.predict_single(image_flipped)
        
        # Flip back
        prob_flipped = torch.flip(prob_flipped, dims=[2])
        
        # Average probabilities
        prob_avg = (prob_original + prob_flipped) / 2.0
        
        # Get final predictions
        predictions = torch.argmax(prob_avg, dim=0)
        
        return predictions
    
    def predict_test_set(self, save_masks=True):
        """
        Predict on entire test set
        
        Args:
            save_masks: Whether to save predicted masks
        
        Returns:
            Dictionary of predictions
        """
        print("\n" + "=" * 60)
        print("Running inference on test set")
        print("=" * 60)
        
        # Create test loader
        test_loader = get_test_loader()
        
        predictions_dict = {}
        
        for image, img_name, original_size in tqdm(test_loader, desc="Predicting"):
            img_name = img_name[0]  # Remove batch dimension
            original_size = (original_size[0].item(), original_size[1].item())
            
            # Predict
            predictions = self.predict_with_tta(image.squeeze(0))
            
            # Move to CPU
            predictions = predictions.cpu().numpy()
            
            # Resize to original size if needed
            if predictions.shape != original_size:
                predictions_pil = Image.fromarray(predictions.astype(np.uint8))
                predictions_pil = predictions_pil.resize(
                    (original_size[1], original_size[0]),
                    Image.NEAREST
                )
                predictions = np.array(predictions_pil)
            
            predictions_dict[img_name] = predictions
            
            # Save mask
            if save_masks:
                self.save_prediction_mask(predictions, img_name)
        
        print(f"\nProcessed {len(predictions_dict)} test images")
        return predictions_dict
    
    def save_prediction_mask(self, mask, img_name):
        """
        Save predicted mask as PNG
        
        Args:
            mask: (H, W) class indices
            img_name: Image filename
        """
        # Create output directory
        os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)
        
        # Save as grayscale (class indices)
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(Config.PREDICTIONS_DIR, mask_name)
        
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img.save(mask_path)
    
    def save_colored_prediction(self, mask, img_name):
        """
        Save colored visualization of prediction
        
        Args:
            mask: (H, W) class indices
            img_name: Image filename
        """
        # Create RGB image
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_idx in range(Config.NUM_CLASSES):
            if class_idx < len(Config.CLASS_COLORS):
                colored_mask[mask == class_idx] = Config.CLASS_COLORS[class_idx]
        
        # Save
        colored_dir = os.path.join(Config.PREDICTIONS_DIR, 'colored')
        os.makedirs(colored_dir, exist_ok=True)
        
        colored_name = img_name.replace('.jpg', '_colored.png').replace('.jpeg', '_colored.png')
        colored_path = os.path.join(colored_dir, colored_name)
        
        colored_img = Image.fromarray(colored_mask)
        colored_img.save(colored_path)
    
    def evaluate_with_ground_truth(self, gt_mask_dir):
        """
        Evaluate predictions against ground truth (if available)
        
        Args:
            gt_mask_dir: Directory containing ground truth masks
        """
        print("\n" + "=" * 60)
        print("Evaluating predictions with ground truth")
        print("=" * 60)
        
        metrics = SegmentationMetrics(Config.NUM_CLASSES, Config.CLASS_NAMES)
        
        # Get all predictions
        pred_files = sorted([f for f in os.listdir(Config.PREDICTIONS_DIR) if f.endswith('.png')])
        
        for pred_file in tqdm(pred_files, desc="Evaluating"):
            # Load prediction
            pred_path = os.path.join(Config.PREDICTIONS_DIR, pred_file)
            pred_mask = np.array(Image.open(pred_path))
            
            # Load ground truth
            gt_path = os.path.join(gt_mask_dir, pred_file)
            if not os.path.exists(gt_path):
                continue
            
            gt_mask = np.array(Image.open(gt_path))
            
            # Ensure masks are single channel
            if len(pred_mask.shape) == 3:
                pred_mask = pred_mask[:, :, 0]
            if len(gt_mask.shape) == 3:
                gt_mask = gt_mask[:, :, 0]
            
            # Update metrics
            metrics.update(pred_mask, gt_mask)
        
        # Print results
        metrics.print_metrics()
        
        # Save confusion matrix
        cm_path = os.path.join(Config.RUNS_DIR, 'test_confusion_matrix.png')
        save_confusion_matrix(
            metrics.get_confusion_matrix(),
            Config.CLASS_NAMES,
            cm_path
        )
        
        return metrics.get_metrics_dict()


def visualize_predictions(image_path, mask_path, output_path):
    """
    Create side-by-side visualization of image and prediction
    
    Args:
        image_path: Path to input image
        mask_path: Path to predicted mask
        output_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Load image and mask
    image = Image.open(image_path).convert('RGB')
    mask = np.array(Image.open(mask_path))
    
    # Create colored mask
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx in range(Config.NUM_CLASSES):
        if class_idx < len(Config.CLASS_COLORS):
            colored_mask[mask == class_idx] = Config.CLASS_COLORS[class_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(colored_mask)
    axes[1].set_title('Predicted Segmentation')
    axes[1].axis('off')
    
    # Overlay
    overlay = np.array(image).astype(float) * 0.5 + colored_mask.astype(float) * 0.5
    overlay = overlay.astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function for testing"""
    Config.create_dirs()
    
    # Path to best model checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    # Create predictor
    predictor = Predictor(checkpoint_path)
    
    # Predict on test set
    predictions = predictor.predict_test_set(save_masks=True)
    
    print("\n" + "=" * 60)
    print("Predictions saved to:", Config.PREDICTIONS_DIR)
    print("=" * 60)
    
    # If ground truth is available, evaluate
    test_mask_dir = Config.TEST_IMG_DIR.replace('images', 'masks')
    if os.path.exists(test_mask_dir):
        print("\nGround truth found. Running evaluation...")
        metrics = predictor.evaluate_with_ground_truth(test_mask_dir)
        print(f"\nTest Set Mean IoU: {metrics['mean_iou']:.4f}")
    else:
        print("\nNo ground truth found. Skipping evaluation.")
    
    # Create some visualizations
    print("\nCreating visualizations...")
    test_images = sorted([f for f in os.listdir(Config.TEST_IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    vis_dir = os.path.join(Config.PREDICTIONS_DIR, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, img_file in enumerate(test_images[:5]):  # Visualize first 5 images
        img_path = os.path.join(Config.TEST_IMG_DIR, img_file)
        mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(Config.PREDICTIONS_DIR, mask_file)
        
        if os.path.exists(mask_path):
            vis_path = os.path.join(vis_dir, f'visualization_{i+1}.png')
            visualize_predictions(img_path, mask_path, vis_path)
    
    print(f"Visualizations saved to: {vis_dir}")


if __name__ == '__main__':
    main()
