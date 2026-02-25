"""
Utility functions for data verification and analysis
"""

import cv2
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from config import Config


def verify_data_structure():
    """
    Verify that data directory structure is correct
    """
    print("\n" + "="*80)
    print("DATA STRUCTURE VERIFICATION")
    print("="*80)
    
    errors = []
    warnings = []
    
    # Check data root
    if not Config.DATA_ROOT.exists():
        errors.append(f"Data root not found: {Config.DATA_ROOT}")
        print(f"❌ Data root not found: {Config.DATA_ROOT}")
        print("\nPlease update DATA_ROOT in config.py to point to your data directory")
        return False
    
    print(f"✓ Data root found: {Config.DATA_ROOT}")
    
    # Check train directories
    if not Config.TRAIN_IMAGE_DIR.exists():
        errors.append(f"Train images not found: {Config.TRAIN_IMAGE_DIR}")
    else:
        train_images = list(Config.TRAIN_IMAGE_DIR.glob("*.png")) + \
                      list(Config.TRAIN_IMAGE_DIR.glob("*.jpg"))
        print(f"✓ Train images found: {len(train_images)} images")
        
        if len(train_images) == 0:
            warnings.append("No training images found")
    
    if not Config.TRAIN_MASK_DIR.exists():
        errors.append(f"Train masks not found: {Config.TRAIN_MASK_DIR}")
    else:
        train_masks = list(Config.TRAIN_MASK_DIR.glob("*.png"))
        print(f"✓ Train masks found: {len(train_masks)} masks")
        
        if len(train_masks) == 0:
            warnings.append("No training masks found")
    
    # Check validation directories
    if not Config.VAL_IMAGE_DIR.exists():
        warnings.append(f"Val images not found: {Config.VAL_IMAGE_DIR}")
    else:
        val_images = list(Config.VAL_IMAGE_DIR.glob("*.png")) + \
                    list(Config.VAL_IMAGE_DIR.glob("*.jpg"))
        print(f"✓ Val images found: {len(val_images)} images")
    
    if not Config.VAL_MASK_DIR.exists():
        warnings.append(f"Val masks not found: {Config.VAL_MASK_DIR}")
    else:
        val_masks = list(Config.VAL_MASK_DIR.glob("*.png"))
        print(f"✓ Val masks found: {len(val_masks)} masks")
    
    # Check test directory
    if not Config.TEST_IMAGE_DIR.exists():
        warnings.append(f"Test images not found: {Config.TEST_IMAGE_DIR}")
    else:
        test_images = list(Config.TEST_IMAGE_DIR.glob("*.png")) + \
                     list(Config.TEST_IMAGE_DIR.glob("*.jpg"))
        print(f"✓ Test images found: {len(test_images)} images")
    
    # Print warnings
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Print errors
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print("\n" + "="*80)
        return False
    
    print("\n" + "="*80)
    print("✓ Data structure verification passed!")
    print("="*80 + "\n")
    return True


def analyze_class_distribution():
    """
    Analyze class distribution in training data
    """
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    if not Config.TRAIN_MASK_DIR.exists():
        print("❌ Training masks not found")
        return
    
    mask_files = list(Config.TRAIN_MASK_DIR.glob("*.png"))
    
    if len(mask_files) == 0:
        print("❌ No mask files found")
        return
    
    print(f"Analyzing {len(mask_files)} mask files...")
    
    # Count pixels per class
    class_counts = Counter()
    
    for mask_path in mask_files[:100]:  # Sample first 100 to avoid long processing
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                class_counts[cls] += count
    
    # Print distribution
    total_pixels = sum(class_counts.values())
    print(f"\nTotal pixels analyzed: {total_pixels:,}")
    print("\nClass distribution:")
    print(f"{'Class ID':<10} {'Class Name':<20} {'Pixels':<15} {'Percentage':<10}")
    print("-" * 80)
    
    for cls_id in sorted(class_counts.keys()):
        if cls_id < len(Config.CLASS_NAMES):
            cls_name = Config.CLASS_NAMES[cls_id]
        else:
            cls_name = f"Unknown_{cls_id}"
        
        count = class_counts[cls_id]
        percentage = (count / total_pixels) * 100
        print(f"{cls_id:<10} {cls_name:<20} {count:<15,} {percentage:<10.2f}%")
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    labels = [Config.CLASS_NAMES[c] if c < len(Config.CLASS_NAMES) else f"Class_{c}" 
              for c in classes]
    
    plt.bar(range(len(classes)), counts)
    plt.xticks(range(len(classes)), labels, rotation=45, ha='right')
    plt.xlabel('Class')
    plt.ylabel('Pixel Count')
    plt.title('Class Distribution in Training Data (Sample)')
    plt.tight_layout()
    
    # Save plot
    output_path = Config.OUTPUT_DIR / 'class_distribution.png'
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nClass distribution plot saved to: {output_path}")
    print("="*80 + "\n")


def visualize_sample_data(num_samples: int = 5):
    """
    Visualize sample images and masks from training data
    
    Args:
        num_samples: Number of samples to visualize
    """
    print("\n" + "="*80)
    print("SAMPLE DATA VISUALIZATION")
    print("="*80)
    
    if not Config.TRAIN_IMAGE_DIR.exists() or not Config.TRAIN_MASK_DIR.exists():
        print("❌ Training data not found")
        return
    
    image_files = sorted(list(Config.TRAIN_IMAGE_DIR.glob("*.png")) + 
                        list(Config.TRAIN_IMAGE_DIR.glob("*.jpg")))
    
    if len(image_files) == 0:
        print("❌ No images found")
        return
    
    # Sample random images
    np.random.seed(42)
    sample_indices = np.random.choice(len(image_files), 
                                     min(num_samples, len(image_files)), 
                                     replace=False)
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Generate class colors
    np.random.seed(42)
    class_colors = np.random.randint(0, 255, size=(Config.NUM_CLASSES, 3))
    
    for idx, img_idx in enumerate(sample_indices):
        img_path = image_files[img_idx]
        mask_path = Config.TRAIN_MASK_DIR / img_path.name
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Create colored mask
            colored_mask = class_colors[mask]
            
            # Display
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title(f'Image: {img_path.name}')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(colored_mask)
            axes[idx, 1].set_title(f'Mask (Colored)')
            axes[idx, 1].axis('off')
        else:
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title(f'Image: {img_path.name}')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].text(0.5, 0.5, 'Mask not found', 
                            ha='center', va='center', fontsize=16)
            axes[idx, 1].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = Config.OUTPUT_DIR / 'sample_data.png'
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sample visualization saved to: {output_path}")
    print("="*80 + "\n")


def check_image_mask_correspondence():
    """
    Check if all images have corresponding masks
    """
    print("\n" + "="*80)
    print("IMAGE-MASK CORRESPONDENCE CHECK")
    print("="*80)
    
    if not Config.TRAIN_IMAGE_DIR.exists() or not Config.TRAIN_MASK_DIR.exists():
        print("❌ Training directories not found")
        return
    
    image_files = set([p.stem for p in Config.TRAIN_IMAGE_DIR.glob("*.png")] + 
                     [p.stem for p in Config.TRAIN_IMAGE_DIR.glob("*.jpg")])
    mask_files = set([p.stem for p in Config.TRAIN_MASK_DIR.glob("*.png")])
    
    missing_masks = image_files - mask_files
    extra_masks = mask_files - image_files
    
    print(f"Images: {len(image_files)}")
    print(f"Masks: {len(mask_files)}")
    
    if missing_masks:
        print(f"\n⚠️  {len(missing_masks)} images without masks:")
        for name in list(missing_masks)[:10]:
            print(f"  - {name}")
        if len(missing_masks) > 10:
            print(f"  ... and {len(missing_masks) - 10} more")
    
    if extra_masks:
        print(f"\n⚠️  {len(extra_masks)} masks without images:")
        for name in list(extra_masks)[:10]:
            print(f"  - {name}")
        if len(extra_masks) > 10:
            print(f"  ... and {len(extra_masks) - 10} more")
    
    if not missing_masks and not extra_masks:
        print("✓ All images have corresponding masks")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    """Run all verification and analysis"""
    
    # Verify data structure
    if verify_data_structure():
        # Analyze class distribution
        analyze_class_distribution()
        
        # Visualize samples
        visualize_sample_data(num_samples=5)
        
        # Check correspondence
        check_image_mask_correspondence()
    else:
        print("\n⚠️  Please fix data structure issues before proceeding")
        print("Update paths in config.py to match your data location")
