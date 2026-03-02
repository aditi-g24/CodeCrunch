# Offroad Semantic Segmentation - Competition Solution

High-performance semantic segmentation solution for Duality AI's Offroad Scene Segmentation Hackathon. Optimized for maximum IoU and strong generalization to unseen desert environments.

## 🎯 Features

### Architecture
- **SegFormer** (default): Superior global context modeling for domain shift robustness
- **DeepLabV3+**: Alternative with ResNet50/101 backbone and ASPP module
- Pretrained ImageNet weights for transfer learning

### Loss Function
- **Hybrid Loss**: `0.5 × CrossEntropy + 0.5 × Dice Loss`
- Automatic class weight computation from training data
- Direct IoU optimization via Dice Loss

### Training Optimizations
- **Mixed Precision (FP16)**: Faster training with reduced memory
- **Strong Augmentation Pipeline**:
  - Random horizontal flip
  - Random rotation (±15°)
  - Color jitter (brightness, contrast, saturation, hue)
  - Random scaling
  - Gaussian noise
- **Learning Rate Scheduling**: CosineAnnealing or ReduceLROnPlateau
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best model based on validation IoU

### Evaluation Metrics
- Per-class IoU
- Mean IoU
- Precision and Recall per class
- Confusion Matrix
- Pixel Accuracy

### Test-Time Augmentation (TTA)
- Horizontal flip averaging for improved predictions

## 📁 Project Structure

```
offroad_segmentation/
├── config.py          # Configuration and hyperparameters
├── dataset.py         # Data loading and augmentation
├── model.py          # Model architectures
├── loss.py           # Hybrid loss implementation
├── metrics.py        # Evaluation metrics
├── train.py          # Training script
├── test.py           # Inference script
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## 🚀 Quick Start for Google Colab

### 1. Setup

```python
# Clone or upload the code to Colab
!git clone <your-repo-url>
# OR upload files directly

# Install dependencies
!pip install segmentation-models-pytorch albumentations --break-system-packages
```

### 2. Upload Dataset

Upload your dataset to Colab with this structure:
```
/content/
├── Train/
│   ├── images/
│   └── masks/
├── Val/
│   ├── images/
│   └── masks/
└── testImages/
```

### 3. Configure Paths

The config.py is already set up for Colab paths:
```python
TRAIN_IMG_DIR = '/content/Train/images'
TRAIN_MASK_DIR = '/content/Train/masks'
VAL_IMG_DIR = '/content/Val/images'
VAL_MASK_DIR = '/content/Val/masks'
TEST_IMG_DIR = '/content/testImages'
```

### 4. Train

```python
!python train.py
```

Training will:
- Automatically compute class weights
- Save best model to `/content/checkpoints/best.pth`
- Log training history to `/content/runs/`
- Display progress bars with loss and IoU

### 5. Test/Inference

```python
!python test.py
```

This will:
- Load the best model
- Run predictions with TTA on test images
- Save masks to `/content/predictions/`
- Create visualizations
- Compute metrics if ground truth available

## ⚙️ Configuration

Edit `config.py` to customize:

### Model Selection
```python
MODEL_NAME = 'segformer'  # or 'deeplabv3plus'
SEGFORMER_ENCODER = 'mit_b3'  # mit_b0 to mit_b5
DEEPLABV3_BACKBONE = 'resnet101'  # resnet50 or resnet101
```

### Hyperparameters
```python
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 0.1  # As required
IMAGE_SIZE = (512, 512)
USE_AMP = True  # Mixed precision
USE_TTA = True  # Test-time augmentation
```

### Augmentation Strength
```python
ROTATION_LIMIT = 15
BRIGHTNESS_LIMIT = 0.2
CONTRAST_LIMIT = 0.2
SCALE_LIMIT = 0.15
```

### Loss Weights
```python
CE_WEIGHT = 0.5
DICE_WEIGHT = 0.5
USE_CLASS_WEIGHTS = True
```

## 📊 Understanding the Output

### Training Output
```
Epoch 1/100 [Train]: 100%|██████| loss: 0.8234 | ce: 0.4567 | dice: 0.3667
Epoch 1/100 [Val]:   100%|██████|
Train Loss: 0.8234 | Train IoU: 0.4523
Val Loss:   0.7891 | Val IoU:   0.4856
✓ Saved best model with IoU: 0.4856
```

### Checkpoints
- `best.pth`: Model with highest validation IoU
- `latest.pth`: Most recent model (for resuming training)

### Predictions
- `/content/predictions/`: Grayscale masks (class indices)
- `/content/predictions/colored/`: RGB visualization
- `/content/predictions/visualizations/`: Side-by-side comparisons

### Metrics Report
```
==============================================================================
SEGMENTATION METRICS
==============================================================================

Overall Metrics:
  Mean IoU:         0.7234
  Pixel Accuracy:   0.8567

Per-Class Metrics:
--------------------------------------------------------------------------------
Class                      IoU   Precision     Recall         F1
--------------------------------------------------------------------------------
Trees                   0.7845      0.8234     0.7456     0.7823
Lush_Bushes             0.6923      0.7345     0.6512     0.6901
...
```

## 🔧 Advanced Usage

### Resume Training
```python
# Modify train.py to load checkpoint:
checkpoint = torch.load('checkpoints/latest.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Custom Dataset Structure
If your dataset has different structure, modify in `config.py`:
```python
TRAIN_IMG_DIR = '/path/to/train/images'
TRAIN_MASK_DIR = '/path/to/train/masks'
```

### Experiment with Different Models
```python
# Try SegFormer variants
SEGFORMER_ENCODER = 'mit_b5'  # Larger model

# Or switch to DeepLabV3+
MODEL_NAME = 'deeplabv3plus'
DEEPLABV3_BACKBONE = 'resnet101'
```

### Adjust for Memory Constraints
```python
BATCH_SIZE = 4  # Reduce if OOM
IMAGE_SIZE = (384, 384)  # Smaller images
USE_AMP = True  # Keep enabled for memory savings
```

## 🎓 Domain Shift Strategy

This solution is optimized for domain shift (train on one desert, test on another):

1. **Strong Augmentation**: Color jitter and noise help model learn invariant features
2. **SegFormer Architecture**: Global attention mechanisms generalize better than pure CNNs
3. **Dice Loss**: Directly optimizes for IoU metric
4. **Class Weighting**: Handles imbalanced classes in desert scenes
5. **Early Stopping**: Prevents overfitting to source domain
6. **TTA**: Averages predictions for robustness

## 📈 Expected Performance

With proper training (50-100 epochs):
- **Validation IoU**: 0.65 - 0.75
- **Test IoU**: 0.60 - 0.70 (domain shift)

Performance varies based on:
- Domain similarity between train/test
- Class balance in dataset
- Model capacity (mit_b3 vs mit_b5)
- Training duration

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
BATCH_SIZE = 4
IMAGE_SIZE = (384, 384)
```

### segmentation_models_pytorch not found
```bash
!pip install segmentation-models-pytorch --break-system-packages
```

### Poor generalization
- Increase augmentation strength
- Train longer but watch for early stopping
- Try mit_b5 encoder for more capacity
- Reduce learning rate to 0.01 or 0.001

### Class imbalance
- Ensure `USE_CLASS_WEIGHTS = True`
- Check class distribution printed during training
- Consider Focal Loss (implemented in loss.py)

## 📝 Citation

If you use this code, please cite:
```
@misc{offroad_segmentation_2024,
  title={Competition-Grade Semantic Segmentation for Offroad Scenes},
  author={Your Name},
  year={2024}
}
```

## 📄 License

MIT License - feel free to use for competitions and research.

## 🤝 Contributing

Suggestions and improvements welcome! Key areas:
- Additional augmentation strategies
- New model architectures
- Multi-scale inference
- Ensemble methods
