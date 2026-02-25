# Complete Setup Guide

## Environment Setup

### Option 1: Conda (Recommended)

```bash
# Create environment
conda create -n offroad python=3.9
conda activate offroad

# Install PyTorch (CUDA 11.8 - adjust based on your GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install segmentation-models-pytorch albumentations opencv-python
pip install scikit-learn matplotlib seaborn tqdm

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Option 2: pip (Alternative)

```bash
# Create virtual environment
python -m venv offroad_env
source offroad_env/bin/activate  # Linux/Mac
# or
offroad_env\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt
```

## Data Preparation

### 1. Download Competition Data

Download the dataset from the competition platform and organize as follows:

```
data/
├── Train/
│   ├── images/       # Training images (.png or .jpg)
│   └── masks/        # Training masks (.png, same names as images)
├── Val/
│   ├── images/       # Validation images
│   └── masks/        # Validation masks
└── testImages/       # Test images (no masks)
```

### 2. Update Configuration

Edit `config.py`:

```python
# Update this path to point to your data directory
DATA_ROOT = Path("/path/to/your/data")
```

### 3. Verify Data Structure

```bash
python utils.py
```

This will:
- Check if all directories exist
- Count images and masks
- Analyze class distribution
- Visualize sample data
- Check image-mask correspondence

## Quick Start

### 1. Verify Everything Works

```bash
# Check data structure and visualize samples
python utils.py
```

### 2. Train the Model

```bash
python train.py
```

Training will:
- Automatically compute class weights
- Save best model to `runs/checkpoints/best_model.pth`
- Generate training curves in `runs/logs/`
- Create confusion matrix
- Save metrics history as JSON

**Expected training time:**
- RTX 3090: ~2-3 hours (150 epochs, batch_size=8)
- RTX 3080: ~3-4 hours
- RTX 3070: ~4-6 hours

### 3. Evaluate on Validation Set

```bash
python test.py --eval
```

### 4. Generate Test Predictions

```bash
python test.py
```

Predictions will be saved to `runs/predictions/`:
- `pred_*.png` - Segmentation masks (submit these)
- `vis_*.png` - Visualizations
- `color_legend.png` - Class color mapping

## Configuration Options

### Model Selection

**SegFormer (Recommended for Generalization):**
```python
MODEL_NAME = "segformer"
ENCODER_NAME = "mit_b3"  # b0 (fastest) to b5 (best quality)
```

**DeepLabV3+ (Strong CNN Baseline):**
```python
MODEL_NAME = "deeplabv3plus"
ENCODER_NAME = "resnet101"  # or resnet50
```

### Training Parameters

```python
# Batch size (reduce if out of memory)
BATCH_SIZE = 8  # Try 4, 2, or 1 if OOM

# Image size (reduce for faster training)
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# Epochs
NUM_EPOCHS = 150  # Can reduce to 50-100 for faster results

# Learning rate
LEARNING_RATE = 1e-4  # Increase to 2e-4 for faster convergence
```

### Augmentation Strength

```python
# For more aggressive augmentation (better generalization)
AUG_PROB = 0.9
ROTATION_LIMIT = 20
BRIGHTNESS_LIMIT = 0.3
CONTRAST_LIMIT = 0.3

# For less augmentation (if already generalizing well)
AUG_PROB = 0.5
ROTATION_LIMIT = 10
BRIGHTNESS_LIMIT = 0.1
CONTRAST_LIMIT = 0.1
```

## Advanced Usage

### Resume Training from Checkpoint

Modify `train.py`:

```python
# At the start of train() function, add:
checkpoint_path = Config.CHECKPOINT_DIR / 'checkpoint_epoch_50.pth'
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
```

### Enable TensorBoard Logging

```bash
# Install tensorboard
pip install tensorboard

# Add to train.py
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(Config.LOG_DIR / 'tensorboard')

# In training loop, add:
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('IoU/val', val_metrics['mean_iou'], epoch)

# View logs
tensorboard --logdir runs/logs/tensorboard
```

### Ensemble Multiple Models

Train multiple models with different seeds:

```python
# Train 3 models
for seed in [42, 123, 456]:
    Config.SEED = seed
    train()  # Will save to different checkpoints
```

Then create ensemble predictions in `test.py`.

## Troubleshooting

### Out of Memory Errors

```python
# In config.py, reduce:
BATCH_SIZE = 4  # or 2, or 1
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384

# Or use gradient accumulation (modify train.py):
accumulation_steps = 4
# Accumulate gradients over 4 batches to simulate larger batch size
```

### Model Not Converging

```python
# Try different learning rate
LEARNING_RATE = 5e-5  # Smaller
# or
LEARNING_RATE = 2e-4  # Larger

# Increase augmentation
AUG_PROB = 0.9

# Check class weights are being computed correctly
```

### Poor Generalization to Test Set

```python
# Strengthen augmentation
ROTATION_LIMIT = 20
BRIGHTNESS_LIMIT = 0.3
CONTRAST_LIMIT = 0.3

# Add more augmentations in dataset.py:
A.RandomGamma(gamma_limit=(70, 130), p=0.5),
A.CLAHE(p=0.3),
A.RandomFog(p=0.1),

# Use stronger early stopping
EARLY_STOPPING_PATIENCE = 15
```

### Class Imbalance Issues

The code automatically computes class weights. If some classes are still underperforming:

```python
# In loss.py, try Focal Loss instead:
from loss import FocalLoss
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
```

## Performance Optimization

### For Fastest Training

```python
MODEL_NAME = "segformer"
ENCODER_NAME = "mit_b0"  # Smallest model
BATCH_SIZE = 16  # Larger if GPU allows
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
NUM_EPOCHS = 50
USE_AMP = True  # Mixed precision
NUM_WORKERS = 8  # More data loading threads
```

### For Best Accuracy

```python
MODEL_NAME = "segformer"
ENCODER_NAME = "mit_b5"  # Largest model
BATCH_SIZE = 4  # Smaller due to model size
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
NUM_EPOCHS = 200
USE_TTA = True  # Test-time augmentation
# Train multiple models and ensemble
```

## Validation During Training

Monitor these metrics:
1. **Validation Mean IoU** - Primary metric (higher is better)
2. **Training Loss** - Should decrease steadily
3. **Validation Loss** - Should decrease without diverging from train loss
4. **Learning Rate** - Check it's decreasing appropriately

**Good signs:**
- Val IoU increasing over time
- Gap between train/val loss is small
- Per-class IoU is balanced

**Warning signs:**
- Val IoU plateaus early → Increase augmentation
- Large train/val gap → Model overfitting
- Some classes have very low IoU → Check class weights

## Submission Checklist

- [ ] Data structure verified (`python utils.py`)
- [ ] Model trained successfully (`python train.py`)
- [ ] Validation IoU > 0.65 (target varies by competition)
- [ ] Test predictions generated (`python test.py`)
- [ ] Predictions are in correct format (.png, uint8)
- [ ] All test images have predictions
- [ ] Best model checkpoint saved
- [ ] Training logs documented

## Common Issues

### "No module named 'segmentation_models_pytorch'"
```bash
pip install segmentation-models-pytorch
```

### "CUDA out of memory"
```python
# Reduce BATCH_SIZE in config.py
BATCH_SIZE = 2
```

### "No images found in directory"
```python
# Update DATA_ROOT in config.py
# Check image extensions (.png, .jpg, .jpeg)
```

### "Class weights contain NaN"
```python
# Some classes might be missing in training data
# Check class distribution with utils.py
```

## Getting Help

1. Run data verification: `python utils.py`
2. Check error messages in console
3. Verify GPU availability: `nvidia-smi`
4. Check CUDA version matches PyTorch
5. Review configuration in `config.py`

## Next Steps After Setup

1. Run `python utils.py` to verify everything
2. Start with small experiment: Set `NUM_EPOCHS = 10` in config
3. Check if training runs without errors
4. Inspect training curves in `runs/logs/training_history.png`
5. If successful, run full training with `NUM_EPOCHS = 150`
6. Generate test predictions
7. Submit to competition platform

Good luck! 🚀
