# Quick Start Guide - Google Colab

## 🚀 Fast Setup (5 minutes)

### Step 1: Open Google Colab
1. Go to https://colab.research.google.com/
2. Click: **File > Upload notebook**
3. Upload `Offroad_Segmentation.ipynb`

### Step 2: Enable GPU
1. Click: **Runtime > Change runtime type**
2. Select: **GPU** (T4 or better)
3. Click: **Save**

### Step 3: Run Setup Cell
```python
!pip install -q segmentation-models-pytorch albumentations scikit-learn seaborn
```

### Step 4: Upload Dataset
Upload your dataset.zip with this structure:
```
dataset.zip
├── Train/
│   ├── images/
│   └── masks/
├── Val/
│   ├── images/
│   └── masks/
└── testImages/
```

### Step 5: Upload Code
Upload all 7 Python files:
- config.py
- dataset.py
- model.py
- loss.py
- metrics.py
- train.py
- test.py

### Step 6: Train
```python
!python train.py
```

### Step 7: Test
```python
!python test.py
```

### Step 8: Download Results
```python
from google.colab import files
files.download('/content/predictions.zip')
```

---

## 📊 Expected Output

### Training (2-4 hours on T4)
```
Epoch 1/100 [Train]: 100%|██| loss: 0.823 | IoU: 0.452
Epoch 1/100 [Val]:   100%|██| loss: 0.789 | IoU: 0.486
✓ Saved best model with IoU: 0.486

...

Best Validation IoU: 0.7234
```

### Testing
```
Running inference on test set
Predicting: 100%|████████████| 150/150
✓ Predictions saved to: /content/predictions/

Test Set Mean IoU: 0.6845
```

---

## ⚙️ Key Configuration

All settings in `config.py`:

```python
# Model
MODEL_NAME = 'segformer'  # Best for generalization
LEARNING_RATE = 0.1       # As required

# Training
BATCH_SIZE = 8
NUM_EPOCHS = 100
USE_AMP = True           # Mixed precision
USE_TTA = True           # Test-time augmentation

# Loss
CE_WEIGHT = 0.5          # CrossEntropy weight
DICE_WEIGHT = 0.5        # Dice Loss weight
```

---

## 🔧 Troubleshooting

### Out of Memory?
```python
BATCH_SIZE = 4
IMAGE_SIZE = (384, 384)
```

### Training too slow?
- Verify GPU enabled
- Reduce NUM_EPOCHS to 50
- Use smaller model: `SEGFORMER_ENCODER = 'mit_b2'`

### Poor results?
- Train longer
- Check class distribution (printed during training)
- Increase augmentation strength

---

## 📈 Competition Tips

1. **Domain Shift**: Model trains on one desert, tests on another
   - Strong augmentation is critical
   - Don't overtrain (watch early stopping)
   
2. **Class Imbalance**: Some classes rare (Flowers, Logs)
   - Class weights computed automatically
   - Check confusion matrix for weak classes

3. **TTA**: Always enable for +1-2% IoU boost

4. **Ensemble**: Train 3-5 models and average predictions

---

## 📁 Output Files

After training and testing:

```
/content/
├── checkpoints/
│   ├── best.pth          # Best model (highest val IoU)
│   └── latest.pth        # Latest checkpoint
├── runs/
│   ├── training_history.json
│   └── confusion_matrix.png
└── predictions/
    ├── *.png            # Predicted masks (grayscale)
    ├── colored/         # RGB visualizations
    └── visualizations/  # Side-by-side comparisons
```

---

## 🎯 Submission Checklist

- [ ] Training complete (best IoU > 0.65)
- [ ] Test predictions generated
- [ ] Predictions in correct format (PNG, class indices)
- [ ] Filenames match test images
- [ ] Downloaded predictions.zip
- [ ] Verified sample predictions visually

---

## 💡 Advanced Usage

### Change Model
```python
# In config.py
MODEL_NAME = 'deeplabv3plus'
DEEPLABV3_BACKBONE = 'resnet101'
```

### Adjust Learning Rate
```python
# For stability
LEARNING_RATE = 0.01  # Lower
OPTIMIZER = 'adam'    # Instead of adamw

# For faster convergence
LEARNING_RATE = 0.1   # Higher (current)
OPTIMIZER = 'adamw'
```

### Custom Augmentation
```python
# In dataset.py get_train_transform()
ROTATION_LIMIT = 30        # More rotation
BRIGHTNESS_LIMIT = 0.3     # More brightness
GAUSSIAN_NOISE_VAR = (20, 100)  # More noise
```

---

## 📞 Support

If stuck:
1. Check console output for errors
2. Verify dataset structure
3. Confirm GPU enabled
4. Review config.py settings

**Good luck with the competition! 🏆**
