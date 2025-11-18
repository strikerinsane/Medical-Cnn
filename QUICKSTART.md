# Quick Start Guide

## Step 1: Setup Environment

```bash
# Install required packages
pip install -r requirements.txt

# Create directory structure
python setup.py
```

## Step 2: Prepare Dataset

Place your medical images in the following folders:

```
./training/normal/      # Normal training images
./training/bleeding/    # Bleeding training images
./validation/normal/    # Normal validation images
./validation/bleeding/  # Bleeding validation images
```

**Recommended**: At least 100 images per class for good results.

## Step 3: Train the Model

```bash
# Basic training (uses defaults)
python train.py

# Advanced training with custom parameters
python train.py --epochs 100 --batch_size 16 --learning_rate 0.0005
```

Training will generate:
- `medical_cnn_model.h5` - Trained model
- `training_history.png` - Performance plots
- `confusion_matrix.png` - Confusion matrix

## Step 4: Make Predictions

### Option 1: GUI (Easiest)
```bash
python predict.py --gui
```
A file picker will open. Select your image and get instant results.

### Option 2: Command Line
```bash
python predict.py --image_path path/to/image.jpg
```

### Option 3: Batch Processing
```bash
python predict.py --image_dir path/to/images/folder
```

## Expected Output

After prediction, you'll see:

```
================================================================================
PREDICTION RESULT
================================================================================
Image: test_image.jpg
Predicted Class: Bleeding
Confidence: 94.23%
Raw Probability (Bleeding): 0.9423
================================================================================
```

## Tips for Best Results

1. **Image Quality**: Use high-quality medical images
2. **Balanced Dataset**: Try to have similar numbers of normal and bleeding images
3. **Training Duration**: Train for at least 30-50 epochs
4. **Data Augmentation**: Already enabled by default during training
5. **Monitor Training**: Watch the training plots to ensure convergence

## Troubleshooting

- **Low accuracy?** Need more training data or more epochs
- **Model not found?** Run `train.py` first
- **Out of memory?** Reduce batch size: `--batch_size 8`
- **Import errors?** Run: `pip install -r requirements.txt`

## Need Help?

Check the full README.md for detailed documentation.
