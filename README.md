# Medical Image Classification - Internal Bleeding Detection

A deep learning project for detecting internal bleeding from medical images using a custom CNN built from scratch.

## Project Overview

This project implements a Convolutional Neural Network (CNN) built entirely from scratch (no pre-trained weights or transfer learning) to classify medical images as either "Normal" or "Bleeding". The system includes comprehensive image preprocessing, enhancement techniques, and a user-friendly prediction interface.

## Features

- **Custom CNN Architecture**: Built from scratch using TensorFlow/Keras
- **Image Enhancement**: Histogram equalization, CLAHE, edge detection, noise reduction
- **Data Augmentation**: Random rotations, flips, and scaling
- **Modular Design**: Separate modules for data loading, model building, training, and prediction
- **Multiple Prediction Modes**: CLI with arguments, GUI file selector, and batch processing
- **Comprehensive Metrics**: Accuracy, precision, recall, AUC, confusion matrix, and classification report

## Directory Structure

```
medical_image_classification/
│
├── enhance.py              # Image enhancement and preprocessing functions
├── data_loader.py          # Data loading and preprocessing module
├── model.py                # CNN model architecture
├── train.py                # Training script
├── predict.py              # Prediction script with CLI and GUI support
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── training/               # Training data (create this folder)
│   ├── normal/            # Normal images
│   └── bleeding/          # Bleeding images
│
└── validation/             # Validation data (create this folder)
    ├── normal/            # Normal images
    └── bleeding/          # Bleeding images
```

## Installation

1. **Clone or extract the project**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare your dataset**:
   - Create the following directory structure:
     ```
     ./training/normal/       # Place normal training images here
     ./training/bleeding/     # Place bleeding training images here
     ./validation/normal/     # Place normal validation images here
     ./validation/bleeding/   # Place bleeding validation images here
     ```
   - Supported formats: PNG, JPG, JPEG, BMP, TIF, TIFF

## Usage

### 1. Training the Model

Train the CNN model with default parameters:

```bash
python train.py
```

With custom parameters:

```bash
python train.py --epochs 50 --batch_size 32 --learning_rate 0.001 --img_size 128
```

**Training Arguments**:
- `--training_dir`: Path to training data directory (default: './training')
- `--validation_dir`: Path to validation data directory (default: './validation')
- `--model_path`: Path to save trained model (default: 'medical_cnn_model.h5')
- `--img_size`: Image size in pixels (default: 128)
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.001)
- `--dropout_rate`: Dropout rate (default: 0.5)
- `--early_stopping_patience`: Early stopping patience (default: 10)

**Training Outputs**:
- `medical_cnn_model.h5`: Trained model file
- `training_history.png`: Training/validation metrics plots
- `confusion_matrix.png`: Confusion matrix visualization

### 2. Making Predictions

#### Option A: Predict Single Image (CLI)

```bash
python predict.py --image_path path/to/your/image.jpg
```

#### Option B: Predict Using GUI File Selector

```bash
python predict.py --gui
```

#### Option C: Batch Prediction

```bash
python predict.py --image_dir path/to/image/directory
```

**Prediction Arguments**:
- `--image_path`: Path to single image for prediction
- `--image_dir`: Directory containing images for batch prediction
- `--model_path`: Path to trained model (default: 'medical_cnn_model.h5')
- `--img_size`: Image size (default: 128)
- `--gui`: Use GUI file selector

## CNN Model Architecture

The model consists of:

1. **Input Layer**: 128×128×3 RGB images
2. **4 Convolutional Blocks**: Each with:
   - 2 Conv2D layers with ReLU activation
   - MaxPooling2D (2×2)
   - Batch Normalization
   - Dropout
   - Progressive filters: 32 → 64 → 128 → 256
3. **Dense Layers**:
   - Flatten layer
   - 3 Dense layers (512 → 256 → 128 neurons)
   - Batch Normalization and Dropout
4. **Output Layer**: Single neuron with sigmoid activation (binary classification)

**Total Parameters**: ~10M trainable parameters

## Image Enhancement Pipeline

The following preprocessing techniques are applied:

1. **Noise Reduction**: Gaussian blur
2. **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. **Data Augmentation** (training only):
   - Random rotations (±15°)
   - Random horizontal/vertical flips
   - Random scaling (0.95-1.05×)
4. **Normalization**: Scale to [0, 1]

## Performance Metrics

The system evaluates model performance using:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **AUC**: Area Under the ROC Curve
- **Confusion Matrix**: Visual representation of predictions
- **Classification Report**: Detailed per-class metrics

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your dataset in the required structure
# (Place images in ./training and ./validation folders)

# 3. Train the model
python train.py --epochs 50 --batch_size 32

# 4. Make predictions
python predict.py --gui  # Use GUI to select image
# OR
python predict.py --image_path test_image.jpg  # Direct path
# OR
python predict.py --image_dir ./test_images  # Batch prediction
```

## Troubleshooting

### Common Issues

1. **"No module named 'tensorflow'"**
   - Solution: Install TensorFlow: `pip install tensorflow`

2. **"Model file not found"**
   - Solution: Train the model first using `train.py`

3. **"No images loaded from directory"**
   - Solution: Check that images are in correct folders and have valid extensions

4. **GPU/Memory Issues**
   - Solution: Reduce batch size: `python train.py --batch_size 16`

5. **Low accuracy**
   - Ensure sufficient training data (recommended: 100+ images per class)
   - Increase training epochs
   - Check data quality and labeling

## Technical Details

- **Framework**: TensorFlow 2.x / Keras
- **Image Processing**: OpenCV
- **Data Science**: NumPy, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **GUI**: Tkinter (built-in)

## Notes

- The model is built entirely from scratch with no pre-trained weights
- All images are automatically resized to 128×128 pixels
- Class imbalance is handled through automatic class weight calculation
- The model uses early stopping to prevent overfitting
- Learning rate reduction on plateau is implemented

## License

This project is provided as-is for educational and research purposes.

## Author

Medical Image Classification System
Built with TensorFlow and Keras
