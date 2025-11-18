"""
Prediction Module
Handles prediction on new images with CLI and GUI support.
"""

import os
import argparse
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow import keras

from enhance import preprocess_image, normalize_image
from model import load_saved_model

class MedicalImagePredictor:
    """
    Predictor for medical image classification.
    """

    def __init__(self, model_path='medical_cnn_model.h5', img_size=(128, 128)):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model
            img_size: Target image size
        """
        self.img_size = img_size
        self.model = load_saved_model(model_path)
        self.class_names = ['Normal', 'Bleeding']

    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction.

        Args:
            image_path: Path to image file
        Returns:
            Preprocessed image array
        """
        # Read image
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Resize
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)

        # Apply enhancement (no augmentation for prediction)
        image = preprocess_image(image, 
                                apply_augmentation=False,
                                apply_enhancement=True)

        # Normalize
        image = normalize_image(image)

        return image

    def predict(self, image_path, verbose=True):
        """
        Predict class for given image.

        Args:
            image_path: Path to image file
            verbose: Whether to print results
        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Preprocess image
        image = self.preprocess_image(image_path)

        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)

        # Predict
        prediction_proba = self.model.predict(image_batch, verbose=0)[0][0]

        # Get class and confidence
        if prediction_proba > 0.5:
            predicted_class = 'Bleeding'
            confidence = prediction_proba
        else:
            predicted_class = 'Normal'
            confidence = 1 - prediction_proba

        if verbose:
            print("\n" + "="*80)
            print("PREDICTION RESULT")
            print("="*80)
            print(f"Image: {image_path}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence*100:.2f}%")
            print(f"Raw Probability (Bleeding): {prediction_proba:.4f}")
            print("="*80 + "\n")

        return predicted_class, confidence

    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images.

        Args:
            image_paths: List of image paths
        Returns:
            List of tuples (predicted_class, confidence)
        """
        results = []

        print(f"\nPredicting {len(image_paths)} images...\n")

        for i, image_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] Processing: {image_path}")
            try:
                predicted_class, confidence = self.predict(image_path, verbose=False)
                results.append((image_path, predicted_class, confidence))
                print(f"  -> {predicted_class} ({confidence*100:.2f}%)")
            except Exception as e:
                print(f"  -> Error: {e}")
                results.append((image_path, "Error", 0.0))

        return results

def select_image_gui():
    """
    Open GUI file selector to choose image.

    Returns:
        Selected image path
    """
    root = tk.Tk()
    root.withdraw()  # Hide main window

    file_path = filedialog.askopenfilename(
        title="Select Medical Image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    return file_path

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Predict internal bleeding from medical images'
    )

    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to image file for prediction')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Directory containing images for batch prediction')
    parser.add_argument('--model_path', type=str, default='medical_cnn_model.h5',
                       help='Path to trained model')
    parser.add_argument('--img_size', type=int, default=128,
                       help='Image size (height and width)')
    parser.add_argument('--gui', action='store_true',
                       help='Use GUI to select image')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        print("Please train the model first using train.py")
        return

    # Create predictor
    print("\nLoading model...")
    predictor = MedicalImagePredictor(
        model_path=args.model_path,
        img_size=(args.img_size, args.img_size)
    )

    # Determine prediction mode
    if args.gui:
        # GUI mode
        print("Opening file selector...")
        image_path = select_image_gui()

        if not image_path:
            print("No image selected. Exiting.")
            return

        predictor.predict(image_path)

    elif args.image_path:
        # Single image prediction
        if not os.path.exists(args.image_path):
            print(f"Error: Image file not found: {args.image_path}")
            return

        predictor.predict(args.image_path)

    elif args.image_dir:
        # Batch prediction
        if not os.path.exists(args.image_dir):
            print(f"Error: Directory not found: {args.image_dir}")
            return

        # Get all image files
        image_files = [f for f in os.listdir(args.image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

        if len(image_files) == 0:
            print(f"No image files found in {args.image_dir}")
            return

        image_paths = [os.path.join(args.image_dir, f) for f in image_files]
        results = predictor.predict_batch(image_paths)

        # Print summary
        print("\n" + "="*80)
        print("BATCH PREDICTION SUMMARY")
        print("="*80)
        for img_path, pred_class, conf in results:
            if pred_class != "Error":
                print(f"{os.path.basename(img_path):40s} -> {pred_class:10s} ({conf*100:5.2f}%)")
            else:
                print(f"{os.path.basename(img_path):40s} -> Error")
        print("="*80 + "\n")

    else:
        print("Error: Please specify either --image_path, --image_dir, or --gui")
        print("Use --help for more information")

if __name__ == '__main__':
    main()
