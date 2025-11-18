"""
Data Loader Module
Handles loading and preprocessing images from specified folder structure.
"""

import os
import cv2
import numpy as np
from enhance import preprocess_image, normalize_image

class MedicalImageDataLoader:
    """
    Data loader for medical image classification.
    Loads images from ./training and ./validation directories.
    """

    def __init__(self, img_size=(128, 128)):
        """
        Initialize data loader.

        Args:
            img_size: Target size for all images (height, width)
        """
        self.img_size = img_size
        self.class_names = ['normal', 'bleeding']
        self.class_to_label = {'normal': 0, 'bleeding': 1}

    def load_image(self, image_path, apply_augmentation=False, apply_enhancement=True):
        """
        Load and preprocess a single image.

        Args:
            image_path: Path to image file
            apply_augmentation: Whether to apply data augmentation
            apply_enhancement: Whether to apply enhancement techniques
        Returns:
            Preprocessed image array
        """
        # Read image
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Resize to target size
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)

        # Apply preprocessing
        image = preprocess_image(image, 
                                apply_augmentation=apply_augmentation,
                                apply_enhancement=apply_enhancement)

        # Normalize
        image = normalize_image(image)

        return image

    def load_dataset(self, data_dir, apply_augmentation=False):
        """
        Load entire dataset from directory.

        Args:
            data_dir: Root directory (e.g., './training' or './validation')
            apply_augmentation: Whether to apply data augmentation
        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []

        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue

            label = self.class_to_label[class_name]

            # Get all image files
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

            print(f"Loading {len(image_files)} images from {class_dir}...")

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)

                try:
                    image = self.load_image(img_path, 
                                          apply_augmentation=apply_augmentation,
                                          apply_enhancement=True)
                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue

        if len(images) == 0:
            raise ValueError(f"No images loaded from {data_dir}")

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        print(f"Loaded {len(images)} images from {data_dir}")
        print(f"Image shape: {images.shape}")
        print(f"Labels distribution: Normal={np.sum(labels==0)}, Bleeding={np.sum(labels==1)}")

        return images, labels

    def load_training_data(self, training_dir='./training', apply_augmentation=True):
        """
        Load training data.

        Args:
            training_dir: Path to training directory
            apply_augmentation: Whether to apply data augmentation
        Returns:
            Tuple of (images, labels)
        """
        return self.load_dataset(training_dir, apply_augmentation=apply_augmentation)

    def load_validation_data(self, validation_dir='./validation'):
        """
        Load validation data.

        Args:
            validation_dir: Path to validation directory
        Returns:
            Tuple of (images, labels)
        """
        return self.load_dataset(validation_dir, apply_augmentation=False)

    def get_class_weights(self, labels):
        """
        Calculate class weights for imbalanced datasets.

        Args:
            labels: Array of labels
        Returns:
            Dictionary of class weights
        """
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        weights = {int(cls): total / (len(unique) * count) 
                  for cls, count in zip(unique, counts)}
        print(f"Class weights: {weights}")
        return weights

def create_data_generators(img_size=(128, 128)):
    """
    Create data loader instance.

    Args:
        img_size: Target image size
    Returns:
        MedicalImageDataLoader instance
    """
    return MedicalImageDataLoader(img_size=img_size)
