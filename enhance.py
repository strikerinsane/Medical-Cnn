"""
Image Enhancement Module
Provides various preprocessing and augmentation techniques for medical images.
"""

import cv2
import numpy as np
from scipy import ndimage

def histogram_equalization(image):
    """
    Apply histogram equalization to improve contrast.

    Args:
        image: Input image (BGR or grayscale)
    Returns:
        Enhanced image
    """
    if len(image.shape) == 3:
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # Equalize the Y channel
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        # Convert back to BGR
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(image)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
    Returns:
        Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Apply CLAHE to L channel
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)

def edge_detection_sobel(image):
    """
    Apply Sobel edge detection.

    Args:
        image: Input image
    Returns:
        Edge-detected image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Sobel in X and Y directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Combine gradients
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel / sobel.max() * 255)

    # Convert back to 3 channels if needed
    if len(image.shape) == 3:
        sobel = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

    return sobel

def noise_reduction_median(image, kernel_size=5):
    """
    Apply median blur for noise reduction.

    Args:
        image: Input image
        kernel_size: Size of the median filter kernel (must be odd)
    Returns:
        Denoised image
    """
    return cv2.medianBlur(image, kernel_size)

def noise_reduction_gaussian(image, kernel_size=(5, 5), sigma=1.0):
    """
    Apply Gaussian blur for noise reduction.

    Args:
        image: Input image
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation
    Returns:
        Denoised image
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

def random_rotation(image, max_angle=20):
    """
    Randomly rotate image for data augmentation.

    Args:
        image: Input image
        max_angle: Maximum rotation angle in degrees
    Returns:
        Rotated image
    """
    angle = np.random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                             flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_REFLECT)
    return rotated

def random_flip(image):
    """
    Randomly flip image horizontally or vertically.

    Args:
        image: Input image
    Returns:
        Flipped image
    """
    flip_type = np.random.choice([0, 1, -1])  # 0: vertical, 1: horizontal, -1: both
    if flip_type == -1:
        return image  # No flip
    return cv2.flip(image, flip_type)

def random_scaling(image, scale_range=(0.9, 1.1)):
    """
    Randomly scale (zoom) image.

    Args:
        image: Input image
        scale_range: Tuple of (min_scale, max_scale)
    Returns:
        Scaled image
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Crop or pad to original size
    if scale > 1.0:
        # Crop center
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return resized[start_h:start_h+h, start_w:start_w+w]
    else:
        # Pad
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        if len(image.shape) == 3:
            padded = np.zeros((h, w, image.shape[2]), dtype=image.dtype)
        else:
            padded = np.zeros((h, w), dtype=image.dtype)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        return padded

def preprocess_image(image, apply_augmentation=False, apply_enhancement=True):
    """
    Complete preprocessing pipeline for medical images.

    Args:
        image: Input image
        apply_augmentation: Whether to apply data augmentation
        apply_enhancement: Whether to apply enhancement techniques
    Returns:
        Preprocessed image
    """
    # Apply enhancement techniques
    if apply_enhancement:
        # Noise reduction
        image = noise_reduction_gaussian(image, kernel_size=(3, 3), sigma=1.0)

        # CLAHE for better contrast
        image = apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))

    # Apply augmentation (only during training)
    if apply_augmentation:
        if np.random.random() > 0.5:
            image = random_rotation(image, max_angle=15)
        if np.random.random() > 0.5:
            image = random_flip(image)
        if np.random.random() > 0.5:
            image = random_scaling(image, scale_range=(0.95, 1.05))

    return image

def normalize_image(image):
    """
    Normalize image to [0, 1] range.

    Args:
        image: Input image
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0
