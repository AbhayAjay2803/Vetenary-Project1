"""
Real thermal image processing for uploaded images.
Extracts temperature statistics for model input.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple

def extract_thermal_feature_vector(image_path: str) -> np.ndarray:
    """
    Extract full feature vector (36-dim) from thermal image.
    Returns array of shape (36,): [mean, max, min, std, 32-bin histogram].
    """
    try:
        # Load image as grayscale (thermal)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.array(Image.open(image_path).convert('L'))
        
        img = cv2.resize(img, (224, 224))
        
        mean_temp = img.mean() / 255.0
        max_temp = img.max() / 255.0
        min_temp = img.min() / 255.0
        std_temp = img.std() / 255.0
        
        # Histogram (32 bins)
        hist, _ = np.histogram(img, bins=32, range=(0, 255))
        hist = hist / hist.sum()
        
        features = np.array([mean_temp, max_temp, min_temp, std_temp])
        features = np.concatenate([features, hist])
        return features.astype(np.float32)
    except Exception as e:
        print(f"Error processing thermal image: {e}")
        return np.zeros(36, dtype=np.float32)

def extract_thermal_features(image_path: str) -> Tuple[float, str]:
    """
    Legacy function returning heuristic probability.
    """
    feat = extract_thermal_feature_vector(image_path)
    abnormality = (feat[0] * 0.4 + feat[1] * 0.4 + feat[3] * 0.2)
    abnormality = np.clip(abnormality, 0, 1)
    label = "abnormal" if abnormality > 0.6 else "normal"
    return abnormality, label