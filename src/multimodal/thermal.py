"""
Real thermal image processing for uploaded images.
Extracts temperature statistics and estimates abnormality.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple

def extract_thermal_features(image_path: str) -> Tuple[float, str]:
    """
    Extract features from thermal image and estimate abnormality probability.
    Returns (abnormality_probability, label) where label is 'abnormal' or 'normal'.
    """
    try:
        # Load image as grayscale (thermal)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Try with PIL
            img = np.array(Image.open(image_path).convert('L'))
        
        # Resize for consistency
        img = cv2.resize(img, (224, 224))
        
        # Temperature statistics (normalized to 0-1)
        mean_temp = img.mean() / 255.0
        max_temp = img.max() / 255.0
        std_temp = img.std() / 255.0
        
        # Abnormality heuristic: high mean and high max, high std
        abnormality = (mean_temp * 0.4 + max_temp * 0.4 + std_temp * 0.2)
        abnormality = np.clip(abnormality, 0, 1)
        
        label = "abnormal" if abnormality > 0.6 else "normal"
        return abnormality, label
    except Exception as e:
        print(f"Error processing thermal image: {e}")
        return 0.5, "unknown"