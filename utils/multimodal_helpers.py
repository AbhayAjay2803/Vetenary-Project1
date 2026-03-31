"""
Multimodal feature extraction for real uploads and synthetic fallback.
"""

import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple

# Import processors (these work on real files)
from src.multimodal.vocalization import extract_audio_features
from src.multimodal.thermal import extract_thermal_features
from src.multimodal.gait import extract_gait_features

# Import synthetic generator for fallback
from src.multimodal.synthetic_data import get_generator

class MultimodalFeatureExtractor:
    def __init__(self, model_dir: str = "models/multimodal/"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_gen = get_generator()
    
    def get_features(self,
                     audio_path: Optional[str] = None,
                     thermal_path: Optional[str] = None,
                     video_path: Optional[str] = None,
                     animal_type: str = "dog") -> np.ndarray:
        """
        Extract features from real uploaded files.
        If a file is not provided, uses synthetic features as fallback.
        Returns 6-dimensional vector: [stress_prob, stress_label, thermal_prob, thermal_label, gait_prob, gait_label]
        """
        features = []
        
        # Audio features
        if audio_path and os.path.exists(audio_path):
            try:
                stress_prob, stress_label = extract_audio_features(audio_path)
                features.extend([stress_prob, 1.0 if stress_label == "stress" else 0.0])
            except Exception as e:
                print(f"⚠️ Audio processing failed: {e}, using synthetic fallback")
                synth = self.synthetic_gen.generate_audio_features(animal_type)
                stress_prob = float(np.mean(synth))  # crude proxy
                features.extend([stress_prob, 0.5])
        else:
            # Synthetic fallback
            synth = self.synthetic_gen.generate_audio_features(animal_type)
            stress_prob = float(np.mean(synth))
            features.extend([stress_prob, 0.5])
        
        # Thermal features
        if thermal_path and os.path.exists(thermal_path):
            try:
                thermal_prob, thermal_label = extract_thermal_features(thermal_path)
                features.extend([thermal_prob, 1.0 if thermal_label == "abnormal" else 0.0])
            except Exception as e:
                print(f"⚠️ Thermal processing failed: {e}, using synthetic fallback")
                synth = self.synthetic_gen.generate_thermal_features(animal_type)
                thermal_prob = float(np.mean(synth[:4]))
                features.extend([thermal_prob, 0.5])
        else:
            synth = self.synthetic_gen.generate_thermal_features(animal_type)
            thermal_prob = float(np.mean(synth[:4]))
            features.extend([thermal_prob, 0.5])
        
        # Gait features
        if video_path and os.path.exists(video_path):
            try:
                gait_prob, gait_label = extract_gait_features(video_path)
                features.extend([gait_prob, 1.0 if gait_label == "lameness" else 0.0])
            except Exception as e:
                print(f"⚠️ Video processing failed: {e}, using synthetic fallback")
                synth = self.synthetic_gen.generate_gait_features(animal_type)
                gait_prob = float(np.mean(synth))
                features.extend([gait_prob, 0.5])
        else:
            synth = self.synthetic_gen.generate_gait_features(animal_type)
            gait_prob = float(np.mean(synth))
            features.extend([gait_prob, 0.5])
        
        return np.array(features, dtype=np.float32)

# Global instance
_extractor = None
def get_multimodal_features(audio_path=None, thermal_path=None, video_path=None, animal_type="dog") -> np.ndarray:
    global _extractor
    if _extractor is None:
        _extractor = MultimodalFeatureExtractor()
    return _extractor.get_features(audio_path, thermal_path, video_path, animal_type)