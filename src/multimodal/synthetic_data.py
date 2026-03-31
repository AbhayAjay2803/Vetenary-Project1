"""
Synthetic multimodal dataset generator for VHAS.
Creates realistic audio features, thermal patterns, and gait landmarks for training/evaluation.
"""

import numpy as np
import random
from pathlib import Path
from typing import Dict, Tuple, List, Optional

class SyntheticMultimodalGenerator:
    """Generates realistic synthetic data for audio, thermal, and gait modalities."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_audio_features(self, animal_type: str = "dog", stress_level: float = None) -> np.ndarray:
        """
        Generate synthetic mel-spectrogram and MFCC features.
        Returns feature vector of length 64+13 = 77.
        """
        if stress_level is None:
            stress_level = random.uniform(0, 1)
        
        # Base patterns for different animals
        animal_patterns = {
            "dog": {"pitch_mean": 300, "pitch_std": 80, "energy_mean": 0.6},
            "cat": {"pitch_mean": 500, "pitch_std": 100, "energy_mean": 0.5},
            "cow": {"pitch_mean": 200, "pitch_std": 50, "energy_mean": 0.7},
            "horse": {"pitch_mean": 250, "pitch_std": 60, "energy_mean": 0.65},
            "default": {"pitch_mean": 400, "pitch_std": 90, "energy_mean": 0.55}
        }
        pattern = animal_patterns.get(animal_type.lower(), animal_patterns["default"])
        
        # Stress modifies features: higher pitch, higher energy, more irregularity
        stress_factor = 1 + stress_level * 0.5  # 1.0 to 1.5
        
        # Generate mel-spectrogram (64 bands)
        mel_bands = np.random.normal(
            loc=pattern["energy_mean"] * (0.8 + stress_level * 0.4),
            scale=0.1 * (1 + stress_level),
            size=64
        )
        mel_bands = np.clip(mel_bands, 0, 1)
        
        # Generate MFCCs (13 coefficients)
        mfccs = np.random.normal(
            loc=pattern["pitch_mean"] / 1000 * stress_factor,
            scale=0.05 * (1 + stress_level),
            size=13
        )
        mfccs = np.clip(mfccs, -1, 1)
        
        features = np.concatenate([mel_bands, mfccs])
        return features.astype(np.float32)
    
    def generate_thermal_features(self, animal_type: str = "dog", abnormality: float = None) -> np.ndarray:
        """
        Generate synthetic thermal image statistics.
        Returns feature vector: [mean_temp, max_temp, min_temp, std_temp, histogram_bins(32)] = 36 total.
        """
        if abnormality is None:
            abnormality = random.uniform(0, 1)
        
        # Base temperature ranges (normalized 0-1)
        base_mean = 0.5
        base_max = 0.6
        base_min = 0.4
        base_std = 0.05
        
        if abnormality > 0.5:
            # Inflamed/infected area: higher max, higher mean, wider spread
            inflame_factor = 1 + (abnormality - 0.5) * 2
            mean_temp = base_mean + 0.1 * inflame_factor
            max_temp = min(0.95, base_max + 0.2 * inflame_factor)
            min_temp = base_min - 0.05 * (abnormality - 0.5)
            std_temp = base_std + 0.08 * (abnormality - 0.5)
        else:
            mean_temp = base_mean - 0.05 * (0.5 - abnormality)
            max_temp = base_max - 0.02 * (0.5 - abnormality)
            min_temp = base_min + 0.02 * (0.5 - abnormality)
            std_temp = base_std - 0.02 * (0.5 - abnormality)
        
        # Ensure bounds
        mean_temp = np.clip(mean_temp, 0.2, 0.8)
        max_temp = np.clip(max_temp, 0.3, 0.95)
        min_temp = np.clip(min_temp, 0.1, 0.6)
        std_temp = np.clip(std_temp, 0.02, 0.2)
        
        # Generate histogram (32 bins) using a skewed distribution if abnormal
        if abnormality > 0.6:
            # Skewed histogram: more mass in high-temperature bins
            bins = np.random.gamma(shape=2, scale=0.1, size=32)
            bins = bins / bins.sum()
            # Shift mass to higher bins
            shift = int(abnormality * 20)
            bins = np.roll(bins, shift)
        else:
            # Normal-ish histogram
            bins = np.random.normal(loc=0.5, scale=0.1, size=32)
            bins = np.exp(bins)
            bins = bins / bins.sum()
        
        features = np.array([mean_temp, max_temp, min_temp, std_temp])
        features = np.concatenate([features, bins])
        return features.astype(np.float32)
    
    def generate_gait_features(self, animal_type: str = "dog", lameness: float = None) -> np.ndarray:
        """
        Generate synthetic pose landmark features (66 values: 33 landmarks * 2 (x,y)).
        """
        if lameness is None:
            lameness = random.uniform(0, 1)
        
        # Base symmetrical pose
        landmarks = np.zeros(66)
        # Landmark indices for key points
        left_hip = 23
        right_hip = 24
        left_knee = 25
        right_knee = 26
        left_ankle = 27
        right_ankle = 28
        
        # Symmetrical positions (x, y) normalised
        for i in range(0, 66, 2):
            landmarks[i] = random.uniform(0.2, 0.8)  # x
            landmarks[i+1] = random.uniform(0.2, 0.9)  # y
        
        # Add lameness asymmetry
        if lameness > 0.5:
            # Left limb lameness: shift left hip, knee, ankle down (higher y) and reduce stride
            asymmetry = (lameness - 0.5) * 2
            landmarks[left_hip*2+1] += asymmetry * 0.05
            landmarks[left_knee*2+1] += asymmetry * 0.1
            landmarks[left_ankle*2+1] += asymmetry * 0.15
        elif lameness < 0.3:
            # Right limb lameness
            asymmetry = (0.3 - lameness) * 2
            landmarks[right_hip*2+1] += asymmetry * 0.05
            landmarks[right_knee*2+1] += asymmetry * 0.1
            landmarks[right_ankle*2+1] += asymmetry * 0.15
        
        # Clip to [0,1]
        landmarks = np.clip(landmarks, 0, 1)
        return landmarks.astype(np.float32)
    
    def generate_label(self, stress: float, thermal_abnorm: float, lameness: float) -> Dict[str, float]:
        """Generate a combined risk label based on multimodal features."""
        # Weighted risk: audio 0.35, thermal 0.35, gait 0.3
        risk = 0.35 * stress + 0.35 * thermal_abnorm + 0.3 * lameness
        return {"stress": stress, "thermal_abnorm": thermal_abnorm, "lameness": lameness, "risk": risk}
    
    def generate_dataset(self, num_samples: int = 1000, save_dir: Optional[Path] = None) -> Dict:
        """Generate a full synthetic dataset."""
        data = {
            "audio_features": [],
            "thermal_features": [],
            "gait_features": [],
            "labels": []
        }
        animal_types = ["dog", "cat", "cow", "horse", "rabbit", "sheep", "goat"]
        
        for i in range(num_samples):
            animal = random.choice(animal_types)
            stress = random.uniform(0, 1)
            thermal_abnorm = random.uniform(0, 1)
            lameness = random.uniform(0, 1)
            
            audio_feat = self.generate_audio_features(animal, stress)
            thermal_feat = self.generate_thermal_features(animal, thermal_abnorm)
            gait_feat = self.generate_gait_features(animal, lameness)
            label = self.generate_label(stress, thermal_abnorm, lameness)
            
            data["audio_features"].append(audio_feat)
            data["thermal_features"].append(thermal_feat)
            data["gait_features"].append(gait_feat)
            data["labels"].append(label)
        
        # Convert to numpy arrays
        data["audio_features"] = np.array(data["audio_features"])
        data["thermal_features"] = np.array(data["thermal_features"])
        data["gait_features"] = np.array(data["gait_features"])
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            np.savez(save_dir / "synthetic_multimodal.npz", **data)
            print(f"✅ Synthetic dataset saved to {save_dir / 'synthetic_multimodal.npz'}")
        
        return data

# Singleton generator for reuse
_generator = None
def get_generator() -> SyntheticMultimodalGenerator:
    global _generator
    if _generator is None:
        _generator = SyntheticMultimodalGenerator()
    return _generator