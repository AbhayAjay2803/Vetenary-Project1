"""
Synthetic multimodal dataset generator – strong signal version.
"""

import numpy as np
import random
from pathlib import Path
from typing import Dict, Optional

class SyntheticMultimodalGenerator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)

    def generate_audio_features(self, animal_type: str = "dog", stress_level: float = None) -> np.ndarray:
        if stress_level is None:
            stress_level = random.uniform(0, 1)
        animal_patterns = {
            "dog": {"pitch_mean": 300, "pitch_std": 80, "energy_mean": 0.6, "mfcc_scale": 0.5},
            "cat": {"pitch_mean": 500, "pitch_std": 100, "energy_mean": 0.5, "mfcc_scale": 0.6},
            "cow": {"pitch_mean": 200, "pitch_std": 50, "energy_mean": 0.7, "mfcc_scale": 0.4},
            "horse": {"pitch_mean": 250, "pitch_std": 60, "energy_mean": 0.65, "mfcc_scale": 0.45},
            "rabbit": {"pitch_mean": 450, "pitch_std": 90, "energy_mean": 0.55, "mfcc_scale": 0.55},
            "sheep": {"pitch_mean": 220, "pitch_std": 55, "energy_mean": 0.68, "mfcc_scale": 0.42},
            "goat": {"pitch_mean": 280, "pitch_std": 70, "energy_mean": 0.63, "mfcc_scale": 0.48},
            "default": {"pitch_mean": 400, "pitch_std": 90, "energy_mean": 0.55, "mfcc_scale": 0.5}
        }
        pattern = animal_patterns.get(animal_type.lower(), animal_patterns["default"])
        stress_factor = 1 + stress_level * 0.6
        mel_bands = np.random.normal(
            loc=pattern["energy_mean"] * (0.8 + stress_level * 0.5),
            scale=0.1 * (1 + stress_level * 0.8),
            size=64
        )
        mel_bands = np.clip(mel_bands, 0, 1)
        mfccs = np.random.normal(
            loc=pattern["mfcc_scale"] * stress_factor,
            scale=0.1 * (1 + stress_level),
            size=13
        )
        mfccs = np.clip(mfccs, -1, 1)
        mod = np.sin(np.linspace(0, 2*np.pi, 64)) * stress_level * 0.1
        mel_bands += mod
        features = np.concatenate([mel_bands, mfccs])
        return features.astype(np.float32)

    def generate_thermal_features(self, animal_type: str = "dog", abnormality: float = None) -> np.ndarray:
        if abnormality is None:
            abnormality = random.uniform(0, 1)

        species_offset = {
            "dog": 0.0, "cat": 0.02, "cow": -0.03, "horse": -0.01,
            "rabbit": 0.04, "sheep": -0.02, "goat": -0.01, "default": 0.0
        }.get(animal_type.lower(), 0.0)

        base_mean = 0.5 + species_offset + random.uniform(-0.05, 0.05)
        mean_temp = base_mean + abnormality * 0.3
        mean_temp = np.clip(mean_temp, 0.4, 0.85)

        base_max = 0.65 + species_offset + random.uniform(-0.05, 0.05)
        max_temp = base_max + abnormality * 0.3
        max_temp = np.clip(max_temp, 0.55, 0.98)

        base_min = 0.45 + species_offset + random.uniform(-0.05, 0.05)
        min_temp = base_min - abnormality * 0.1
        min_temp = np.clip(min_temp, 0.3, 0.55)

        base_std = 0.07 + random.uniform(0, 0.03)
        std_temp = base_std + abnormality * 0.1
        std_temp = np.clip(std_temp, 0.04, 0.18)

        bins_centers = np.linspace(0, 1, 32)
        normal_hist = np.exp(-0.5 * ((bins_centers - mean_temp) / (std_temp * 0.8))**2)
        if abnormality > 0.2:
            hot_center = mean_temp + abnormality * 0.2
            hot_std = 0.05
            hot_hist = np.exp(-0.5 * ((bins_centers - hot_center) / hot_std)**2)
            normal_weight = 1 - abnormality * 0.8
            hot_weight = abnormality * 0.8
            hist = normal_weight * normal_hist + hot_weight * hot_hist
        else:
            hist = normal_hist

        hist = hist / hist.sum()
        features = np.array([mean_temp, max_temp, min_temp, std_temp])
        features = np.concatenate([features, hist])
        return features.astype(np.float32)

    def generate_gait_features(self, animal_type: str = "dog", lameness: float = None) -> np.ndarray:
        """
        Strong gait signal: lameness causes large, linear asymmetry.
        """
        if lameness is None:
            lameness = random.uniform(0, 1)

        landmarks = np.zeros(66)
        left_hip, right_hip = 23, 24
        left_knee, right_knee = 25, 26
        left_ankle, right_ankle = 27, 28

        height_map = {"dog": 0.6, "cat": 0.55, "cow": 0.7, "horse": 0.65,
                      "rabbit": 0.5, "sheep": 0.68, "goat": 0.62, "default": 0.6}
        height = height_map.get(animal_type.lower(), 0.6)

        # baseline healthy positions
        for i in range(0, 66, 2):
            landmarks[i] = random.uniform(0.2, 0.8)
            landmarks[i+1] = random.uniform(0.2, 0.9) * (height / 0.6)

        # small natural asymmetry
        landmarks[left_hip*2+1] += random.uniform(-0.005, 0.005)
        landmarks[right_hip*2+1] += random.uniform(-0.005, 0.005)
        landmarks[left_knee*2+1] += random.uniform(-0.005, 0.005)
        landmarks[right_knee*2+1] += random.uniform(-0.005, 0.005)
        landmarks[left_ankle*2+1] += random.uniform(-0.005, 0.005)
        landmarks[right_ankle*2+1] += random.uniform(-0.005, 0.005)

        # lameness – large, linear effect
        if lameness > 0:
            # decide limb: left for lameness > 0.5, else right (with a small random flip)
            if lameness > 0.6:
                limb = "left"
            elif lameness > 0.4:
                limb = random.choice(["left", "right"])
            else:
                limb = "right"

            intensity = lameness * 0.35  # up to 0.35 displacement
            if limb == "left":
                landmarks[left_hip*2+1] += intensity
                landmarks[left_knee*2+1] += intensity * 1.3
                landmarks[left_ankle*2+1] += intensity * 1.6
                landmarks[left_ankle*2] -= intensity * 0.12
            else:
                landmarks[right_hip*2+1] += intensity
                landmarks[right_knee*2+1] += intensity * 1.3
                landmarks[right_ankle*2+1] += intensity * 1.6
                landmarks[right_ankle*2] -= intensity * 0.12

        # add tiny noise to avoid perfect linearity
        landmarks += np.random.normal(0, 0.003, size=66)
        landmarks = np.clip(landmarks, 0, 1)
        return landmarks.astype(np.float32)

    def generate_label(self, stress: float, thermal_abnorm: float, lameness: float) -> Dict[str, float]:
        return {"stress": stress, "thermal_abnorm": thermal_abnorm, "lameness": lameness,
                "risk": 0.35 * stress + 0.35 * thermal_abnorm + 0.3 * lameness}

    def generate_dataset(self, num_samples: int = 1000, save_dir: Optional[Path] = None) -> Dict:
        data = {
            "audio_features": [],
            "thermal_features": [],
            "gait_features": [],
            "labels": []
        }
        animal_types = ["dog", "cat", "cow", "horse", "rabbit", "sheep", "goat"]
        for _ in range(num_samples):
            animal = random.choice(animal_types)
            stress = random.uniform(0, 1)
            thermal_abnorm = random.uniform(0, 1)
            lameness = random.uniform(0, 1)

            data["audio_features"].append(self.generate_audio_features(animal, stress))
            data["thermal_features"].append(self.generate_thermal_features(animal, thermal_abnorm))
            data["gait_features"].append(self.generate_gait_features(animal, lameness))
            data["labels"].append(self.generate_label(stress, thermal_abnorm, lameness))

        data["audio_features"] = np.array(data["audio_features"])
        data["thermal_features"] = np.array(data["thermal_features"])
        data["gait_features"] = np.array(data["gait_features"])

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            np.savez(save_dir / "synthetic_multimodal.npz", **data)
            print(f"✅ Synthetic dataset saved to {save_dir / 'synthetic_multimodal.npz'}")
        return data

_generator = None
def get_generator() -> SyntheticMultimodalGenerator:
    global _generator
    if _generator is None:
        _generator = SyntheticMultimodalGenerator()
    return _generator