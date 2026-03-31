"""
Real audio processing for uploaded files.
Extracts mel-spectrogram and MFCC features for model input.
"""

import numpy as np
import librosa
from typing import Tuple

def extract_audio_feature_vector(file_path: str) -> np.ndarray:
    """
    Extract full feature vector (77-dim) from audio file.
    Returns array of shape (77,).
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=16000, duration=5.0)
        
        # Extract mel spectrogram (64 bands)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=512)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = np.mean(mel_db, axis=1)  # 64
        
        # Extract MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=512)
        mfcc_mean = np.mean(mfccs, axis=1)  # 13
        
        features = np.concatenate([mel_mean, mfcc_mean])
        return features.astype(np.float32)
    except Exception as e:
        print(f"Error processing audio: {e}")
        return np.zeros(77, dtype=np.float32)

def extract_audio_features(file_path: str) -> Tuple[float, str]:
    """
    Legacy function returning heuristic probability.
    Kept for backward compatibility.
    """
    feat = extract_audio_feature_vector(file_path)
    # Simple heuristic: energy and variance
    energy = np.mean(feat[:64])
    variance = np.var(feat[:64])
    stress_prob = np.clip((energy + variance * 10) / 50, 0, 1)
    label = "stress" if stress_prob > 0.5 else "normal"
    return stress_prob, label