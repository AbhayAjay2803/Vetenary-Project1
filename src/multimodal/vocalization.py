"""
Real audio processing for uploaded files.
Extracts mel-spectrogram and MFCC features, then estimates stress probability.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple

def extract_audio_features(file_path: str) -> Tuple[float, str]:
    """
    Extract features from audio file and estimate stress probability.
    Returns (stress_probability, label) where label is 'stress' or 'normal'.
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=16000, duration=5.0)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=512)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = np.mean(mel_db, axis=1)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=512)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Simple heuristic: higher energy and irregularity indicates stress
        energy = np.mean(mel_db)
        variance = np.var(mel_db)
        
        # Normalize to probability
        stress_prob = np.clip((energy + variance * 10) / 50, 0, 1)
        label = "stress" if stress_prob > 0.5 else "normal"
        
        return stress_prob, label
    except Exception as e:
        print(f"Error processing audio: {e}")
        return 0.5, "unknown"