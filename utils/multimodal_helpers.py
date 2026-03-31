import numpy as np
import torch
import torch.nn as nn
import os
from pathlib import Path
from typing import Optional

from src.multimodal.vocalization import extract_audio_feature_vector, extract_audio_features
from src.multimodal.thermal import extract_thermal_feature_vector, extract_thermal_features
from src.multimodal.gait import extract_gait_feature_vector, extract_gait_features
from src.multimodal.synthetic_data import get_generator

# Define SimpleMLP here to avoid import issues
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class MultimodalFeatureExtractor:
    def __init__(self, model_dir: str = "models/multimodal/"):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_model = None
        self.thermal_model = None
        self.gait_model = None
        self.synthetic_gen = get_generator()
        self._load_models()
    
    def _load_models(self):
        # Load audio model
        audio_path = self.model_dir / "audio_model.pth"
        if audio_path.exists():
            try:
                self.audio_model = SimpleMLP(input_dim=77)
                self.audio_model.load_state_dict(torch.load(audio_path, map_location=self.device))
                self.audio_model.to(self.device)
                self.audio_model.eval()
                print("✅ Loaded audio model")
            except Exception as e:
                print(f"⚠️ Audio model load error: {e}")
        
        # Load thermal model
        thermal_path = self.model_dir / "thermal_model.pth"
        if thermal_path.exists():
            try:
                self.thermal_model = SimpleMLP(input_dim=36)
                self.thermal_model.load_state_dict(torch.load(thermal_path, map_location=self.device))
                self.thermal_model.to(self.device)
                self.thermal_model.eval()
                print("✅ Loaded thermal model")
            except Exception as e:
                print(f"⚠️ Thermal model load error: {e}")
        
        # Load gait model
        gait_path = self.model_dir / "gait_model.pth"
        if gait_path.exists():
            try:
                self.gait_model = SimpleMLP(input_dim=66)
                self.gait_model.load_state_dict(torch.load(gait_path, map_location=self.device))
                self.gait_model.to(self.device)
                self.gait_model.eval()
                print("✅ Loaded gait model")
            except Exception as e:
                print(f"⚠️ Gait model load error: {e}")
    
    def _predict_with_model(self, model, features):
        if model is None:
            return 0.5
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            return model(x).item()
    
    def get_features(self,
                     audio_path: Optional[str] = None,
                     thermal_path: Optional[str] = None,
                     video_path: Optional[str] = None,
                     animal_type: str = "dog") -> np.ndarray:
        features = []
        
        # Audio
        if audio_path and os.path.exists(audio_path):
            try:
                audio_vec = extract_audio_feature_vector(audio_path)
                if self.audio_model is not None:
                    prob = self._predict_with_model(self.audio_model, audio_vec)
                    label = "stress" if prob > 0.5 else "normal"
                else:
                    prob, label = extract_audio_features(audio_path)
                features.extend([prob, 1.0 if label == "stress" else 0.0])
            except Exception as e:
                print(f"⚠️ Audio error: {e}")
                synth = self.synthetic_gen.generate_audio_features(animal_type)
                prob = float(np.mean(synth))
                features.extend([prob, 0.5])
        else:
            synth = self.synthetic_gen.generate_audio_features(animal_type)
            prob = float(np.mean(synth))
            features.extend([prob, 0.5])
        
        # Thermal
        if thermal_path and os.path.exists(thermal_path):
            try:
                thermal_vec = extract_thermal_feature_vector(thermal_path)
                if self.thermal_model is not None:
                    prob = self._predict_with_model(self.thermal_model, thermal_vec)
                    label = "abnormal" if prob > 0.5 else "normal"
                else:
                    prob, label = extract_thermal_features(thermal_path)
                features.extend([prob, 1.0 if label == "abnormal" else 0.0])
            except Exception as e:
                print(f"⚠️ Thermal error: {e}")
                synth = self.synthetic_gen.generate_thermal_features(animal_type)
                prob = float(np.mean(synth[:4]))
                features.extend([prob, 0.5])
        else:
            synth = self.synthetic_gen.generate_thermal_features(animal_type)
            prob = float(np.mean(synth[:4]))
            features.extend([prob, 0.5])
        
        # Gait
        if video_path and os.path.exists(video_path):
            try:
                gait_vec = extract_gait_feature_vector(video_path)
                if self.gait_model is not None:
                    prob = self._predict_with_model(self.gait_model, gait_vec)
                    label = "lameness" if prob > 0.5 else "normal"
                else:
                    prob, label = extract_gait_features(video_path)
                features.extend([prob, 1.0 if label == "lameness" else 0.0])
            except Exception as e:
                print(f"⚠️ Gait error: {e}")
                synth = self.synthetic_gen.generate_gait_features(animal_type)
                prob = float(np.mean(synth))
                features.extend([prob, 0.5])
        else:
            synth = self.synthetic_gen.generate_gait_features(animal_type)
            prob = float(np.mean(synth))
            features.extend([prob, 0.5])
        
        return np.array(features, dtype=np.float32)

_extractor = None
def get_multimodal_features(audio_path=None, thermal_path=None, video_path=None, animal_type="dog") -> np.ndarray:
    global _extractor
    if _extractor is None:
        _extractor = MultimodalFeatureExtractor()
    return _extractor.get_features(audio_path, thermal_path, video_path, animal_type)