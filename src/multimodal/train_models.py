"""
Train models – strong signal, larger dataset, increased capacity for gait.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
from .synthetic_data import get_generator
from src.multimodal.train_models import SimpleMLP

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

def compute_metrics(y_true, y_pred, y_prob):
    y_true_bin = (np.array(y_true) > 0.5).astype(int)
    y_pred_bin = (np.array(y_pred) > 0.5).astype(int)
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    auc = roc_auc_score(y_true_bin, y_prob) if len(np.unique(y_true_bin)) > 1 else 0.5
    return acc, prec, rec, f1, auc

def train_model(model, X_train, y_train, X_val, y_val, epochs=200, lr=0.001, weight_decay=1e-4, patience=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=12, verbose=True)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()
            val_prob = val_pred.cpu().numpy().flatten()
            val_true = y_val

        scheduler.step(val_loss)

        if (epoch+1) % 20 == 0:
            acc, prec, rec, f1, auc = compute_metrics(val_true, val_prob, val_prob)
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            print(f"      Val Metrics - Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    return model

def main():
    print("="*60)
    print("Training multimodal models – strong signal synthetic data")
    print("="*60)

    # generate 80,000 samples (more data for better generalization)
    gen = get_generator()
    data = gen.generate_dataset(num_samples=80000, save_dir="data/multimodal")

    stress_labels = np.array([l['stress'] for l in data['labels']])
    thermal_labels = np.array([l['thermal_abnorm'] for l in data['labels']])
    gait_labels = np.array([l['lameness'] for l in data['labels']])

    strat_stress = (stress_labels > 0.5).astype(int)
    strat_thermal = (thermal_labels > 0.5).astype(int)
    strat_gait = (gait_labels > 0.5).astype(int)

    X_audio_tr, X_audio_val, y_audio_tr, y_audio_val = train_test_split(
        data['audio_features'], stress_labels, test_size=0.2, random_state=42, stratify=strat_stress
    )
    X_thermal_tr, X_thermal_val, y_thermal_tr, y_thermal_val = train_test_split(
        data['thermal_features'], thermal_labels, test_size=0.2, random_state=42, stratify=strat_thermal
    )
    X_gait_tr, X_gait_val, y_gait_tr, y_gait_val = train_test_split(
        data['gait_features'], gait_labels, test_size=0.2, random_state=42, stratify=strat_gait
    )

    print("\n🎵 Training audio model...")
    audio_model = SimpleMLP(input_dim=77, hidden=64)
    audio_model = train_model(audio_model, X_audio_tr, y_audio_tr, X_audio_val, y_audio_val,
                              epochs=200, lr=0.001, weight_decay=1e-4, patience=30)

    print("\n🌡️ Training thermal model...")
    thermal_model = SimpleMLP(input_dim=36, hidden=64)
    thermal_model = train_model(thermal_model, X_thermal_tr, y_thermal_tr, X_thermal_val, y_thermal_val,
                                epochs=200, lr=0.001, weight_decay=1e-4, patience=30)

    print("\n🎬 Training gait model...")
    # Gait: larger capacity, longer training, lower regularization
    gait_model = SimpleMLP(input_dim=66, hidden=128, dropout=0.15)  # increased hidden, reduced dropout
    gait_model = train_model(gait_model, X_gait_tr, y_gait_tr, X_gait_val, y_gait_val,
                             epochs=300, lr=0.001, weight_decay=5e-5, patience=40)

    # Save models
    model_dir = Path("models/multimodal")
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(audio_model.state_dict(), model_dir / "audio_model.pth")
    torch.save(thermal_model.state_dict(), model_dir / "thermal_model.pth")
    torch.save(gait_model.state_dict(), model_dir / "gait_model.pth")

    print("\n📊 Final Validation Metrics:")
    for name, model, X_val, y_val in [
        ("Audio", audio_model, X_audio_val, y_audio_val),
        ("Thermal", thermal_model, X_thermal_val, y_thermal_val),
        ("Gait", gait_model, X_gait_val, y_gait_val)
    ]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            pred = model(X_t).cpu().numpy().flatten()
        acc, prec, rec, f1, auc = compute_metrics(y_val, pred, pred)
        print(f"  {name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    print("\n✅ Models saved to:", model_dir.absolute())
    print("="*60)

if __name__ == "__main__":
    main()