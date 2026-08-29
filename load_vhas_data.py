#!/usr/bin/env python3
"""
load_vhas_data.py

This script replicates the data loading and preprocessing steps used in the
VHAS training pipeline. It uses the exact same classes and parameters as
train_models.py, so the generated data and features are identical to those
used for model training.

The script:
1. Generates the synthetic dataset using VeterinaryDatasetLoader (30,000 samples).
2. Saves the raw dataset to 'vhas_dataset_25k.csv'.
3. Applies preprocessing (encoding, normalisation, target creation).
4. Prepares both traditional ML features and SCT/LSTM deep features.
5. Prints summary statistics and optionally saves processed outputs.

This file is provided for reproducibility and to allow reviewers to verify
the data preparation logic.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib

# Add the src directory to the path so we can import the data loader and feature engineer
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import VeterinaryDatasetLoader
from src.feature_engineer import VeterinaryFeatureEngineer

# Configuration
SAVE_RAW_CSV = True           # Save the raw dataset as vhas_dataset_25k.csv
SAVE_PROCESSED = False        # Set to True if you want to save processed data
OUTPUT_DIR = 'processed_data'

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    print("=" * 70)
    print("VHAS DATA LOADER & PREPROCESSOR (Replicates Training Pipeline)")
    print("=" * 70)

    # 1. Instantiate the data loader (same as in train_models.py)
    print("\n[Step 1] Loading and preprocessing the dataset...")
    data_loader = VeterinaryDatasetLoader()
    
    # 2. Create the dataset (same parameters: n_samples=30000)
    #    The filename is vhas_dataset_25k.csv for consistency with the paper, 
    #    but the actual number of samples is 30,000 as used in training.
    df = data_loader.create_comprehensive_dataset(n_samples=30000)
    
    # Save the raw dataset to CSV
    if SAVE_RAW_CSV:
        csv_filename = 'vhas_dataset_25k.csv'
        df.to_csv(csv_filename, index=False)
        print(f"\n[Raw dataset saved to {csv_filename}] (30,000 samples)")
    
    # 3. Preprocess the data (encoding, normalisation, target creation)
    processed_df = data_loader.preprocess_data(df)
    
    print("\n[Dataset Summary]")
    print(f"  Total records: {len(processed_df)}")
    print(f"  Animals: {len(data_loader.all_animals)}")
    print(f"  Symptoms: {len(data_loader.all_symptoms)}")
    print(f"  Positive ratio (High risk): {processed_df['target'].mean():.3f}")
    
    # 4. Prepare features using the feature engineer (same as in train_models.py)
    print("\n[Step 2] Preparing features for all model types...")
    feature_engineer = VeterinaryFeatureEngineer(data_loader)
    
    # Traditional ML features
    X_traditional, y_traditional = feature_engineer.prepare_traditional_features(processed_df)
    print(f"  Traditional features shape: {X_traditional.shape}")
    print(f"  Target shape: {y_traditional.shape}  |  Positive ratio: {y_traditional.mean():.3f}")
    
    # SCT / LSTM features (deep learning)
    sct_features = feature_engineer.prepare_sct_features(processed_df)
    print(f"  SCT features: {len(sct_features['symptom_indices'])} samples, "
          f"sequence length {sct_features['symptom_indices'].shape[1]}")
    
    # 5. Inspect a few samples of the processed data
    print("\n[Sample of processed data (first 5 rows)]")
    print(processed_df[['AnimalName', 'Breed', 'Age', 'Weight', 'Symptom_Count',
                        'Dangerous', 'Danger_Score', 'target']].head())
    
    # 6. Optional: save the processed data for later use
    if SAVE_PROCESSED:
        ensure_dir(OUTPUT_DIR)
        processed_df.to_csv(os.path.join(OUTPUT_DIR, 'processed_data.csv'), index=False)
        np.save(os.path.join(OUTPUT_DIR, 'X_traditional.npy'), X_traditional)
        np.save(os.path.join(OUTPUT_DIR, 'y_traditional.npy'), y_traditional)
        # Save SCT features as tensors or as numpy arrays
        sct_np = {k: v.numpy() if torch.is_tensor(v) else v for k, v in sct_features.items()}
        joblib.dump(sct_np, os.path.join(OUTPUT_DIR, 'sct_features.joblib'))
        print(f"\n[Processed data saved to {OUTPUT_DIR}/]")
    
    # 7. Also save the encoders and mappings for completeness
    encoders = {
        'le_animal': data_loader.le_animal,
        'le_breed': data_loader.le_breed,
        'le_age': data_loader.le_age,
        'symptom_severity_weights': data_loader.symptom_severity_weights,
        'symptom_clusters': data_loader.symptom_clusters,
        'all_animals': data_loader.all_animals,
        'all_symptoms': data_loader.all_symptoms,
    }
    if SAVE_PROCESSED:
        joblib.dump(encoders, os.path.join(OUTPUT_DIR, 'encoders.joblib'))
    
    print("\n[All data ready]")
    print("  - The raw dataset has been saved to 'vhas_dataset_25k.csv'.")
    print("  - The processed DataFrame is available as 'processed_df'.")
    print("  - Traditional features: X_traditional, y_traditional.")
    print("  - Deep learning features: sct_features dictionary.")
    print("  - Encoders and mappings are stored in the data_loader and feature_engineer objects.")
    print("\nYou can now use these objects for further analysis or model inspection.")

if __name__ == "__main__":
    main()