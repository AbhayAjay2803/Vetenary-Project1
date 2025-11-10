# train_models.py
import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import xgboost as xgb

# Add src to path
sys.path.append('src')

from src.data_loader import VeterinaryDatasetLoader
from src.feature_engineer import VeterinaryFeatureEngineer
from src.models import ImprovedStructuredClinicalTransformer, VeterinaryLSTM
from src.trainer import ImprovedSCTTrainer, LSTMTrainer

def ensure_directory(directory):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_traditional_models(X, y, models_dir):
    """Train traditional ML models"""
    print("[] Training traditional ML models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature selection
    selector = SelectKBest(f_classif, k=min(50, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    models = {}
    results = {}
    
    # Random Forest
    print("[] Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,  # Increased for larger dataset
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_selected, y_train)
    models['RandomForest'] = rf_model
    
    # Evaluate RF
    y_pred_rf = rf_model.predict(X_test_selected)
    y_proba_rf = rf_model.predict_proba(X_test_selected)[:, 1]
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf, recall_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='binary', zero_division=0)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    
    results['RandomForest'] = {
        'accuracy': accuracy_rf,
        'precision': precision_rf,
        'recall': recall_rf,
        'f1_score': f1_rf,
        'auc_score': auc_rf
    }
    print(f" Random Forest - Accuracy: {accuracy_rf:.4f}, F1: {f1_rf:.4f}, AUC: {auc_rf:.4f}")
    
    # Neural Network
    print("[] Training Neural Network...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    nn_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),  # Increased for larger dataset
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1000,  # Increased for larger dataset
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15
    )
    nn_model.fit(X_train_scaled, y_train)
    models['NeuralNetwork'] = nn_model
    models['Scaler'] = scaler
    
    # Evaluate NN
    y_pred_nn = nn_model.predict(X_test_scaled)
    y_proba_nn = nn_model.predict_proba(X_test_scaled)[:, 1]
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    precision_nn, recall_nn, f1_nn, _ = precision_recall_fscore_support(y_test, y_pred_nn, average='binary', zero_division=0)
    auc_nn = roc_auc_score(y_test, y_proba_nn)
    
    results['NeuralNetwork'] = {
        'accuracy': accuracy_nn,
        'precision': precision_nn,
        'recall': recall_nn,
        'f1_score': f1_nn,
        'auc_score': auc_nn
    }
    print(f" Neural Network - Accuracy: {accuracy_nn:.4f}, F1: {f1_nn:.4f}, AUC: {auc_nn:.4f}")
    
    # XGBoost
    print("[] Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,  # Increased for larger dataset
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # Evaluate XGBoost
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    precision_xgb, recall_xgb, f1_xgb, _ = precision_recall_fscore_support(y_test, y_pred_xgb, average='binary', zero_division=0)
    auc_xgb = roc_auc_score(y_test, y_proba_xgb)
    
    results['XGBoost'] = {
        'accuracy': accuracy_xgb,
        'precision': precision_xgb,
        'recall': recall_xgb,
        'f1_score': f1_xgb,
        'auc_score': auc_xgb
    }
    print(f" XGBoost - Accuracy: {accuracy_xgb:.4f}, F1: {f1_xgb:.4f}, AUC: {auc_xgb:.4f}")
    
    # Save models
    joblib.dump(rf_model, os.path.join(models_dir, 'randomforest.joblib'))
    joblib.dump(nn_model, os.path.join(models_dir, 'neuralnetwork.joblib'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgboost.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    
    return models, results

def main():
    """Main training function"""
    print("=" * 80)
    print("VETERINARY HEALTH ASSESSMENT - COMPREHENSIVE MODEL TRAINING (25,000 SAMPLES)")
    print("=" * 80)
    
    # Create models directory
    models_dir = 'models'
    ensure_directory(models_dir)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Step 1: Load and preprocess data
    print("\n[STEP 1] Loading and preprocessing veterinary dataset...")
    data_loader = VeterinaryDatasetLoader()
    df = data_loader.create_comprehensive_dataset(n_samples=25000)  # CHANGED: 25,000 samples
    processed_df = data_loader.preprocess_data(df)
    
    # Step 2: Prepare features
    print("\n[STEP 2] Preparing features for all models...")
    feature_engineer = VeterinaryFeatureEngineer(data_loader)
    
    # Traditional ML features
    X_traditional, y_traditional = feature_engineer.prepare_traditional_features(processed_df)
    print(f" Traditional features shape: {X_traditional.shape}")
    
    # SCT features
    sct_features = feature_engineer.prepare_sct_features(processed_df)
    print(f" SCT features prepared: {len(sct_features['symptom_indices'])} samples")
    
    # Step 3: Train traditional models
    print("\n[STEP 3] Training traditional ML models...")
    traditional_models, traditional_results = train_traditional_models(X_traditional, y_traditional, models_dir)
    
    # Step 4: Train deep learning models
    print("\n[STEP 4] Training Deep Learning Models...")
    
    # Improved SCT
    print("[] Training Improved Structured Clinical Transformer...")
    sct_trainer = ImprovedSCTTrainer(feature_engineer, data_loader)
    sct_results = sct_trainer.train_improved_sct(
        sct_features, epochs=35, learning_rate=1.5e-3, batch_size=64  # Increased epochs for larger dataset
    )
    
    # Save SCT model
    torch.save({
        'model_state_dict': sct_trainer.best_model_state,
        'model_config': sct_trainer.model_config,
        'training_metrics': sct_results.get('ImprovedSCT', {})
    }, os.path.join(models_dir, 'sct_model.pth'))
    
    # LSTM
    print("[] Training LSTM Model...")
    lstm_trainer = LSTMTrainer(feature_engineer, data_loader)
    lstm_results = lstm_trainer.train_lstm(
        sct_features, epochs=25, learning_rate=1e-3, batch_size=64  # Increased epochs for larger dataset
    )
    
    # Save LSTM model
    torch.save({
        'model_state_dict': lstm_trainer.best_model_state,
        'model_config': {
            'num_symptoms': len(feature_engineer.symptom_to_idx),
            'num_animals': len(data_loader.all_animals),
            'num_breeds': len(data_loader.le_breed.classes_),
            'num_ages': len(data_loader.le_age.classes_),
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3
        },
        'training_metrics': lstm_results.get('LSTM', {})
    }, os.path.join(models_dir, 'lstm_model.pth'))
    
    # Step 5: Save encoders and feature mappings
    print("\n[STEP 5] Saving encoders and feature mappings...")
    encoders_data = {
        'symptom_to_idx': feature_engineer.symptom_to_idx,
        'cluster_to_idx': feature_engineer.cluster_to_idx,
        'le_animal': data_loader.le_animal,
        'le_breed': data_loader.le_breed,
        'le_age': data_loader.le_age,
        'symptom_severity_weights': data_loader.symptom_severity_weights,
        'symptom_clusters': data_loader.symptom_clusters,
        'all_animals': data_loader.all_animals,
        'all_symptoms': data_loader.all_symptoms
    }
    joblib.dump(encoders_data, os.path.join(models_dir, 'encoders.joblib'))
    
    # Step 6: Generate performance summary
    print("\n[STEP 6] Generating performance summary...")
    
    # Collect all results
    all_results = {}
    all_results.update(traditional_results)
    all_results['ImprovedSCT'] = sct_results.get('ImprovedSCT', {})
    all_results['LSTM'] = lstm_results.get('LSTM', {})
    
    # Create results dataframe
    results_data = []
    for model_name, metrics in all_results.items():
        results_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
            'Precision': f"{metrics.get('precision', 0):.4f}",
            'Recall': f"{metrics.get('recall', 0):.4f}",
            'F1-Score': f"{metrics.get('f1_score', 0):.4f}",
            'AUC': f"{metrics.get('auc_score', 0):.4f}"
        })
    
    results_df = pd.DataFrame(results_data)
    print("\n" + "=" * 100)
    print("COMPREHENSIVE MODEL PERFORMANCE COMPARISON (25,000 SAMPLES)")
    print("=" * 100)
    print(results_df.to_string(index=False))
    
    # Find best model
    best_idx = results_df['F1-Score'].astype(float).idxmax()
    best_model = results_df.iloc[best_idx]
    
    print(f"\n[] BEST PERFORMING MODEL: {best_model['Model']}")
    print(f"[] F1-Score: {best_model['F1-Score']}, Accuracy: {best_model['Accuracy']}, AUC: {best_model['AUC']}")
    
    # Save performance results
    performance_data = {
        'training_timestamp': pd.Timestamp.now().isoformat(),
        'dataset_size': len(processed_df),
        'class_balance': f"{processed_df['target'].mean()*100:.1f}% positive",
        'detailed_results': results_data,
        'best_model': best_model.to_dict()
    }
    
    joblib.dump(performance_data, os.path.join(models_dir, 'performance_summary.joblib'))
    
    print(f"\n[] All models saved to: {models_dir}/")
    print("[] Training completed successfully!")

if __name__ == "__main__":
    main()