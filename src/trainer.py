# src/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

class ImprovedSCTTrainer:
    def __init__(self, feature_engineer, data_loader):
        self.feature_engineer = feature_engineer
        self.data_loader = data_loader
        self.model = None
        self.results = {}
        self.best_model_state = None
        self.model_config = None

    def _create_data_loader(self, features_dict, batch_size, shuffle=True):
        """Create DataLoader from dataset dictionary"""
        tensor_dataset = TensorDataset(
            features_dict['symptom_indices'], 
            features_dict['symptom_severities'], 
            features_dict['symptom_clusters'], 
            features_dict['clinical_priors'], 
            features_dict['animal_indices'], 
            features_dict['breed_indices'], 
            features_dict['age_indices'], 
            features_dict['weight_values'], 
            features_dict['symptom_counts'], 
            features_dict['risk_counts'], 
            features_dict['targets'] 
        )
        return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)

    def train_improved_sct(self, features_dict, epochs=25, learning_rate=1e-3, batch_size=32):
        """Train improved SCT model with enhanced training strategy"""
        print("[] Training IMPROVED Structured Clinical Transformer...")

        # Split data
        dataset_size = len(features_dict['symptom_indices'])
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        train_size = int(0.7 * dataset_size)
        val_size = int(0.15 * dataset_size)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create datasets
        train_features = {k: v[train_indices] for k, v in features_dict.items()}
        val_features = {k: v[val_indices] for k, v in features_dict.items()}
        test_features = {k: v[test_indices] for k, v in features_dict.items()}

        # Create data loaders
        train_loader = self._create_data_loader(train_features, batch_size, shuffle=True)
        val_loader = self._create_data_loader(val_features, batch_size, shuffle=False)
        test_loader = self._create_data_loader(test_features, batch_size, shuffle=False)

        # Initialize improved model
        from .models import ImprovedStructuredClinicalTransformer
        
        self.model = ImprovedStructuredClinicalTransformer(
            num_symptoms=len(self.feature_engineer.symptom_to_idx),
            num_animals=len(self.data_loader.all_animals),
            num_breeds=len(self.data_loader.le_breed.classes_),
            num_ages=len(self.data_loader.le_age.classes_),
            num_clusters=len(self.feature_engineer.cluster_to_idx),
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=0.2
        ) 
        self.model_config = self.model.config

        print(f"[] IMPROVED SCT initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")

        # Enhanced training setup
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999))
        criterion = nn.BCELoss()

        # Enhanced learning rate scheduler with warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0
        )

        best_val_f1 = 0
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        print("\n[] Starting IMPROVED SCT Training...")
        print("Epoch | Train Loss | Val Loss | Val Acc | Val F1 | Val AUC | Val Prec | Val Rec | LR")
        print("-" * 90)

        # Enhanced training loop with progress bars
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0

            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
            for batch in train_pbar:
                optimizer.zero_grad()

                # Unpack batch
                (symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                 animal_indices, breed_indices, age_indices, weight_values,
                 symptom_counts, risk_counts, targets) = batch

                outputs = self.model(
                    symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                    animal_indices, breed_indices, age_indices, weight_values,
                    symptom_counts, risk_counts
                )
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

                # Update progress bar
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Validation phase
            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []
            val_probabilities = []

            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
            with torch.no_grad():
                for batch in val_pbar:
                    # Unpack batch
                    (symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                     animal_indices, breed_indices, age_indices, weight_values,
                     symptom_counts, risk_counts, targets) = batch

                    outputs = self.model(
                        symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                        animal_indices, breed_indices, age_indices, weight_values,
                        symptom_counts, risk_counts
                    )
                    loss = criterion(outputs.squeeze(), targets)
                    val_loss += loss.item()

                    probabilities = outputs.squeeze().cpu().numpy()
                    val_probabilities.extend(probabilities)
                    val_predictions.extend([1 if p > 0.5 else 0 for p in probabilities])
                    val_targets.extend(targets.cpu().numpy())

                    # Update progress bar
                    val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy = accuracy_score(val_targets, val_predictions)
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                val_targets, val_predictions, average='binary', zero_division=0)
            val_auc = roc_auc_score(val_targets, val_probabilities)
            current_lr = scheduler.get_last_lr()[0]

            # Print epoch results
            print(f"{epoch+1:5d} | {train_loss:10.4f} | {val_loss:8.4f} | {val_accuracy:7.4f} | "
                  f"{val_f1:6.4f} | {val_auc:7.4f} | {val_precision:8.4f} | {val_recall:7.4f} | {current_lr:.1e}")

            # Enhanced early stopping based on both F1 and loss
            if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_loss < best_val_loss):
                best_val_f1 = val_f1
                best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f" [ ] New best model saved! (Val F1: {val_f1:.4f}, Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f" [ ] Early stopping at epoch {epoch+1}")
                    break

        # Load best model and evaluate on test set
        print("\n[] Loading best model for final evaluation...")
        self.model.load_state_dict(self.best_model_state)

        print("[] Evaluating on test set...")
        test_metrics = self.evaluate_model(test_loader, criterion)

        self.results['ImprovedSCT'] = test_metrics
        print(f"\n[] IMPROVED SCT Training Complete!")
        print(f" Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f" Test F1-Score: {test_metrics['f1_score']:.4f}")
        print(f" Test AUC: {test_metrics['auc_score']:.4f}")
        print(f" Test Precision: {test_metrics['precision']:.4f}")
        print(f" Test Recall: {test_metrics['recall']:.4f}")

        return self.results

    def evaluate_model(self, data_loader, criterion=None):
        """Evaluate model performance"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0

        eval_pbar = tqdm(data_loader, desc='Evaluating', leave=False)
        with torch.no_grad():
            for batch in eval_pbar:
                # Unpack batch
                (symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                 animal_indices, breed_indices, age_indices, weight_values,
                 symptom_counts, risk_counts, targets) = batch

                outputs = self.model(
                    symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                    animal_indices, breed_indices, age_indices, weight_values,
                    symptom_counts, risk_counts
                )

                if criterion:
                    loss = criterion(outputs.squeeze(), targets)
                    total_loss += loss.item()

                probabilities = outputs.squeeze().cpu().numpy()
                predictions = [1 if p > 0.5 else 0 for p in probabilities]
                all_probabilities.extend(probabilities)
                all_predictions.extend(predictions)
                all_targets.extend(targets.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='binary', zero_division=0
        )
        auc = roc_auc_score(all_targets, all_probabilities)
        avg_loss = total_loss / len(data_loader) if criterion else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'loss': avg_loss
        }

class LSTMTrainer:
    def __init__(self, feature_engineer, data_loader):
        self.feature_engineer = feature_engineer
        self.data_loader = data_loader
        self.model = None
        self.results = {}
        self.best_model_state = None

    def _create_data_loader(self, features_dict, batch_size, shuffle=True):
        """Create DataLoader from dataset dictionary"""
        tensor_dataset = TensorDataset(
            features_dict['symptom_indices'], 
            features_dict['symptom_severities'], 
            features_dict['symptom_clusters'], 
            features_dict['clinical_priors'], 
            features_dict['animal_indices'], 
            features_dict['breed_indices'], 
            features_dict['age_indices'], 
            features_dict['weight_values'], 
            features_dict['symptom_counts'], 
            features_dict['risk_counts'], 
            features_dict['targets']
        )
        return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)

    def train_lstm(self, features_dict, epochs=15, learning_rate=1e-3, batch_size=32):
        """Train LSTM model"""
        print("[] Training LSTM Model...")

        # Split data (same split as SCT for fair comparison)
        dataset_size = len(features_dict['symptom_indices'])
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        train_size = int(0.7 * dataset_size)
        val_size = int(0.15 * dataset_size)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create datasets
        train_features = {k: v[train_indices] for k, v in features_dict.items()}
        val_features = {k: v[val_indices] for k, v in features_dict.items()}
        test_features = {k: v[test_indices] for k, v in features_dict.items()}

        # Create data loaders
        train_loader = self._create_data_loader(train_features, batch_size, shuffle=True)
        val_loader = self._create_data_loader(val_features, batch_size, shuffle=False)
        test_loader = self._create_data_loader(test_features, batch_size, shuffle=False)

        # Initialize model
        from .models import VeterinaryLSTM
        
        self.model = VeterinaryLSTM(
            num_symptoms=len(self.feature_engineer.symptom_to_idx),
            num_animals=len(self.data_loader.all_animals),
            num_breeds=len(self.data_loader.le_breed.classes_),
            num_ages=len(self.data_loader.le_age.classes_)
        )

        print(f"[] LSTM initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")

        # Training setup
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        best_val_loss = float('inf')
        best_val_f1 = 0
        patience = 8
        patience_counter = 0

        print("\n[] Starting LSTM Training...")
        print("Epoch | Train Loss | Val Loss | Val Acc | Val F1 | Val AUC | Val Prec | Val Rec")
        print("-" * 80)

        # Training loop with progress bars
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0

            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
            for batch in train_pbar:
                optimizer.zero_grad()

                # Unpack batch
                (symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                 animal_indices, breed_indices, age_indices, weight_values,
                 symptom_counts, risk_counts, targets) = batch

                outputs = self.model(
                    symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                    animal_indices, breed_indices, age_indices, weight_values,
                    symptom_counts, risk_counts
                )
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Update progress bar
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Validation phase
            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []
            val_probabilities = []

            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
            with torch.no_grad():
                for batch in val_pbar:
                    # Unpack batch
                    (symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                     animal_indices, breed_indices, age_indices, weight_values,
                     symptom_counts, risk_counts, targets) = batch

                    outputs = self.model(
                        symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                        animal_indices, breed_indices, age_indices, weight_values,
                        symptom_counts, risk_counts
                    )
                    loss = criterion(outputs.squeeze(), targets)
                    val_loss += loss.item()
                    probabilities = outputs.squeeze().cpu().numpy()
                    val_probabilities.extend(probabilities)
                    val_predictions.extend([1 if p > 0.5 else 0 for p in probabilities])
                    val_targets.extend(targets.cpu().numpy())
                    # Update progress bar
                    val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy = accuracy_score(val_targets, val_predictions)
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                val_targets, val_predictions, average='binary', zero_division=0)
            val_auc = roc_auc_score(val_targets, val_probabilities)

            # Print epoch results
            print(f"{epoch+1:5d} | {train_loss:10.4f} | {val_loss:8.4f} | {val_accuracy:7.4f} | "
                  f"{val_f1:6.4f} | {val_auc:7.4f} | {val_precision:8.4f} | {val_recall:7.4f}")

            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f" [ ] New best model saved! (Val loss: {val_loss:.4f}, Val F1: {val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f" [ ] Early stopping at epoch {epoch+1}")
                    break

            scheduler.step(val_loss)

        # Load best model and evaluate on test set
        print("\n[] Loading best model for final evaluation...")
        self.model.load_state_dict(self.best_model_state)

        print("[] Evaluating on test set...")
        test_metrics = self.evaluate_model(test_loader, criterion)

        self.results['LSTM'] = test_metrics
        print(f"\n[] LSTM Training Complete!")
        print(f" Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f" Test F1-Score: {test_metrics['f1_score']:.4f}")
        print(f" Test AUC: {test_metrics['auc_score']:.4f}")
        print(f" Test Precision: {test_metrics['precision']:.4f}")
        print(f" Test Recall: {test_metrics['recall']:.4f}")

        return self.results

    def evaluate_model(self, data_loader, criterion=None):
        """Evaluate model performance"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0

        eval_pbar = tqdm(data_loader, desc='Evaluating', leave=False)
        with torch.no_grad():
            for batch in eval_pbar:
                # Unpack batch
                (symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                 animal_indices, breed_indices, age_indices, weight_values,
                 symptom_counts, risk_counts, targets) = batch

                outputs = self.model(
                    symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                    animal_indices, breed_indices, age_indices, weight_values,
                    symptom_counts, risk_counts
                )

                if criterion:
                    loss = criterion(outputs.squeeze(), targets)
                    total_loss += loss.item()

                probabilities = outputs.squeeze().cpu().numpy()
                predictions = [1 if p > 0.5 else 0 for p in probabilities]
                all_probabilities.extend(probabilities)
                all_predictions.extend(predictions)
                all_targets.extend(targets.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='binary', zero_division=0
        )
        auc = roc_auc_score(all_targets, all_probabilities)
        avg_loss = total_loss / len(data_loader) if criterion else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'loss': avg_loss
        }