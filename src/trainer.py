# src/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score

class ImprovedSCTTrainer:
    def __init__(self, feature_engineer, data_loader):
        self.feature_engineer = feature_engineer
        self.data_loader = data_loader
        self.model = None
        self.results = {}
        self.best_model_state = None
        self.model_config = None
        self.best_threshold = 0.5

    def _create_data_loader(self, features_dict, batch_size, shuffle=True):
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

    def train_improved_sct(self, features_dict, epochs=60, learning_rate=5e-4, batch_size=128):
        print("[] Training IMPROVED Structured Clinical Transformer...")

        dataset_size = len(features_dict['symptom_indices'])
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        train_size = int(0.7 * dataset_size)
        val_size = int(0.15 * dataset_size)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_features = {k: v[train_indices] for k, v in features_dict.items()}
        val_features = {k: v[val_indices] for k, v in features_dict.items()}
        test_features = {k: v[test_indices] for k, v in features_dict.items()}

        train_loader = self._create_data_loader(train_features, batch_size, shuffle=True)
        val_loader = self._create_data_loader(val_features, batch_size, shuffle=False)
        test_loader = self._create_data_loader(test_features, batch_size, shuffle=False)

        from .models import ImprovedStructuredClinicalTransformer

        self.model = ImprovedStructuredClinicalTransformer(
            num_symptoms=len(self.feature_engineer.symptom_to_idx),
            num_animals=len(self.data_loader.all_animals),
            num_breeds=len(self.data_loader.le_breed.classes_),
            num_ages=len(self.data_loader.le_age.classes_),
            num_clusters=len(self.feature_engineer.cluster_to_idx),
            d_model=384,
            nhead=8,
            num_layers=4,
            dropout=0.2
        )
        self.model_config = self.model.config
        print(f"[] IMPROVED SCT initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")

        # pos_weight = 2.0 to favour recall
        pos_weight = torch.tensor([2.0], dtype=torch.float32)
        print(f"[] pos_weight = {pos_weight.item():.2f}")

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.02)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1.5e-3,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.2,
            div_factor=10.0,
            final_div_factor=100.0
        )

        best_val_f1 = 0
        patience = 15
        patience_counter = 0

        print("\n[] Starting IMPROVED SCT Training...")
        print("Epoch | Train Loss | Val Loss | Val Acc | Val F1 | Val AUC | Val Prec | Val Rec | LR")
        print("-" * 90)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
            for batch in train_pbar:
                optimizer.zero_grad()
                (symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                 animal_indices, breed_indices, age_indices, weight_values,
                 symptom_counts, risk_counts, targets) = batch

                outputs = self.model(
                    symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                    animal_indices, breed_indices, age_indices, weight_values,
                    symptom_counts, risk_counts
                )
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            self.model.eval()
            val_loss = 0
            val_probabilities = []
            val_targets = []
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
            with torch.no_grad():
                for batch in val_pbar:
                    (symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                     animal_indices, breed_indices, age_indices, weight_values,
                     symptom_counts, risk_counts, targets) = batch

                    outputs = self.model(
                        symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                        animal_indices, breed_indices, age_indices, weight_values,
                        symptom_counts, risk_counts
                    )
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    val_probabilities.extend(probs)
                    val_targets.extend(targets.cpu().numpy())
                    val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            val_loss /= len(val_loader)
            val_probabilities = np.array(val_probabilities)
            val_targets = np.array(val_targets)

            # Find best threshold on validation set
            best_thr = 0.5
            best_f1 = 0.0
            for thr in np.arange(0.3, 0.7, 0.01):
                preds = (val_probabilities >= thr).astype(int)
                f1 = f1_score(val_targets, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr

            val_predictions = (val_probabilities >= best_thr).astype(int)
            val_accuracy = accuracy_score(val_targets, val_predictions)
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                val_targets, val_predictions, average='binary', zero_division=0)
            val_auc = roc_auc_score(val_targets, val_probabilities)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"{epoch+1:5d} | {train_loss/len(train_loader):10.4f} | {val_loss:8.4f} | {val_accuracy:7.4f} | "
                  f"{val_f1:6.4f} | {val_auc:7.4f} | {val_precision:8.4f} | {val_recall:7.4f} | {current_lr:.1e}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                self.best_threshold = best_thr
                patience_counter = 0
                print(f" [ ] New best model saved! (Val F1: {val_f1:.4f}, best threshold: {best_thr:.2f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f" [ ] Early stopping at epoch {epoch+1}")
                    break

        print("\n[] Loading best model for final evaluation...")
        self.model.load_state_dict(self.best_model_state)
        print(f"[] Using threshold = {self.best_threshold:.2f} for predictions")
        print("[] Evaluating on test set...")
        test_metrics = self.evaluate_model(test_loader, criterion, threshold=self.best_threshold)

        self.results['ImprovedSCT'] = test_metrics
        print(f"\n[] IMPROVED SCT Training Complete!")
        print(f" Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f" Test F1-Score: {test_metrics['f1_score']:.4f}")
        print(f" Test AUC: {test_metrics['auc_score']:.4f}")
        print(f" Test Precision: {test_metrics['precision']:.4f}")
        print(f" Test Recall: {test_metrics['recall']:.4f}")
        return self.results

    def evaluate_model(self, data_loader, criterion=None, threshold=0.5):
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0
        eval_pbar = tqdm(data_loader, desc='Evaluating', leave=False)
        with torch.no_grad():
            for batch in eval_pbar:
                (symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                 animal_indices, breed_indices, age_indices, weight_values,
                 symptom_counts, risk_counts, targets) = batch

                outputs = self.model(
                    symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                    animal_indices, breed_indices, age_indices, weight_values,
                    symptom_counts, risk_counts
                )
                if criterion:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs >= threshold).astype(int)
                all_probabilities.extend(probs)
                all_predictions.extend(preds)
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

# LSTMTrainer (similarly speed up)
class LSTMTrainer:
    def __init__(self, feature_engineer, data_loader):
        self.feature_engineer = feature_engineer
        self.data_loader = data_loader
        self.model = None
        self.results = {}
        self.best_model_state = None

    def _create_data_loader(self, features_dict, batch_size, shuffle=True):
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

    def train_lstm(self, features_dict, epochs=40, learning_rate=1e-3, batch_size=128):
        print("[] Training LSTM Model...")
        dataset_size = len(features_dict['symptom_indices'])
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        train_size = int(0.7 * dataset_size)
        val_size = int(0.15 * dataset_size)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_features = {k: v[train_indices] for k, v in features_dict.items()}
        val_features = {k: v[val_indices] for k, v in features_dict.items()}
        test_features = {k: v[test_indices] for k, v in features_dict.items()}

        train_loader = self._create_data_loader(train_features, batch_size, shuffle=True)
        val_loader = self._create_data_loader(val_features, batch_size, shuffle=False)
        test_loader = self._create_data_loader(test_features, batch_size, shuffle=False)

        from .models import VeterinaryLSTM

        self.model = VeterinaryLSTM(
            num_symptoms=len(self.feature_engineer.symptom_to_idx),
            num_animals=len(self.data_loader.all_animals),
            num_breeds=len(self.data_loader.le_breed.classes_),
            num_ages=len(self.data_loader.le_age.classes_),
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        )

        print(f"[] LSTM initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")

        # Use similar pos_weight
        targets = train_features['targets']
        pos_count = targets.sum().item()
        neg_count = len(targets) - pos_count
        pos_weight = torch.tensor([max(neg_count / pos_count, 1.5)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2e-3,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.2,
            div_factor=10.0,
            final_div_factor=100.0
        )

        best_val_f1 = 0
        patience = 10
        patience_counter = 0

        print("\n[] Starting LSTM Training...")
        print("Epoch | Train Loss | Val Loss | Val Acc | Val F1 | Val AUC | Val Prec | Val Rec | LR")
        print("-" * 90)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
            for batch in train_pbar:
                optimizer.zero_grad()
                (symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                 animal_indices, breed_indices, age_indices, weight_values,
                 symptom_counts, risk_counts, targets) = batch

                outputs = self.model(
                    symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                    animal_indices, breed_indices, age_indices, weight_values,
                    symptom_counts, risk_counts
                )
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []
            val_probabilities = []
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
            with torch.no_grad():
                for batch in val_pbar:
                    (symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                     animal_indices, breed_indices, age_indices, weight_values,
                     symptom_counts, risk_counts, targets) = batch

                    outputs = self.model(
                        symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                        animal_indices, breed_indices, age_indices, weight_values,
                        symptom_counts, risk_counts
                    )
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    probabilities = torch.sigmoid(outputs).cpu().numpy()
                    val_probabilities.extend(probabilities)
                    val_predictions.extend([1 if p > 0.5 else 0 for p in probabilities])
                    val_targets.extend(targets.cpu().numpy())
                    val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy = accuracy_score(val_targets, val_predictions)
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                val_targets, val_predictions, average='binary', zero_division=0)
            val_auc = roc_auc_score(val_targets, val_probabilities)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"{epoch+1:5d} | {train_loss:10.4f} | {val_loss:8.4f} | {val_accuracy:7.4f} | "
                  f"{val_f1:6.4f} | {val_auc:7.4f} | {val_precision:8.4f} | {val_recall:7.4f} | {current_lr:.1e}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f" [ ] New best model saved! (Val F1: {val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f" [ ] Early stopping at epoch {epoch+1}")
                    break

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
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0
        eval_pbar = tqdm(data_loader, desc='Evaluating', leave=False)
        with torch.no_grad():
            for batch in eval_pbar:
                (symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                 animal_indices, breed_indices, age_indices, weight_values,
                 symptom_counts, risk_counts, targets) = batch

                outputs = self.model(
                    symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                    animal_indices, breed_indices, age_indices, weight_values,
                    symptom_counts, risk_counts
                )
                if criterion:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                probabilities = torch.sigmoid(outputs).cpu().numpy()
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