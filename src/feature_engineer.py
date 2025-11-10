import numpy as np
import torch
from tqdm.auto import tqdm

class VeterinaryFeatureEngineer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.symptom_to_idx = None
        self.cluster_to_idx = None

    def prepare_traditional_features(self, processed_df):
        """Prepare features for traditional ML models"""
        print("[] Preparing traditional features...")
        features = []
        targets = []
        
        symptom_columns = [col for col in processed_df.columns if col.startswith("Symptom_")]
        
        for _, row in tqdm(processed_df.iterrows(), total=len(processed_df), desc="Processing features"):
            animal_features = [1 if row['AnimalName'] == animal else 0 for animal in self.data_loader.all_animals]
            symptom_features = [1 if symptom in [row[col] for col in symptom_columns] else 0 for symptom in self.data_loader.all_symptoms]
            
            symptom_count = row['Symptom_Count']
            has_emergency = 1 if any(
                symptom in ['seizures', 'unconsciousness', 'bleeding', 'paralysis']
                for symptom in [row[col] for col in symptom_columns]
            ) else 0

            severity_score = 0
            for col in symptom_columns:
                symptom = row[col]
                if symptom != 'none':
                    severity_score += self.data_loader.symptom_severity_weights.get(symptom, 0.1)
            severity_score = min(severity_score / 3.0, 1.0)

            demo_features = [row['Animal_encoded'], row['Breed_encoded'], row['Age_encoded'], row['Weight_normalized']]
            risk_features = [row['High_Risk_Count'], row['Medium_Risk_Count']]

            combined_features = animal_features + symptom_features + [severity_score, symptom_count, has_emergency] + demo_features + risk_features

            features.append(combined_features)
            targets.append(row['target'])
        
        return np.array(features), np.array(targets)

    def prepare_sct_features(self, processed_df):
        """Prepare features for Structured Clinical Transformer"""
        print("[] Preparing SCT features...")
        symptom_columns = [col for col in processed_df.columns if col.startswith('Symptom_')]

        if self.symptom_to_idx is None:
            self.symptom_to_idx = {symptom: idx + 1 for idx, symptom in enumerate(self.data_loader.all_symptoms)}
            self.symptom_to_idx['none'] = 0

        # Create Cluster mappings
        self.cluster_to_idx = {}
        idx = 1
        for cluster_name, symptoms in self.data_loader.symptom_clusters.items():
            self.cluster_to_idx[cluster_name] = idx
            idx += 1

        max_symptoms = 10
        symptom_indices = []
        symptom_severities = []
        symptom_clusters = []
        clinical_priors = []

        for _, row in tqdm(processed_df.iterrows(), total=len(processed_df), desc="Processing SCT features"):
            symptom_seq = []
            severity_seq = []
            cluster_seq = []
            prior_seq = []
            
            for col in symptom_columns:
                symptom = row[col]
                if symptom != 'none' and symptom in self.symptom_to_idx:
                    symptom_seq.append(self.symptom_to_idx[symptom])
                    severity_val = self.data_loader.symptom_severity_weights.get(symptom, 0.1)
                    severity_seq.append(severity_val)

                    # Find which cluster this symptom belongs to
                    cluster_id = 0
                    for cluster_name, symptoms in self.data_loader.symptom_clusters.items():
                        if symptom in symptoms:
                            cluster_id = self.cluster_to_idx[cluster_name]
                            break
                    cluster_seq.append(cluster_id)

                    # Clinical prior: combine severity and cluster information
                    prior_value = severity_val * (cluster_id * 0.1 + 0.5)
                    prior_seq.append(prior_value)
                else:
                    # For padding tokens
                    symptom_seq.append(0)
                    severity_seq.append(0.0)
                    cluster_seq.append(0)
                    prior_seq.append(0.0)

            # Ensure we have exactly max_symptoms
            symptom_seq = symptom_seq[:max_symptoms]
            severity_seq = severity_seq[:max_symptoms]
            cluster_seq = cluster_seq[:max_symptoms]
            prior_seq = prior_seq[:max_symptoms]

            # Pad if necessary
            if len(symptom_seq) < max_symptoms:
                symptom_seq.extend([0] * (max_symptoms - len(symptom_seq)))
                severity_seq.extend([0.0] * (max_symptoms - len(severity_seq)))
                cluster_seq.extend([0] * (max_symptoms - len(cluster_seq)))
                prior_seq.extend([0.0] * (max_symptoms - len(prior_seq)))

            symptom_indices.append(symptom_seq)
            symptom_severities.append(severity_seq)
            symptom_clusters.append(cluster_seq)
            clinical_priors.append(prior_seq)

        # Convert to tensors with proper dtype handling
        features_dict = {
            'symptom_indices': torch.tensor(symptom_indices, dtype=torch.long),
            'symptom_severities': torch.tensor(symptom_severities, dtype=torch.float32),
            'symptom_clusters': torch.tensor(symptom_clusters, dtype=torch.long),
            'clinical_priors': torch.tensor(clinical_priors, dtype=torch.float32),
            'animal_indices': torch.tensor(processed_df['Animal_encoded'].values, dtype=torch.long),
            'breed_indices': torch.tensor(processed_df['Breed_encoded'].values, dtype=torch.long),
            'age_indices': torch.tensor(processed_df['Age_encoded'].values, dtype=torch.long),
            'weight_values': torch.tensor(processed_df['Weight_normalized'].values, dtype=torch.float32),
            'symptom_counts': torch.tensor(processed_df['Symptom_Count'].values, dtype=torch.float32),
            'risk_counts': torch.tensor(processed_df[['High_Risk_Count', 'Medium_Risk_Count']].values.astype(np.float32), dtype=torch.float32),
            'targets': torch.tensor(processed_df['target'].values, dtype=torch.float32)
        }

        print(f"[] Prepared SCT features: {len(symptom_indices)} samples")
        return features_dict