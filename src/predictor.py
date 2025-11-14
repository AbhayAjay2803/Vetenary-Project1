import torch
import numpy as np
import joblib
from .models import ImprovedStructuredClinicalTransformer, VeterinaryLSTM

class VeterinaryPredictor:
    def __init__(self):
        self.models = {}
        self.feature_engineer = None
        self.data_loader = None
        self.loaded = False

    def load_models(self, model_paths):
        """Load all trained models and encoders"""
        print("[] Loading veterinary models...")
        try:
            # Load encoders
            encoders_data = joblib.load(model_paths['encoders'])

            # Recreate feature engineer and data loader
            from .data_loader import VeterinaryDatasetLoader
            from .feature_engineer import VeterinaryFeatureEngineer
            
            self.feature_engineer = VeterinaryFeatureEngineer(VeterinaryDatasetLoader())
            self.feature_engineer.symptom_to_idx = encoders_data['symptom_to_idx']
            self.feature_engineer.cluster_to_idx = encoders_data['cluster_to_idx']
            
            self.data_loader = VeterinaryDatasetLoader()
            self.data_loader.le_animal = encoders_data['le_animal']
            self.data_loader.le_breed = encoders_data['le_breed']
            self.data_loader.le_age = encoders_data['le_age']
            self.data_loader.symptom_severity_weights = encoders_data['symptom_severity_weights']
            self.data_loader.symptom_clusters = encoders_data['symptom_clusters']
            self.data_loader.all_animals = encoders_data['all_animals']
            self.data_loader.all_symptoms = encoders_data['all_symptoms']

            # Load traditional models
            for model_name, model_path in model_paths.items():
                if model_name in ['RandomForest', 'NeuralNetwork', 'XGBoost']:
                    self.models[model_name] = joblib.load(model_path)
                    print(f" Loaded {model_name}")

            # Load SCT model with proper configuration
            if 'SCT' in model_paths:
                sct_data = torch.load(model_paths['SCT'], map_location='cpu')
                self.models['SCT'] = ImprovedStructuredClinicalTransformer(**sct_data['model_config'])
                self.models['SCT'].load_state_dict(sct_data['model_state_dict'])
                self.models['SCT'].eval()
                print(f" Loaded SCT (d_model={sct_data['model_config']['d_model']}, layers={sct_data['model_config']['num_layers']})")

            # Load LSTM model with proper configuration
            if 'LSTM' in model_paths:
                lstm_data = torch.load(model_paths['LSTM'], map_location='cpu')
                self.models['LSTM'] = VeterinaryLSTM(**lstm_data['model_config'])
                self.models['LSTM'].load_state_dict(lstm_data['model_state_dict'])
                self.models['LSTM'].eval()
                print(f" Loaded LSTM")
            
            self.loaded = True
            print(f"[] All models loaded successfully!")
            print(f" Available models: {list(self.models.keys())}")
            print(f" Animals: {len(self.data_loader.all_animals)}")
            print(f" Symptoms: {len(self.data_loader.all_symptoms)}")

            return True

        except Exception as e:
            print(f"[] Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _check_symptom_supremacy(self, symptoms):
        """Check if any symptom triggers the supremacy rule (HIGH RISK symptoms)"""
        high_risk_symptoms = []
        for symptom in symptoms:
            severity = self.data_loader.symptom_severity_weights.get(symptom, 0.1)
            if severity > 0.7:  # HIGH RISK threshold
                high_risk_symptoms.append(symptom)
        return high_risk_symptoms

    def _check_model_supremacy(self, predictions):
        """Check if model votes trigger the supremacy rule (majority dangerous)"""
        dangerous_votes = 0
        total_models = len(predictions)
        
        for model_name, prediction in predictions.items():
            if prediction.get('dangerous', False):
                dangerous_votes += 1
        
        return dangerous_votes, total_models, dangerous_votes >= 3  # Majority rule (3/5 or more)

    def predict_ensemble(self, animal, breed, age, weight, symptoms, model_names=None):
        """Make ensemble prediction with FAIL-SAFE SUPREMACY RULES"""
        if not self.loaded:
            return "Models not loaded. Please call load_models() first."

        if model_names is None:
            model_names = list(self.models.keys())
        
        try:
            predictions = {}
            for model_name in model_names:
                if model_name in self.models:
                    if model_name in ['RandomForest', 'NeuralNetwork', 'XGBoost']:
                        # Traditional model prediction
                        prediction = self._predict_traditional(model_name, animal, breed, age, weight, symptoms)
                    else:
                        # Deep learning model prediction
                        prediction = self._predict_dl(model_name, animal, breed, age, weight, symptoms)
                    predictions[model_name] = prediction

            # Calculate ensemble prediction (weighted average probability)
            probabilities = []
            weights = []

            # Assign weights based on model performance (SCT gets highest weight)
            for model_name, pred in predictions.items():
                probabilities.append(pred['probability'])
                if model_name == 'SCT':
                    weights.append(0.4)  # Highest weight for SCT
                elif model_name == 'LSTM':
                    weights.append(0.25)
                elif model_name == 'XGBoost':
                    weights.append(0.2)
                else:
                    weights.append(0.15)

            # Normalize weights
            weights = np.array(weights) / sum(weights)

            weighted_probability = np.average(probabilities, weights=weights)
            
            # ===== FAIL-SAFE SUPREMACY RULES =====
            supremacy_triggered = False
            supremacy_reason = ""
            
            # Rule 1: Symptom Supremacy - Any HIGH RISK symptom
            high_risk_symptoms = self._check_symptom_supremacy(symptoms)
            if high_risk_symptoms:
                supremacy_triggered = True
                supremacy_reason = f"HIGH-RISK SYMPTOMS: {', '.join(high_risk_symptoms)}"
                weighted_probability = max(weighted_probability, 0.8)  # Force high probability
                print(f"🚨 SYMPTOM SUPREMACY TRIGGERED: {supremacy_reason}")

            # Rule 2: Model Supremacy - Majority vote dangerous
            dangerous_votes, total_models, model_supremacy = self._check_model_supremacy(predictions)
            if model_supremacy:
                supremacy_triggered = True
                supremacy_reason = f"MODEL SUPREMACY: {dangerous_votes}/{total_models} models voted DANGEROUS"
                weighted_probability = max(weighted_probability, 0.7)  # Force high probability
                print(f"🚨 MODEL SUPREMACY TRIGGERED: {supremacy_reason}")

            # Apply supremacy rules to final decision
            if supremacy_triggered:
                ensemble_dangerous = True
                # Boost confidence when supremacy is triggered
                ensemble_confidence = min(weighted_probability + 0.2, 0.95)
            else:
                ensemble_dangerous = weighted_probability > 0.5
                ensemble_confidence = weighted_probability if ensemble_dangerous else 1 - weighted_probability

            # Count model agreement
            agreement_count = sum(1 for p in probabilities if (p > 0.5) == ensemble_dangerous)

            return {
                'ensemble': {
                    'dangerous': ensemble_dangerous,
                    'probability': weighted_probability,
                    'confidence': f"{ensemble_confidence:.1%}",
                    'model_agreement': f"{agreement_count}/{len(probabilities)}",
                    'weighted_combination': True,
                    'supremacy_triggered': supremacy_triggered,
                    'supremacy_reason': supremacy_reason,
                    'high_risk_symptoms': high_risk_symptoms,
                    'dangerous_votes': dangerous_votes,
                    'total_models': total_models,
                    'symptom_severity_weights': self.data_loader.symptom_severity_weights  # Pass severity data for report
                },
                'individual_predictions': predictions
            }
        except Exception as e:
            return f"Prediction error: {str(e)}"

    def _predict_traditional(self, model_name, animal, breed, age, weight, symptoms):
        """Predict using traditional ML models"""
        try:
            # Prepare features for traditional models
            animal_features = [1 if animal == a else 0 for a in self.data_loader.all_animals]
            symptom_features = [1 if s in symptoms else 0 for s in self.data_loader.all_symptoms]
            symptom_count = len(symptoms)
            has_emergency = 1 if any(s in ['seizures', 'unconsciousness', 'bleeding', 'paralysis'] for s in symptoms) else 0

            severity_score = sum(self.data_loader.symptom_severity_weights.get(s, 0.1) for s in symptoms)
            severity_score = min(severity_score / 3.0, 1.0)

            # Encode categorical features
            animal_encoded = self.data_loader.le_animal.transform([animal])[0]
            breed_encoded = self.data_loader.le_breed.transform([breed])[0]
            age_encoded = self.data_loader.le_age.transform([age])[0]
            weight_normalized = (weight - 250.5) / 144.3  # Approximate normalization

            # Risk counts
            high_risk_count = sum(1 for s in symptoms if self.data_loader.symptom_severity_weights.get(s, 0) > 0.7)
            medium_risk_count = sum(1 for s in symptoms if 0.4 < self.data_loader.symptom_severity_weights.get(s, 0) <= 0.7)

            demo_features = [animal_encoded, breed_encoded, age_encoded, weight_normalized]
            risk_features = [high_risk_count, medium_risk_count]
            features = animal_features + symptom_features + [severity_score, symptom_count, has_emergency] + demo_features + risk_features

            features = np.array(features).reshape(1, -1)

            # Make prediction
            model = self.models[model_name]
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features)[0, 1]
            else:
                probability = model.predict(features)[0]

            is_dangerous = probability > 0.5
            confidence = probability if is_dangerous else 1 - probability

            return {
                'dangerous': is_dangerous,
                'probability': probability,
                'confidence': f"{confidence:.1%}",
                'model': model_name
            }
        except Exception as e:
            return {
                'dangerous': False,
                'probability': 0.0,
                'confidence': "0.0%",
                'model': model_name,
                'error': str(e)
            }

    def _predict_dl(self, model_name, animal, breed, age, weight, symptoms):
        """Predict using deep learning models"""
        try:
            # Prepare features (similar to SCT training)
            animal_encoded = self.data_loader.le_animal.transform([animal])[0]
            breed_encoded = self.data_loader.le_breed.transform([breed])[0]
            age_encoded = self.data_loader.le_age.transform([age])[0]
            weight_normalized = (weight - 250.5) / 144.3

            # Prepare symptom features
            symptom_indices = []
            symptom_severities = []
            symptom_clusters = []
            clinical_priors = []

            max_symptoms = 10
            for symptom in symptoms:
                if symptom in self.feature_engineer.symptom_to_idx:
                    symptom_indices.append(self.feature_engineer.symptom_to_idx[symptom])
                    severity = self.data_loader.symptom_severity_weights.get(symptom, 0.1)
                    symptom_severities.append(severity)

                    # Find Cluster
                    cluster_id = 0
                    for cluster_name, cluster_symptoms in self.data_loader.symptom_clusters.items():
                        if symptom in cluster_symptoms:
                            cluster_id = self.feature_engineer.cluster_to_idx[cluster_name]
                            break
                    symptom_clusters.append(cluster_id)

                    # Clinical prior
                    prior_value = severity * (cluster_id * 0.1 + 0.5)
                    clinical_priors.append(prior_value)

            # Pad sequences
            while len(symptom_indices) < max_symptoms:
                symptom_indices.append(0)
                symptom_severities.append(0.0)
                symptom_clusters.append(0)
                clinical_priors.append(0.0)

            symptom_indices = symptom_indices[:max_symptoms]
            symptom_severities = symptom_severities[:max_symptoms]
            symptom_clusters = symptom_clusters[:max_symptoms]
            clinical_priors = clinical_priors[:max_symptoms]

            # Convert to tensors
            symptom_indices_t = torch.tensor([symptom_indices], dtype=torch.long)
            symptom_severities_t = torch.tensor([symptom_severities], dtype=torch.float32)
            symptom_clusters_t = torch.tensor([symptom_clusters], dtype=torch.long)
            clinical_priors_t = torch.tensor([clinical_priors], dtype=torch.float32)
            animal_indices_t = torch.tensor([animal_encoded], dtype=torch.long)
            breed_indices_t = torch.tensor([breed_encoded], dtype=torch.long)
            age_indices_t = torch.tensor([age_encoded], dtype=torch.long)
            weight_values_t = torch.tensor([weight_normalized], dtype=torch.float32)
            symptom_counts_t = torch.tensor([len(symptoms)], dtype=torch.float32)

            # Calculate risk counts
            high_risk_count = sum(1 for s in symptoms if self.data_loader.symptom_severity_weights.get(s, 0) > 0.7)
            medium_risk_count = sum(1 for s in symptoms if 0.4 < self.data_loader.symptom_severity_weights.get(s, 0) <= 0.7)
            risk_counts_t = torch.tensor([[high_risk_count, medium_risk_count]], dtype=torch.float32)

            # Make prediction
            model = self.models[model_name]
            with torch.no_grad():
                prediction = model(
                    symptom_indices_t, symptom_severities_t,
                    symptom_clusters_t, clinical_priors_t,
                    animal_indices_t, breed_indices_t, age_indices_t,
                    weight_values_t,
                    symptom_counts_t, risk_counts_t
                )
                probability = prediction.item()
                is_dangerous = probability > 0.5
                confidence = probability if is_dangerous else 1 - probability

                return {
                    'dangerous': is_dangerous,
                    'probability': probability,
                    'confidence': f"{confidence:.1%}",
                    'model': model_name
                }
        except Exception as e:
            return {
                'dangerous': False,
                'probability': 0.0,
                'confidence': "0.0%",
                'model': model_name,
                'error': str(e)
            }

    def get_available_animals(self):
        """Get list of available animal types"""
        return self.data_loader.all_animals if self.loaded else []

    def get_available_symptoms(self):
        """Get list of available symptoms"""
        return self.data_loader.all_symptoms if self.loaded else []

    def get_available_breeds(self, animal):
        """Get available breeds for a specific animal"""
        if not self.loaded:
            return []
        
        breed_mapping = {
            'dog': ['labrador', 'german_shepherd', 'golden_retriever', 'bulldog', 'poodle'],
            'cat': ['siamese', 'persian', 'maine_coon', 'bengal', 'ragdoll'],
            'cow': ['holstein', 'angus', 'hereford', 'jersey'],
            'horse': ['arabian', 'quarter_horse', 'thoroughbred']
        }
        return breed_mapping.get(animal, ['mixed'])