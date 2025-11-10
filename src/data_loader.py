import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

class VeterinaryDatasetLoader:
    def __init__(self):
        self.dataset = None
        self.processed_df = None
        self.all_symptoms = set()
        self.all_animals = set()
        self.le_animal = None
        self.le_breed = None
        self.le_age = None

        # Enhanced symptom severity weights with clinical priors
        self.symptom_severity_weights = {
            'seizures': 0.95, 'unconsciousness': 0.95, 'bleeding': 0.90, 'paralysis': 0.92,
            'rapid_breathing': 0.85, 'jaundice': 0.80, 'pale_gums': 0.75, 'abdominal_pain': 0.70,
            'fever': 0.65, 'vomiting': 0.60, 'diarrhea': 0.55, 'dehydration': 0.70,
            'weight_loss': 0.50, 'lethargy': 0.45, 'loss_of_appetite': 0.40, 'coughing': 0.35,
            'sneezing': 0.20, 'itching': 0.15, 'swelling': 0.45, 'nasal_discharge': 0.30,
            'eye_discharge': 0.30, 'lameness': 0.40, 'constipation': 0.25, 'excessive_thirst': 0.20,
            'urination_problems': 0.50, 'reproductive_issues': 0.45, 'skin_lesions': 0.35,
            'hair_loss': 0.20, 'tremors': 0.65, 'pain': 0.55, 'colic': 0.70, 'skin_rashes': 0.25,
            'respiratory_distress': 0.80, 'anorexia': 0.50, 'weakness': 0.40, 'depression': 0.30,
            'nausea': 0.35, 'chills': 0.40, 'headache': 0.25, 'muscle_pain': 0.35, 'joint_pain': 0.40,
            'bloating': 0.30, 'flatulence': 0.15, 'bad_breath': 0.10, 'discharge': 0.25,
            'inflammation': 0.35, 'redness': 0.20, 'wheezing': 0.45, 'gasping': 0.60,
            'convulsions': 0.90, 'twitching': 0.50, 'salivating': 0.25, 'ulcers': 0.40,
            'lesions': 0.35, 'abscesses': 0.45, 'infection': 0.55, 'malaise': 0.30,
            'fatigue': 0.25, 'emaciation': 0.60, 'pneumonia': 0.75, 'torticollis': 0.40,
            'dyspnea': 0.65, 'cyanosis': 0.70, 'edema': 0.45, 'ascites': 0.50, 'anemia': 0.55,
            'hemorrhage': 0.85, 'shock': 0.90, 'coma': 0.95, 'ataxia': 0.50, 'blindness': 0.60,
            'deafness': 0.30, 'paraplegia': 0.80, 'hyperesthesia': 0.40, 'aggression': 0.35,
            'disorientation': 0.45, 'staggering': 0.50, 'circling': 0.40, 'head_pressing': 0.55,
            'nystagmus': 0.45, 'opisthotonos': 0.70, 'paresis': 0.60
        }

        # Clinical prior: symptom clusters
        self.symptom_clusters = {
            'neurological': ['seizures', 'unconsciousness', 'paralysis', 'tremors', 'convulsions',
                           'disorientation', 'staggering', 'twitching', 'ataxia', 'hyperesthesia',
                           'circling', 'head_pressing', 'nystagmus', 'opisthotonos', 'paresis'],
            'respiratory': ['rapid_breathing', 'coughing', 'sneezing', 'nasal_discharge',
                          'respiratory_distress', 'wheezing', 'gasping', 'pneumonia', 'dyspnea'],
            'gastrointestinal': ['vomiting', 'diarrhea', 'constipation', 'bloating', 'flatulence',
                               'abdominal_pain', 'colic', 'loss_of_appetite', 'dehydration'],
            'systemic': ['fever', 'lethargy', 'weight_loss', 'weakness', 'jaundice', 'pale_gums',
                        'anemia', 'shock', 'coma', 'infection', 'malaise', 'fatigue'],
            'dermatological': ['itching', 'skin_lesions', 'hair_loss', 'skin_rashes', 'ulcers',
                              'lesions', 'abscesses', 'inflammation', 'redness']
        }

    def create_comprehensive_dataset(self, n_samples=5000):
        """Create comprehensive veterinary dataset"""
        animals = ['dog', 'cat', 'cow', 'horse', 'rabbit', 'goat', 'sheep', 'chicken', 'pig',
                  'parrot', 'hamster', 'guinea_pig', 'turtle', 'duck', 'turkey']
        
        breeds = {
            'dog': ['labrador', 'german_shepherd', 'golden_retriever', 'bulldog', 'poodle'],
            'cat': ['siamese', 'persian', 'maine_coon', 'bengal', 'ragdoll'],
            'cow': ['holstein', 'angus', 'hereford', 'jersey'],
            'horse': ['arabian', 'quarter_horse', 'thoroughbred'],
            'rabbit': ['dutch', 'flemish_giant', 'rex'],
            'goat': ['boer', 'nubian', 'alpine'],
            'sheep': ['merino', 'dorset', 'suffolk'],
            'chicken': ['rhode_island_red', 'leghorn', 'sussex'],
            'pig': ['yorkshire', 'duroc', 'hampshire'],
            'parrot': ['african_grey', 'macaw', 'cockatoo'],
            'hamster': ['syrian', 'dwarf', 'roborovski'],
            'guinea_pig': ['american', 'abby', 'peruvian'],
            'turtle': ['red_eared_slider', 'box_turtle', 'painted_turtle'],
            'duck': ['pekin', 'mallard', 'muscovy'],
            'turkey': ['broad_breasted_bronze', 'bourbon_red', 'narragansett']
        }
        
        symptoms_list = list(self.symptom_severity_weights.keys())
        
        # Enhanced animal-symptom mapping
        animal_symptoms = {
            'dog': ['fever', 'vomiting', 'lethargy', 'loss_of_appetite', 'coughing', 'diarrhea', 'lameness'],
            'cat': ['fever', 'vomiting', 'lethargy', 'sneezing', 'eye_discharge', 'loss_of_appetite', 'hair_loss'],
            'cow': ['weight_loss', 'dehydration', 'reproductive_issues', 'lameness', 'fever', 'bloating'],
            'horse': ['lameness', 'weight_loss', 'colic', 'nasal_discharge', 'fever', 'swelling'],
            'rabbit': ['lethargy', 'loss_of_appetite', 'hair_loss', 'skin_lesions', 'diarrhea', 'sneezing'],
            'goat': ['weight_loss', 'diarrhea', 'coughing', 'reproductive_issues', 'fever', 'bloating'],
            'sheep': ['weight_loss', 'lameness', 'coughing', 'reproductive_issues', 'fever'],
            'chicken': ['lethargy', 'loss_of_appetite', 'sneezing', 'nasal_discharge', 'diarrhea'],
            'pig': ['fever', 'coughing', 'loss_of_appetite', 'lameness', 'diarrhea', 'vomiting'],
            'parrot': ['lethargy', 'loss_of_appetite', 'feather_plucking', 'sneezing', 'nasal_discharge'],
            'hamster': ['lethargy', 'hair_loss', 'weight_loss', 'wet_tail', 'sneezing'],
            'guinea_pig': ['lethargy', 'weight_loss', 'hair_loss', 'dental_problems', 'sneezing'],
            'turtle': ['lethargy', 'loss_of_appetite', 'shell_rot', 'swollen_eyes', 'respiratory_distress'],
            'duck': ['lethargy', 'loss_of_appetite', 'lameness', 'respiratory_distress', 'diarrhea'],
            'turkey': ['lethargy', 'loss_of_appetite', 'respiratory_distress', 'diarrhea', 'swelling']
        }
        
        np.random.seed(42)
        data = []
        target_positive_rate = 0.6  # Aim for 60% positive cases

        print("[] Creating dataset...")
        for i in tqdm(range(n_samples), desc="Generating samples"):
            animal = np.random.choice(animals)
            breed = np.random.choice(breeds.get(animal, ['mixed']))
            age = np.random.choice(['young', 'adult', 'senior'])
            weight = np.random.uniform(1.0, 500.0)
            
            common_symptoms = animal_symptoms.get(animal, ['fever', 'lethargy', 'loss_of_appetite'])
            num_symptoms = np.random.choice([2, 3, 4, 5, 6], p=[0.15, 0.25, 0.3, 0.2, 0.1])
            
            # Create symptom sequences with dependencies
            base_symptoms = []
            if common_symptoms:
                base_count = min(2, len(common_symptoms))
                base_symptoms = list(np.random.choice(common_symptoms, base_count, replace=False))
            
            # Add correlated symptoms
            remaining_symptoms = [s for s in symptoms_list if s not in base_symptoms]
            additional_count = num_symptoms - len(base_symptoms)
            
            if additional_count > 0 and remaining_symptoms:
                correlated_symptoms = []
                for symptom in base_symptoms:
                    if symptom == 'fever' and 'lethargy' in remaining_symptoms:
                        correlated_symptoms.append('lethargy')
                    elif symptom == 'vomiting' and 'dehydration' in remaining_symptoms:
                        correlated_symptoms.append('dehydration')
                    elif symptom == 'diarrhea' and 'dehydration' in remaining_symptoms:
                        correlated_symptoms.append('dehydration')
                
                correlated_symptoms = list(set(correlated_symptoms))
                remaining_after_correlation = [s for s in remaining_symptoms if s not in correlated_symptoms]
                
                available_from_correlation = min(len(correlated_symptoms), additional_count)
                needed_from_remaining = additional_count - available_from_correlation
                
                additional_symptoms = correlated_symptoms[:available_from_correlation]
                
                if needed_from_remaining > 0 and remaining_after_correlation:
                    actual_needed = min(needed_from_remaining, len(remaining_after_correlation))
                    additional_symptoms.extend(list(np.random.choice(
                        remaining_after_correlation, actual_needed, replace=False)))
                
                symptoms = base_symptoms + additional_symptoms
            else:
                symptoms = base_symptoms
            
            symptoms = symptoms[:num_symptoms]  # Ensure we don't exceed desired count

            # Enhanced danger assessment
            high_risk_count = sum(1 for s in symptoms if self.symptom_severity_weights.get(s, 0) > 0.7)
            medium_risk_count = sum(1 for s in symptoms if 0.4 < self.symptom_severity_weights.get(s, 0) <= 0.7)

            base_danger = (high_risk_count * 0.7 + medium_risk_count * 0.4) / max(len(symptoms), 1)

            # Clinical prior interactions
            interaction_boost = 0
            if 'fever' in symptoms and 'lethargy' in symptoms:
                interaction_boost += 0.1
            if 'vomiting' in symptoms and 'diarrhea' in symptoms:
                interaction_boost += 0.15
            if 'seizures' in symptoms and 'unconsciousness' in symptoms:
                interaction_boost += 0.25

            danger_score = base_danger + interaction_boost + np.random.normal(0, 0.02)

            # Enhanced demographic factors
            if age == 'senior':
                danger_score += 0.12
            elif age == 'young':
                danger_score += 0.06

            # Weight-based risk (extremes are riskier)
            if weight < 2.0 or weight > 300.0:
                danger_score += 0.04

            # Breed-specific vulnerabilities
            breed_risk_factors = {
                'bulldog': 0.06, 'persian': 0.05, 'arabian': 0.05, 'poodle': -0.02, 'labrador': -0.01
            }
            
            danger_score += breed_risk_factors.get(breed, 0)
            danger_score = max(0, min(1, danger_score))

            # Better balanced targets with controlled positive rate
            if danger_score > 0.5:
                dangerous = 'Yes'
            elif danger_score < 0.2:
                dangerous = 'No'
            else:
                prob_positive = (danger_score - 0.2) / 0.3
                adjusted_prob = prob_positive * (target_positive_rate / 0.6)
                dangerous = 'Yes' if np.random.random() < adjusted_prob else 'No'

            record = {
                'animal_id': i,
                'AnimalName': animal,
                'Breed': breed,
                'Age': age,
                'Weight': weight,
                'Symptom_Count': len(symptoms),
                'Dangerous': dangerous,
                'Danger_Score': danger_score,
                'High_Risk_Count': high_risk_count,
                'Medium_Risk_Count': medium_risk_count,
            }

            for j in range(10):
                record[f'Symptom_{j+1}'] = symptoms[j] if j < len(symptoms) else 'none'

            data.append(record)

        df = pd.DataFrame(data)
        print(f"[] Created enhanced dataset: {len(df)} records")
        dangerous_count = (df['Dangerous'] == 'Yes').sum()
        safe_count = (df['Dangerous'] == 'No').sum()
        print(f"[] Class balance - Dangerous: {dangerous_count} ({dangerous_count/len(df)*100:.1f}%), Safe: {safe_count} ({safe_count/len(df)*100:.1f}%)")

        return df

    def preprocess_data(self, df):
        """Preprocess the veterinary data"""
        print("[] Preprocessing data...")
        processed_df = df.copy()

        processed_df['AnimalName'] = processed_df['AnimalName'].str.lower().str.strip()
        processed_df['Breed'] = processed_df['Breed'].str.lower().str.strip()
        processed_df['Age'] = processed_df['Age'].str.lower().str.strip()

        symptom_columns = [col for col in processed_df.columns if col.startswith('Symptom_')]

        temp_symptoms = set()
        for col in symptom_columns:
            processed_df[col] = processed_df[col].astype(str).str.lower().str.strip()
            temp_symptoms.update(set(processed_df[col].unique()))
        
        self.all_symptoms = temp_symptoms
        self.all_symptoms.discard('none')
        self.all_symptoms.discard('unknown')
        self.all_symptoms.discard('nan')
        self.all_symptoms = sorted([s for s in self.all_symptoms if s and s != 'none'])
        self.all_animals = sorted(processed_df['AnimalName'].unique())
        
        processed_df['target'] = processed_df['Dangerous'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
        processed_df['target'] = processed_df['target'].fillna(0)
        
        self.le_animal = LabelEncoder()
        self.le_breed = LabelEncoder()
        self.le_age = LabelEncoder()

        processed_df['Animal_encoded'] = self.le_animal.fit_transform(processed_df['AnimalName'])
        processed_df['Breed_encoded'] = self.le_breed.fit_transform(processed_df['Breed'])
        processed_df['Age_encoded'] = self.le_age.fit_transform(processed_df['Age'])
        
        # Normalize weight
        processed_df['Weight_normalized'] = (processed_df['Weight'] - processed_df['Weight'].mean()) / processed_df['Weight'].std()
        
        # Ensure numeric types for PyTorch compatibility
        processed_df['High_Risk_Count'] = pd.to_numeric(processed_df['High_Risk_Count'], errors='coerce').fillna(0).astype(np.float32)
        processed_df['Medium_Risk_Count'] = pd.to_numeric(processed_df['Medium_Risk_Count'], errors='coerce').fillna(0).astype(np.float32)
        processed_df['Symptom_Count'] = pd.to_numeric(processed_df['Symptom_Count'], errors='coerce').fillna(0).astype(np.float32)

        self.processed_df = processed_df
        print(f"[] Dataset Summary:")
        print(f"  Total Records: {len(processed_df)}")
        print(f"  Animals: {len(self.all_animals)}")
        print(f"  Symptoms: {len(self.all_symptoms)}")
        print(f"  Dangerous cases: {processed_df['target'].sum()} ({processed_df['target'].mean()*100:.1f}%)")

        return processed_df