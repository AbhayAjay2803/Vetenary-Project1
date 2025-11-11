# src/config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys with fallbacks
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    
    # Model paths
    MODEL_DIR = os.getenv('MODEL_DIR', 'models/')
    
    # Application settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Available animals and breeds
    ANIMAL_BREEDS = {
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
    
    SYMPTOMS = [
        'fever', 'vomiting', 'diarrhea', 'lethargy', 'loss_of_appetite', 'coughing', 'sneezing',
        'seizures', 'unconsciousness', 'bleeding', 'paralysis', 'rapid_breathing', 'jaundice',
        'pale_gums', 'abdominal_pain', 'dehydration', 'weight_loss', 'swelling', 'nasal_discharge',
        'eye_discharge', 'lameness', 'constipation', 'excessive_thirst', 'urination_problems',
        'reproductive_issues', 'skin_lesions', 'hair_loss', 'tremors', 'pain', 'colic', 'skin_rashes',
        'respiratory_distress', 'anorexia', 'weakness', 'depression', 'nausea', 'chills', 'headache',
        'muscle_pain', 'joint_pain', 'bloating', 'flatulence', 'bad_breath', 'discharge', 'inflammation',
        'redness', 'wheezing', 'gasping', 'convulsions', 'twitching', 'salivating', 'ulcers', 'lesions',
        'abscesses', 'infection', 'malaise', 'fatigue', 'emaciation', 'pneumonia', 'torticollis',
        'dyspnea', 'cyanosis', 'edema', 'ascites', 'anemia', 'hemorrhage', 'shock', 'coma', 'ataxia',
        'blindness', 'deafness', 'paraplegia', 'hyperesthesia', 'aggression', 'disorientation',
        'staggering', 'circling', 'head_pressing', 'nystagmus', 'opisthotonos', 'paresis'
    ]