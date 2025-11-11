import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    
    # Model paths
    MODEL_DIR = "models/"
    
    # Available animals and breeds
    ANIMAL_BREEDS = {
        'dog': ['labrador', 'german_shepherd', 'golden_retriever', 'bulldog', 'poodle',
               'beagle', 'boxer', 'dachshund', 'siberian_husky', 'australian_shepherd'],
        'cat': ['siamese', 'persian', 'maine_coon', 'bengal', 'ragdoll',
               'british_shorthair', 'sphynx', 'russian_blue', 'scottish_fold', 'burmese'],
        'cow': ['holstein', 'angus', 'hereford', 'jersey', 'guernsey', 'limousin', 'charolais'],
        'horse': ['arabian', 'quarter_horse', 'thoroughbred', 'appaloosa', 'andalusian',
                 'friesian', 'mustang', 'clydesdale'],
        'rabbit': ['dutch', 'flemish_giant', 'rex', 'mini_lop', 'lionhead', 'holland_lop'],
        'goat': ['boer', 'nubian', 'alpine', 'saanen', 'toggenburg'],
        'sheep': ['merino', 'dorset', 'suffolk', 'rambouillet', 'dorper'],
        'chicken': ['rhode_island_red', 'leghorn', 'sussex', 'plymouth_rock', 'silkie'],
        'pig': ['yorkshire', 'duroc', 'hampshire', 'berkshire', 'landrace'],
        'parrot': ['african_grey', 'macaw', 'cockatoo', 'amazon', 'cockatiel', 'budgerigar'],
        'hamster': ['syrian', 'dwarf', 'roborovski', 'campbell', 'winter_white'],
        'guinea_pig': ['american', 'abby', 'peruvian', 'silkie', 'teddy'],
        'turtle': ['red_eared_slider', 'box_turtle', 'painted_turtle', 'snapping_turtle'],
        'duck': ['pekin', 'mallard', 'muscovy', 'rouen', 'khaki_campbell'],
        'turkey': ['broad_breasted_bronze', 'bourbon_red', 'narragansett', 'royal_palm']
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