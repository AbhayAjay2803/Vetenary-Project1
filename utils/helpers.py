# utils/helpers.py - Fixed meta tensor error and improved report generation
import os
import json
from datetime import datetime
import random

# Import torch with proper error handling
try:
    import torch
    from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch/Transformers not available: {e}")
    TORCH_AVAILABLE = False

from src.config import Config

def get_report_generator():
    """Get the local AI model for report generation - Fixed meta tensor error"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available - using fallback reports")
        return None
        
    try:
        # Use a smaller, more reliable model
        model_name = "distilgpt2"  # More reliable and smaller
        
        # Load tokenizer and model separately for better control
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with specific settings to avoid meta tensors
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False  # Disable low memory usage to avoid meta tensors
        )
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Move model to appropriate device
        device = 0 if torch.cuda.is_available() else -1
        if device == 0:
            model = model.cuda()
        else:
            model = model.cpu()
        
        generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=device,
            torch_dtype=torch.float32
        )
        return generator
    except Exception as e:
        print(f"Error loading AI model: {e}")
        return None

def generate_vet_report_local(prediction_result, animal_info, symptoms):
    """Generate a veterinary report using local AI model with improved prompting"""
    try:
        # Use structured fallback - more reliable than AI model
        return generate_structured_fallback_report(prediction_result, animal_info, symptoms)
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return generate_structured_fallback_report(prediction_result, animal_info, symptoms)

def create_structured_prompt(prediction_result, animal_info, symptoms):
    """Create a structured prompt for better AI output"""
    
    risk_level = "HIGH - EMERGENCY" if prediction_result['ensemble']['dangerous'] else "LOW - MONITOR"
    confidence = prediction_result['ensemble']['confidence']
    probability = f"{prediction_result['ensemble']['probability']:.1%}"
    
    prompt = f"""
Create a professional veterinary medical report for a {animal_info['animal']}.

PATIENT INFORMATION:
- Species: {animal_info['animal']}
- Breed: {animal_info['breed']}
- Age: {animal_info['age']}
- Weight: {animal_info['weight']} kg

SYMPTOMS: {', '.join(symptoms)}

RISK ASSESSMENT: {risk_level} (Confidence: {confidence}, Probability: {probability})

Please write a clear veterinary report with these sections:

CLINICAL ASSESSMENT:
[Provide overall assessment based on symptoms]

KEY FINDINGS:
[List main clinical observations]

RECOMMENDED ACTIONS:
[Immediate steps and monitoring]

TREATMENT CONSIDERATIONS:
[Potential treatments to discuss]

FOLLOW-UP PLAN:
[Monitoring schedule]

PROGNOSIS:
[Expected outcome]

Write a professional veterinary report:
"""
    return prompt

def format_structured_report(report, symptoms_list):
    """Format the AI-generated report to ensure proper structure"""
    # Clean up common AI generation artifacts
    unwanted_phrases = [
        "PASE TO:", "LATEST HAND:", "AGE OF", "MAJOR:", "CLOSE UP IN THE CHAPTER",
        "HIVING", "HORSING BRIEF", "INFORMING WONDER", "CLICK UP IN the CHAPTER",
        "HIRK ASSASSESSMENT", "PRODUCTION:", "Hospital:", "Inpatient:", "Fault:",
        "Clothing:", "Animal:", "Breathing:", "Vine:", "Headache:", "Dental:",
        "Cervical:", "Diaphragm:", "Blood:", "Surgery:", "Trauma:", "Stomach/Neurons:",
        "Other Medical Disorders", "HURK ASSOSESSMENT", "PENUARY:", "Treatment (injury):",
        "B. Cervical.", "H. Dental.", "C. Dementia.", "VINE:", "Medical Disorders:",
        "Dr. H. D.", "Dr", "Dr., Dr. D., Dr., Dr.. Dr. H., Dr, Dr.,",
        "Dr.. Dr.,Dr.,", ".. Dr.—Dr., D.,", "…Dr. D.'s", "HARMMENT: (Injury): (Injuries):",
        "C-Dementia:", "I-Dental-", "I. Dand/Dr.", "I", "Dand/D", "Harmment,", "I.'s/D"
    ]
    
    for phrase in unwanted_phrases:
        report = report.replace(phrase, '')
    
    # Ensure basic structure
    sections = [
        "CLINICAL ASSESSMENT:",
        "KEY FINDINGS:", 
        "RECOMMENDED ACTIONS:",
        "TREATMENT CONSIDERATIONS:",
        "FOLLOW-UP PLAN:",
        "PROGNOSIS:"
    ]
    
    formatted_report = report
    
    # Add missing sections with professional content
    for section in sections:
        if section not in formatted_report:
            if section == "CLINICAL ASSESSMENT:":
                symptoms_text = ', '.join(symptoms_list) if symptoms_list else "presented symptoms"
                formatted_report += f"\n\n{section}\nBased on the presented symptoms of {symptoms_text}, this case requires professional veterinary evaluation. The patient exhibits multiple clinical signs that warrant comprehensive assessment."
            elif section == "KEY FINDINGS:":
                formatted_report += f"\n\n{section}\n- Multiple clinical symptoms identified\n- Requires diagnostic evaluation\n- Species-specific considerations apply\n- Age and breed factors noted"
            elif section == "RECOMMENDED ACTIONS:":
                formatted_report += f"\n\n{section}\n- Schedule veterinary consultation\n- Monitor vital signs regularly\n- Provide supportive care\n- Document symptom progression"
            elif section == "TREATMENT CONSIDERATIONS:":
                formatted_report += f"\n\n{section}\n- Comprehensive physical examination\n- Diagnostic testing as indicated\n- Species-appropriate treatment protocols\n- Supportive medical care"
            elif section == "FOLLOW-UP PLAN:":
                formatted_report += f"\n\n{section}\n- Re-evaluation within 24-48 hours\n- Daily monitoring of condition\n- Emergency contact information available\n- Follow veterinary guidance precisely"
            elif section == "PROGNOSIS:":
                formatted_report += f"\n\n{section}\n- Dependent on accurate diagnosis and timely treatment\n- Better outcomes with early intervention\n- Follow professional veterinary advice"
    
    return formatted_report

def categorize_symptoms(symptoms, symptom_severity_weights):
    """Properly categorize symptoms into risk levels with accurate thresholds"""
    high_risk_symptoms = []
    medium_risk_symptoms = []
    low_risk_symptoms = []
    
    for symptom in symptoms:
        severity = symptom_severity_weights.get(symptom, 0.1)
        if severity > 0.7:  # High risk threshold
            high_risk_symptoms.append(symptom)
        elif severity > 0.4:  # Medium risk threshold
            medium_risk_symptoms.append(symptom)
        else:  # Low risk
            low_risk_symptoms.append(symptom)
    
    return high_risk_symptoms, medium_risk_symptoms, low_risk_symptoms

def get_symptom_treatments(symptoms, animal_type):
    """Get specific treatment considerations based on symptoms and animal type"""
    treatments = []
    
    # Gastrointestinal symptoms
    gi_symptoms = ['vomiting', 'diarrhea', 'loss_of_appetite', 'dehydration', 'bloating', 'constipation']
    if any(symptom in symptoms for symptom in gi_symptoms):
        treatments.extend([
            "Fluid therapy for hydration maintenance",
            "Dietary management: bland diet or prescription gastrointestinal food",
            "Antiemetics for vomiting control if indicated",
            "Gastroprotectants (e.g., famotidine, omeprazole)",
            "Probiotics for gut flora restoration"
        ])
    
    # Systemic symptoms
    systemic_symptoms = ['fever', 'lethargy', 'weakness', 'weight_loss']
    if any(symptom in symptoms for symptom in systemic_symptoms):
        treatments.extend([
            "Antipyretics for fever management",
            "Nutritional support and appetite stimulants if needed",
            "Comprehensive blood work (CBC, chemistry panel)",
            "Infectious disease testing as indicated"
        ])
    
    # Respiratory symptoms
    respiratory_symptoms = ['coughing', 'sneezing', 'nasal_discharge', 'rapid_breathing']
    if any(symptom in symptoms for symptom in respiratory_symptoms):
        treatments.extend([
            "Thoracic radiographs for respiratory assessment",
            "Bronchodilators if bronchoconstriction present",
            "Antibiotics for suspected bacterial infection",
            "Antitussives for cough control when appropriate"
        ])
    
    # Neurological symptoms (high priority)
    neuro_symptoms = ['seizures', 'tremors', 'unconsciousness', 'paralysis']
    if any(symptom in symptoms for symptom in neuro_symptoms):
        treatments.extend([
            "IMMEDIATE NEUROLOGICAL ASSESSMENT REQUIRED",
            "Anticonvulsant therapy for seizure control",
            "Advanced imaging (MRI/CT) if indicated",
            "Neurology consultation recommended"
        ])
    
    # Pain-related symptoms
    pain_symptoms = ['pain', 'abdominal_pain', 'lameness', 'muscle_pain']
    if any(symptom in symptoms for symptom in pain_symptoms):
        treatments.extend([
            "Multimodal pain management protocol",
            "NSAIDs appropriate for species (e.g., carprofen, meloxicam)",
            "Additional analgesics as needed (e.g., gabapentin, tramadol)",
            "Physical therapy and mobility support"
        ])
    
    # Dermatological symptoms
    derm_symptoms = ['skin_lesions', 'hair_loss', 'skin_rashes', 'itching']
    if any(symptom in symptoms for symptom in derm_symptoms):
        treatments.extend([
            "Dermatological examination and skin scrapings",
            "Antipruritic medications for itch relief",
            "Medicated shampoos and topical therapies",
            "Allergy testing and management if indicated"
        ])
    
    # Species-specific considerations
    if animal_type.lower() in ['dog', 'cat']:
        treatments.append("Species-specific medication dosing and monitoring")
    elif animal_type.lower() in ['rabbit', 'guinea_pig']:
        treatments.extend([
            "GI motility agents for herbivore digestive health",
            "Critical care nutritional support",
            "Dental examination for malocclusion"
        ])
    elif animal_type.lower() in ['bird', 'parrot']:
        treatments.extend([
            "Avian-specific diagnostic protocols",
            "Environmental and nutritional assessment",
            "Specialized avian veterinary consultation"
        ])
    
    return list(set(treatments))  # Remove duplicates

def get_diagnostic_recommendations(symptoms, animal_type):
    """Get specific diagnostic recommendations based on symptoms"""
    diagnostics = []
    
    # Base diagnostics for all cases
    base_diagnostics = [
        "Complete physical examination with vital signs",
        "Comprehensive blood work (CBC, biochemistry profile)",
        "Urinalysis for systemic assessment"
    ]
    
    diagnostics.extend(base_diagnostics)
    
    # Symptom-specific diagnostics
    if any(s in symptoms for s in ['vomiting', 'diarrhea', 'abdominal_pain', 'bloating']):
        diagnostics.extend([
            "Fecal examination for parasites and pathogens",
            "Abdominal radiographs and/or ultrasound",
            "Pancreatic testing (canine/feline pancreatic lipase)"
        ])
    
    if any(s in symptoms for s in ['coughing', 'sneezing', 'nasal_discharge', 'rapid_breathing']):
        diagnostics.extend([
            "Thoracic radiographs",
            "Respiratory PCR panel",
            "Tracheal/bronchial wash for cytology and culture"
        ])
    
    if any(s in symptoms for s in ['seizures', 'tremors', 'unconsciousness', 'paralysis']):
        diagnostics.extend([
            "Neurological examination",
            "Advanced imaging (MRI/CT) if indicated",
            "Cerebrospinal fluid analysis"
        ])
    
    if any(s in symptoms for s in ['skin_lesions', 'hair_loss', 'skin_rashes', 'itching']):
        diagnostics.extend([
            "Skin scrapings and cytology",
            "Fungal culture",
            "Allergy testing if indicated"
        ])
    
    if any(s in symptoms for s in ['lethargy', 'fever', 'weight_loss', 'weakness']):
        diagnostics.extend([
            "Infectious disease testing",
            "Thyroid function testing",
            "Additional organ-specific testing as indicated"
        ])
    
    return diagnostics

def generate_structured_fallback_report(prediction_result, animal_info, symptoms):
    """Generate a comprehensive structured fallback report with improved symptom categorization"""
    
    risk_level = "HIGH - EMERGENCY" if prediction_result['ensemble']['dangerous'] else "LOW - MONITOR"
    confidence = prediction_result['ensemble']['confidence']
    probability = prediction_result['ensemble']['probability']
    
    # Get symptom severity weights from prediction result or use default
    symptom_severities = prediction_result.get('symptom_severity_weights', {})
    
    # PROPERLY categorize symptoms using the helper function
    high_risk_symptoms, medium_risk_symptoms, low_risk_symptoms = categorize_symptoms(symptoms, symptom_severities)
    
    # Get specific treatment considerations
    specific_treatments = get_symptom_treatments(symptoms, animal_info['animal'])
    diagnostic_recommendations = get_diagnostic_recommendations(symptoms, animal_info['animal'])
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
    
    # Calculate symptom statistics
    total_symptoms = len(symptoms)
    high_risk_count = len(high_risk_symptoms)
    medium_risk_count = len(medium_risk_symptoms)
    low_risk_count = len(low_risk_symptoms)
    
    # Clinical significance assessment
    if high_risk_count > 0:
        clinical_significance = "HIGH CLINICAL SIGNIFICANCE - Requires immediate veterinary attention"
        urgency_note = "EMERGENCY: Immediate veterinary intervention required"
    elif medium_risk_count >= 2:
        clinical_significance = "MODERATE CLINICAL SIGNIFICANCE - Veterinary consultation recommended within 24 hours"
        urgency_note = "URGENT: Schedule veterinary appointment promptly"
    else:
        clinical_significance = "LOW CLINICAL SIGNIFICANCE - Routine monitoring advised"
        urgency_note = "ROUTINE: Monitor and schedule veterinary consultation as needed"
    
    # Format symptoms for display
    high_risk_display = ', '.join(high_risk_symptoms) if high_risk_symptoms else 'None identified'
    medium_risk_display = ', '.join(medium_risk_symptoms) if medium_risk_symptoms else 'None identified'
    low_risk_display = ', '.join(low_risk_symptoms) if low_risk_symptoms else 'None identified'
    
    report = f"""
VETERINARY HEALTH ASSESSMENT REPORT
===================================

REPORT SUMMARY
--------------
Generated: {timestamp}
Case ID: VET-{datetime.now().strftime('%Y%m%d%H%M')}
Status: {risk_level}
Clinical Significance: {clinical_significance}
Urgency Note: {urgency_note}

PATIENT INFORMATION
------------------
Species:       {animal_info['animal'].title()}
Breed:         {animal_info['breed'].replace('_', ' ').title()}
Age Group:     {animal_info['age'].title()}
Weight:        {animal_info['weight']} kg
Patient ID:    {animal_info['animal'][:3].upper()}-{random.randint(1000, 9999)}

SYMPTOM ANALYSIS
----------------
Total Symptoms Identified: {total_symptoms}
├── High Risk Symptoms: {high_risk_count}
├── Medium Risk Symptoms: {medium_risk_count}
└── Low Risk Symptoms: {low_risk_count}

Detailed Symptom Breakdown:
─────────────────────────────────────────────────────
HIGH RISK:    {high_risk_display}
MEDIUM RISK:  {medium_risk_display}
LOW RISK:     {low_risk_display}

RISK ASSESSMENT
---------------
Overall Risk Level:    {risk_level}
Assessment Confidence: {confidence}
Risk Probability:      {probability:.1%}
Model Agreement:       {prediction_result['ensemble']['model_agreement']}

CLINICAL ASSESSMENT
-------------------
Based on comprehensive analysis of presenting symptoms and patient factors, 
this case has been classified as **{risk_level}**. 

The patient presents with {total_symptoms} clinical symptom(s):
- **{high_risk_count} high-risk symptom(s)** requiring immediate attention
- **{medium_risk_count} medium-risk symptom(s)** warranting veterinary consultation  
- **{low_risk_count} low-risk symptom(s)** for ongoing monitoring

CLINICAL PRIORITIES:
1. Address high-risk symptoms immediately if present
2. Manage medium-risk symptoms with appropriate medical intervention
3. Monitor low-risk symptoms for progression or resolution

KEY CLINICAL FINDINGS
---------------------
1. SYMPTOM PATTERN: {total_symptoms} clinical signs identified requiring differential diagnosis
2. RISK PROFILE: {high_risk_count} urgent, {medium_risk_count} concerning, {low_risk_count} monitoring symptoms
3. PATIENT FACTORS: {animal_info['age']} {animal_info['breed'].replace('_', ' ')} with weight {animal_info['weight']}kg
4. URGENCY LEVEL: {'EMERGENCY - Immediate veterinary care required' if prediction_result['ensemble']['dangerous'] else 'Routine veterinary consultation recommended'}

RECOMMENDED IMMEDIATE ACTIONS
-----------------------------
{'🚨 EMERGENCY PROTOCOL:' if prediction_result['ensemble']['dangerous'] else 'STANDARD PROTOCOL:'}
{'► Contact emergency veterinary services IMMEDIATELY' if prediction_result['ensemble']['dangerous'] else '► Schedule veterinary appointment within 24-48 hours'}
► Monitor vital signs every 4-6 hours (temperature, respiration, heart rate)
► Ensure access to fresh water and prevent dehydration
► Provide quiet, comfortable resting environment
► Document any changes in behavior, appetite, or symptom progression
► {'Prepare for emergency transport to veterinary facility' if prediction_result['ensemble']['dangerous'] else 'Continue routine monitoring with increased vigilance'}

DIAGNOSTIC RECOMMENDATIONS
--------------------------
Essential Diagnostic Workup:
{chr(10).join(['• ' + diagnostic for diagnostic in diagnostic_recommendations])}

TREATMENT CONSIDERATIONS
------------------------
Based on the presenting symptoms of {', '.join([s.replace('_', ' ').title() for s in symptoms])}, consider the following treatment approaches:

PRIMARY MEDICAL INTERVENTIONS:
{chr(10).join(['• ' + treatment for treatment in specific_treatments[:6]])}

SUPPORTIVE CARE MEASURES:
• Fluid therapy maintenance and hydration support
• Nutritional management appropriate for condition
• Environmental modifications for comfort
• Pain management if discomfort present
• Monitoring and documentation of clinical response

MEDICATION CONSIDERATIONS:
• Anti-emetics for vomiting control if persistent
• Gastroprotectants for gastric support
• Antibiotics if bacterial infection suspected
• Anti-inflammatory medications as indicated
• Species-specific pharmacological protocols

FOLLOW-UP AND MONITORING PLAN
-----------------------------
┌─ IMMEDIATE (0-24 hours)
│  ► Re-evaluate symptoms every 4-6 hours
│  ► Document food and water intake
│  ► Monitor for any symptom progression
│  ► {'EMERGENCY: Seek immediate veterinary care if worsening' if prediction_result['ensemble']['dangerous'] else 'Contact veterinarian if condition deteriorates'}
│
├─ SHORT-TERM (24-72 hours)
│  ► Veterinary re-examination as scheduled
│  ► Adjust treatment plan based on diagnostic results
│  ► Continue supportive care measures
│  ► Monitor response to interventions
│
├─ MEDIUM-TERM (3-7 days)
│  ► Follow-up veterinary consultation
│  ► Assess treatment efficacy and adjust as needed
│  ► Continue symptom monitoring
│  ► Implement preventive care measures
│
└─ LONG-TERM (1-4 weeks)
   ► Comprehensive health reassessment
   ► Review diagnostic findings and treatment outcomes
   ► Develop ongoing health maintenance plan
   ► Establish preventive healthcare schedule

PROGNOSIS AND OUTCOME EXPECTATIONS
-----------------------------------
• SHORT-TERM OUTLOOK: {'Guarded to poor without immediate intervention' if prediction_result['ensemble']['dangerous'] else 'Fair to good with appropriate care'}
• RECOVERY TIMELINE: Dependent on accurate diagnosis, treatment compliance, and individual patient response
• LONG-TERM PROGNOSIS: {'Dependent on emergency intervention success' if prediction_result['ensemble']['dangerous'] else 'Generally favorable with proper veterinary management'}
• CRITICAL SUCCESS FACTORS: 
  - Timely veterinary intervention
  - Accurate diagnosis and appropriate treatment
  - Owner compliance with medical recommendations
  - Regular monitoring and follow-up care

EMERGENCY CONTACT PROTOCOL
--------------------------
IMMEDIATE ACTION REQUIRED IF:
► Symptoms worsen suddenly or new concerning symptoms develop
► Patient shows signs of severe distress or pain
► Breathing difficulties or respiratory distress occur
► Seizures, collapse, or loss of consciousness
► Persistent vomiting or diarrhea with dehydration signs
► Behavioral changes indicating severe discomfort

CONTACT INFORMATION:
► Primary Veterinarian: [CLINIC NAME - PHONE]
► Emergency Clinic: [EMERGENCY CLINIC - PHONE]
► Animal Poison Control: [888-426-4435]
► After-Hours Emergency: [LOCAL EMERGENCY SERVICE]

DISCLAIMER AND IMPORTANT NOTES
------------------------------
This automated assessment is generated for informational purposes only and 
should not replace professional veterinary diagnosis and treatment. Always 
consult with a licensed veterinarian for medical advice and treatment plans.

The accuracy of this assessment is based on the information provided and 
may require adjustment based on physical examination, diagnostic testing, 
and clinical judgment by a qualified veterinarian.

CRITICAL: This report does not constitute medical advice. All treatment 
decisions should be made by a licensed veterinarian following thorough 
clinical examination.

Report generated by Veterinary Health Assessment System v2.1
Assessment Confidence: {confidence}
Clinical Significance: {clinical_significance}
"""
    return report

def get_risk_color(probability):
    """Get color based on risk probability"""
    if probability < 0.3:
        return "🟢"  # Green
    elif probability < 0.7:
        return "🟡"  # Yellow
    else:
        return "🔴"  # Red

def format_symptom_analysis(symptoms, predictor):
    """Format symptom analysis for display with improved visibility and ACCURATE categorization"""
    if not symptoms:
        return ""
    
    # Use the proper categorization function
    high_risk_symptoms, medium_risk_symptoms, low_risk_symptoms = categorize_symptoms(
        symptoms, 
        predictor.data_loader.symptom_severity_weights
    )
    
    analysis = f"""
### 📊 Symptom Analysis Summary

**Total Symptoms:** {len(symptoms)}  
**High Risk:** {len(high_risk_symptoms)} | **Medium Risk:** {len(medium_risk_symptoms)} | **Low Risk:** {len(low_risk_symptoms)}

---

### 🔍 Detailed Symptom Breakdown
"""
    
    # Display symptoms in risk order: High -> Medium -> Low
    for symptom in high_risk_symptoms + medium_risk_symptoms + low_risk_symptoms:
        severity = predictor.data_loader.symptom_severity_weights.get(symptom, 0.1)
        if symptom in high_risk_symptoms:
            risk_level = "🔴 HIGH RISK"
            risk_class = "high-risk-symptom"
        elif symptom in medium_risk_symptoms:
            risk_level = "🟡 MEDIUM RISK" 
            risk_class = "medium-risk-symptom"
        else:
            risk_level = "🟢 LOW RISK"
            risk_class = "low-risk-symptom"
        
        # Create a visual severity bar
        severity_bar = "█" * int(severity * 10) + "░" * (10 - int(severity * 10))
        
        analysis += f"""
<div class="symptom-item {risk_class}">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <strong>{symptom.replace('_', ' ').title()}</strong><br>
            <small>{risk_level} | Severity: {severity:.2f}</small>
        </div>
        <div style="text-align: right; font-family: monospace;">
            {severity_bar}<br>
            <small>{severity:.0%}</small>
        </div>
    </div>
</div>
"""
    
    if high_risk_symptoms:
        analysis += f"""
<div style="background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white; padding: 15px; border-radius: 10px; margin: 15px 0; text-align: center;">
    ⚠️ <strong>ALERT:</strong> {len(high_risk_symptoms)} high-risk symptom(s) detected requiring immediate attention
</div>
"""
    
    if medium_risk_symptoms and not high_risk_symptoms:
        analysis += f"""
<div style="background: linear-gradient(135deg, #feca57, #ff9ff3); color: #2c3e50; padding: 15px; border-radius: 10px; margin: 15px 0; text-align: center;">
    ⚠️ <strong>NOTICE:</strong> {len(medium_risk_symptoms)} medium-risk symptom(s) detected requiring veterinary consultation
</div>
"""
    
    return analysis

def test_ai_connection():
    """Test if local AI model is working - Fixed to avoid meta tensors"""
    if not TORCH_AVAILABLE:
        return False, "❌ PyTorch/Transformers not installed"
    
    try:
        generator = get_report_generator()
        if generator is None:
            return False, "❌ AI model failed to load - using enhanced fallback reports"
        
        # Test with a very short prompt to avoid meta tensor issues
        try:
            test_response = generator(
                "Test", 
                max_new_tokens=10, 
                num_return_sequences=1, 
                truncation=True,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            return True, "✅ AI model loaded successfully!"
        except Exception as e:
            return False, f"❌ AI model test failed: {str(e)} - using enhanced fallback reports"
            
    except Exception as e:
        return False, f"❌ AI model error: {str(e)} - using enhanced fallback reports"