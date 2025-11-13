# utils/helpers.py - Fixed with corrected variable scope
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
    """Get the local AI model for report generation - Using a better model"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available - using fallback reports")
        return None
        
    try:
        # Use a more capable model for better results
        model_name = "microsoft/DialoGPT-medium"  # Better at following instructions
        
        # Load tokenizer and model separately for better control
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float32,
            device=0 if torch.cuda.is_available() else -1
        )
        return generator
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to distilgpt2 if the primary model fails
        try:
            model_name = "distilgpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            generator = pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float32,
                device=0 if torch.cuda.is_available() else -1
            )
            return generator
        except Exception as e2:
            print(f"Fallback model also failed: {e2}")
            return None

def generate_vet_report_local(prediction_result, animal_info, symptoms):
    """Generate a veterinary report using local AI model with improved prompting"""
    try:
        # Always use structured fallback for now due to AI model issues
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
                # FIXED: Use symptoms_list parameter instead of undefined symptoms variable
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

def generate_structured_fallback_report(prediction_result, animal_info, symptoms):
    """Generate a comprehensive structured fallback report"""
    
    risk_level = "HIGH - EMERGENCY" if prediction_result['ensemble']['dangerous'] else "LOW - MONITOR"
    confidence = prediction_result['ensemble']['confidence']
    probability = prediction_result['ensemble']['probability']
    
    # Calculate symptom analysis
    high_risk_symptoms = []
    medium_risk_symptoms = []
    low_risk_symptoms = []
    
    # Use the symptom_severity_weights from the predictor if available, otherwise use default
    symptom_severities = prediction_result.get('symptom_severity', {})
    for symptom in symptoms:
        severity = symptom_severities.get(symptom, 0.1)
        if severity > 0.7:
            high_risk_symptoms.append(symptom)
        elif severity > 0.4:
            medium_risk_symptoms.append(symptom)
        else:
            low_risk_symptoms.append(symptom)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
    
    report = f"""
VETERINARY HEALTH ASSESSMENT REPORT
===================================

REPORT SUMMARY
--------------
Generated: {timestamp}
Case ID: VET-{datetime.now().strftime('%Y%m%d%H%M')}
Status: {risk_level}

PATIENT INFORMATION
------------------
Species:       {animal_info['animal']}
Breed:         {animal_info['breed']}
Age Group:     {animal_info['age']}
Weight:        {animal_info['weight']} kg
Patient ID:    {animal_info['animal'][:3].upper()}-{random.randint(1000, 9999)}

SYMPTOM ANALYSIS
----------------
Total Symptoms Identified: {len(symptoms)}
├── High Risk Symptoms: {len(high_risk_symptoms)}
├── Medium Risk Symptoms: {len(medium_risk_symptoms)}
└── Low Risk Symptoms: {len(low_risk_symptoms)}

Detailed Symptom Breakdown:
─────────────────────────────────────────────────────
{'● ' + ', '.join(high_risk_symptoms) if high_risk_symptoms else '● No high-risk symptoms'}
{'● ' + ', '.join(medium_risk_symptoms) if medium_risk_symptoms else '● No medium-risk symptoms'}  
{'● ' + ', '.join(low_risk_symptoms) if low_risk_symptoms else '● No low-risk symptoms'}

RISK ASSESSMENT
---------------
Overall Risk Level:    {risk_level}
Assessment Confidence: {confidence}
Risk Probability:      {probability:.1%}
Model Agreement:       {prediction_result['ensemble']['model_agreement']}

CLINICAL ASSESSMENT
-------------------
Based on the comprehensive analysis of presenting symptoms and patient factors, 
this case has been classified as {risk_level.lower()}. The assessment indicates 
{confidence} confidence level with a risk probability of {probability:.1%}.

The patient presents with {len(symptoms)} clinical symptom(s) requiring professional 
veterinary evaluation. {f'Of particular concern are {len(high_risk_symptoms)} high-risk symptom(s) that warrant immediate attention.' if high_risk_symptoms else 'No immediate high-risk symptoms were identified.'}

KEY CLINICAL FINDINGS
---------------------
1. SYMPTOM PATTERN: Multiple clinical signs present requiring differential diagnosis
2. RISK FACTORS: {animal_info['age']} patient with breed-specific considerations
3. URGENCY LEVEL: { 'Requires emergency veterinary consultation' if prediction_result['ensemble']['dangerous'] else 'Routine veterinary consultation recommended'}
4. MONITORING NEEDS: Continuous observation of symptom progression

RECOMMENDED IMMEDIATE ACTIONS
-----------------------------
{'🚨 EMERGENCY PROTOCOL:' if prediction_result['ensemble']['dangerous'] else 'STANDARD PROTOCOL:'}
{'► Contact emergency veterinary services immediately' if prediction_result['ensemble']['dangerous'] else '► Schedule veterinary appointment within 24-48 hours'}
► Monitor vital signs (temperature, respiration, heart rate)
► Ensure access to fresh water and comfortable environment
► Document any changes in behavior or condition
► {'Prepare for emergency transport' if prediction_result['ensemble']['dangerous'] else 'Continue routine monitoring'}

TREATMENT CONSIDERATIONS
------------------------
• DIAGNOSTIC WORKUP: Complete physical examination and laboratory tests
• MEDICAL INTERVENTION: Species-appropriate treatment protocols
• SUPPORTIVE CARE: Pain management, hydration, nutritional support
• SPECIFIC CONSIDERATIONS: Age and breed-appropriate medications

FOLLOW-UP AND MONITORING PLAN
-----------------------------
┌─ SHORT-TERM (24-48 hours)
│  ► Re-evaluation of symptoms every 6-8 hours
│  ► Document response to any interventions
│  ► Monitor food and water intake
│
├─ MEDIUM-TERM (3-7 days)  
│  ► Follow-up veterinary consultation
│  ► Adjust treatment plan as needed
│  ► Continue symptom monitoring
│
└─ LONG-TERM (1-4 weeks)
   ► Comprehensive health assessment
   ► Preventive care planning
   ► Ongoing health maintenance

PROGNOSIS AND OUTCOME EXPECTATIONS
-----------------------------------
• SHORT-TERM OUTLOOK: Guarded to favorable based on timely intervention
• RECOVERY TIMELINE: Dependent on accurate diagnosis and treatment compliance
• LONG-TERM PROGNOSIS: Generally good with proper veterinary care and follow-up
• KEY SUCCESS FACTORS: Early intervention, proper diagnosis, owner compliance

EMERGENCY CONTACT PROTOCOL
--------------------------
IMMEDIATE ACTION REQUIRED IF:
► Symptoms worsen suddenly
► New concerning symptoms develop  
► Patient shows signs of distress
► Behavioral changes occur

CONTACT INFORMATION:
► Primary Veterinarian: [CLINIC NAME - PHONE]
► Emergency Clinic: [EMERGENCY CLINIC - PHONE]
► Animal Poison Control: [888-426-4435]

DISCLAIMER AND IMPORTANT NOTES
------------------------------
This automated assessment is generated for informational purposes only and 
should not replace professional veterinary diagnosis and treatment. Always 
consult with a licensed veterinarian for medical advice and treatment plans.

The accuracy of this assessment is based on the information provided and 
may require adjustment based on physical examination and diagnostic testing.

Report generated by Veterinary Health Assessment System v2.0
Confidence Score: {confidence}
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
    """Format symptom analysis for display with improved visibility"""
    if not symptoms:
        return ""
    
    # Calculate risk counts
    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0
    
    for symptom in symptoms:
        severity = predictor.data_loader.symptom_severity_weights.get(symptom, 0.1)
        if severity > 0.7:
            high_risk_count += 1
        elif severity > 0.4:
            medium_risk_count += 1
        else:
            low_risk_count += 1
    
    analysis = f"""
### 📊 Symptom Analysis Summary

**Total Symptoms:** {len(symptoms)}  
**High Risk:** {high_risk_count} | **Medium Risk:** {medium_risk_count} | **Low Risk:** {low_risk_count}

---

### 🔍 Detailed Symptom Breakdown
"""
    
    for symptom in symptoms:
        severity = predictor.data_loader.symptom_severity_weights.get(symptom, 0.1)
        if severity > 0.7:
            risk_level = "🔴 HIGH RISK"
            risk_class = "high-risk-symptom"
        elif severity > 0.4:
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
    
    if high_risk_count > 0:
        analysis += f"""
<div style="background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white; padding: 15px; border-radius: 10px; margin: 15px 0; text-align: center;">
    ⚠️ <strong>ALERT:</strong> {high_risk_count} high-risk symptom(s) detected requiring immediate attention
</div>
"""
    
    return analysis

def test_ai_connection():
    """Test if local AI model is working"""
    if not TORCH_AVAILABLE:
        return False, "❌ PyTorch/Transformers not installed"
    
    try:
        generator = get_report_generator()
        if generator is None:
            return False, "❌ AI model failed to load - using fallback reports"
        
        # Test with a short prompt
        test_response = generator("Test veterinary", max_new_tokens=20, num_return_sequences=1, truncation=True)
        return True, "✅ AI model loaded successfully!"
    except Exception as e:
        return False, f"❌ AI model error: {str(e)} - using fallback reports"