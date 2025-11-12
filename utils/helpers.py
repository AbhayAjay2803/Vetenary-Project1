# utils/helpers.py
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
    """Get the local AI model for report generation"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available - using fallback reports")
        return None
        
    try:
        # Use a small, fast model for local generation
        model_name = "distilgpt2"
        
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
        return None

def generate_vet_report_local(prediction_result, animal_info, symptoms):
    """Generate a veterinary report using local AI model"""
    try:
        if not TORCH_AVAILABLE:
            return generate_structured_fallback_report(prediction_result, animal_info, symptoms)
            
        # Set seed for reproducibility
        set_seed(42)
        
        # Get the generator
        generator = get_report_generator()
        if generator is None:
            return generate_structured_fallback_report(prediction_result, animal_info, symptoms)
        
        # Create a more structured prompt
        prompt = create_structured_prompt(prediction_result, animal_info, symptoms)
        
        # Generate report with better parameters
        generated_text = generator(
            prompt,
            max_new_tokens=400,
            num_return_sequences=1,
            temperature=0.8,  # Slightly higher for more creativity
            do_sample=True,
            pad_token_id=50256,
            truncation=True,
            no_repeat_ngram_size=3
        )[0]['generated_text']
        
        # Extract and clean the generated part
        report = generated_text[len(prompt):].strip()
        
        # Format the report
        formatted_report = format_structured_report(report)
        
        return formatted_report
        
    except Exception as e:
        print(f"Error generating report with AI: {e}")
        return generate_structured_fallback_report(prediction_result, animal_info, symptoms)

def create_structured_prompt(prediction_result, animal_info, symptoms):
    """Create a structured prompt for better AI output"""
    
    risk_level = "HIGH - EMERGENCY" if prediction_result['ensemble']['dangerous'] else "LOW - MONITOR"
    confidence = prediction_result['ensemble']['confidence']
    probability = f"{prediction_result['ensemble']['probability']:.1%}"
    
    # High-risk symptoms
    high_risk_symptoms = [s for s in symptoms if prediction_result.get('symptom_severity', {}).get(s, 0) > 0.7]
    
    prompt = f"""
Create a veterinary medical report for a {animal_info['animal']}:

Patient: {animal_info['breed']} {animal_info['animal']}, {animal_info['age']}, {animal_info['weight']}kg
Symptoms: {', '.join(symptoms)}
Risk Level: {risk_level} ({probability})

Please generate a professional veterinary report with these sections:

CLINICAL ASSESSMENT:
Provide a clear assessment of the patient's condition based on the symptoms.

KEY FINDINGS:
- List the main clinical observations
- Note any high-risk symptoms

RECOMMENDED ACTIONS:
- Immediate steps to take
- Monitoring requirements
- When to seek emergency care

TREATMENT SUGGESTIONS:
- Potential treatments to discuss with veterinarian
- Supportive care recommendations

FOLLOW-UP PLAN:
- Monitoring schedule
- When to re-evaluate

PROGNOSIS:
- Expected outcome based on current assessment

Please write a clear, professional report:
"""
    return prompt

def format_structured_report(report):
    """Format the AI-generated report to ensure proper structure"""
    # Clean up any weird formatting
    report = report.replace('FACT:', '').replace('OF THE MONISCADORS', '').replace('AGE:', '')
    report = report.replace('COMMUNITY:', '').replace('HIVING:', '').replace('EMBODY:', '')
    report = report.replace('SUBJECT:', '').replace('MEMPHIS:', '').replace('MONISCA:', '')
    report = report.replace('CHILD:', '').replace('TALE:', '').replace('THE DIVEST', '')
    report = report.replace('LATEST FACTS', '').replace('ALL', '').replace('FOR THE MALE', '')
    report = report.replace('THIRD PARTIES', '').replace('CHURCHIC REPORT', 'VETERINARY REPORT')
    
    # Ensure basic sections exist
    sections = [
        "CLINICAL ASSESSMENT:",
        "KEY FINDINGS:", 
        "RECOMMENDED ACTIONS:",
        "TREATMENT SUGGESTIONS:",
        "FOLLOW-UP PLAN:",
        "PROGNOSIS:"
    ]
    
    formatted_report = report
    
    # Add missing sections with template content
    for section in sections:
        if section not in formatted_report:
            if section == "CLINICAL ASSESSMENT:":
                formatted_report += f"\n\n{section}\nBased on the presented symptoms, this case requires professional veterinary evaluation."
            elif section == "KEY FINDINGS:":
                formatted_report += f"\n\n{section}\n- Multiple symptoms present requiring assessment"
            elif section == "RECOMMENDED ACTIONS:":
                formatted_report += f"\n\n{section}\n- Schedule veterinary consultation\n- Monitor vital signs\n- Provide supportive care"
            elif section == "TREATMENT SUGGESTIONS:":
                formatted_report += f"\n\n{section}\n- Treatment should be determined by licensed veterinarian\n- Follow professional medical advice"
            elif section == "FOLLOW-UP PLAN:":
                formatted_report += f"\n\n{section}\n- Re-evaluate in 24 hours or if condition worsens\n- Maintain communication with veterinary provider"
            elif section == "PROGNOSIS:":
                formatted_report += f"\n\n{section}\n- Dependent on accurate diagnosis and timely treatment\n- Follow veterinary guidance for best outcomes"
    
    return formatted_report

def generate_structured_fallback_report(prediction_result, animal_info, symptoms):
    """Generate a structured fallback report"""
    
    risk_level = "HIGH - EMERGENCY" if prediction_result['ensemble']['dangerous'] else "LOW - MONITOR"
    confidence = prediction_result['ensemble']['confidence']
    
    # Calculate symptom severity summary
    high_risk_count = sum(1 for s in symptoms if prediction_result.get('symptom_severity', {}).get(s, 0) > 0.7)
    medium_risk_count = sum(1 for s in symptoms if 0.4 < prediction_result.get('symptom_severity', {}).get(s, 0) <= 0.7)
    
    report = f"""
VETERINARY MEDICAL REPORT
=========================

PATIENT INFORMATION:
-------------------
Species: {animal_info['animal']}
Breed: {animal_info['breed']}
Age: {animal_info['age']}
Weight: {animal_info['weight']} kg

PRESENTING SYMPTOMS:
-------------------
{chr(10).join(f"- {symptom}" for symptom in symptoms)}

RISK ASSESSMENT:
---------------
Overall Risk: {risk_level}
Confidence: {confidence}
Probability: {prediction_result['ensemble']['probability']:.1%}

SYMPTOM ANALYSIS:
----------------
Total Symptoms: {len(symptoms)}
High-Risk Symptoms: {high_risk_count}
Medium-Risk Symptoms: {medium_risk_count}

CLINICAL ASSESSMENT:
-------------------
Based on the symptom presentation and risk assessment, this case has been classified as {risk_level.lower()}. 
The system indicates {confidence} confidence in this assessment.

KEY FINDINGS:
------------
- Multiple clinical symptoms requiring professional evaluation
- {f"{high_risk_count} high-risk symptom(s) identified" if high_risk_count > 0 else "No immediate high-risk symptoms"}
- Species-specific considerations for {animal_info['animal']}
- Age factor: {animal_info['age']} patient

RECOMMENDED ACTIONS:
-------------------
{"🚨 IMMEDIATE EMERGENCY CARE REQUIRED:" if prediction_result['ensemble']['dangerous'] else "Recommended Steps:"}
- {"Contact emergency veterinarian immediately" if prediction_result['ensemble']['dangerous'] else "Schedule veterinary consultation"}
- Monitor vital signs (temperature, heart rate, respiration)
- Ensure access to fresh water
- Keep patient in quiet, comfortable environment
- {"Prepare for emergency transport" if prediction_result['ensemble']['dangerous'] else "Monitor for changes in condition"}

TREATMENT SUGGESTIONS:
---------------------
- Complete physical examination by licensed veterinarian
- Diagnostic tests as indicated by clinical presentation
- Species-appropriate treatment protocols
- Supportive care and monitoring

FOLLOW-UP PLAN:
--------------
- Re-evaluation within 24 hours
- Daily monitoring of symptoms
- Document any changes in condition
- Follow veterinary discharge instructions

PROGNOSIS:
---------
- Dependent on accurate diagnosis and treatment
- Timely intervention improves outcomes
- Follow professional veterinary guidance

EMERGENCY PROTOCOL:
------------------
- Contact emergency services if condition worsens
- Have veterinary clinic information readily available
- Transport safely to veterinary facility

DISCLAIMER:
----------
This report is generated by an AI system for informational purposes only.
Always consult a licensed veterinarian for professional diagnosis and treatment.
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
    """Format symptom analysis for display"""
    if not symptoms:
        return ""
    
    analysis = "### 🔍 Symptom Analysis\n\n"
    total_severity = 0
    high_risk_count = 0
    
    for symptom in symptoms:
        severity = predictor.data_loader.symptom_severity_weights.get(symptom, 0.1)
        total_severity += severity
        
        if severity > 0.7:
            risk_level = "🔴 HIGH RISK"
            high_risk_count += 1
        elif severity > 0.4:
            risk_level = "🟡 MEDIUM RISK"
        else:
            risk_level = "🟢 LOW RISK"
        
        analysis += f"- **{symptom}**: {risk_level} (Severity: {severity:.2f})\n"
    
    if high_risk_count > 0:
        analysis += f"\n⚠️ **Warning**: {high_risk_count} high-risk symptom(s) detected!\n"
    
    return analysis

def test_ai_connection():
    """Test if local AI model is working"""
    if not TORCH_AVAILABLE:
        return False, "❌ PyTorch/Transformers not installed"
    
    try:
        generator = get_report_generator()
        if generator is None:
            return False, "❌ Local AI model failed to load"
        
        # Test with a short prompt
        test_response = generator("Test", max_new_tokens=10, num_return_sequences=1, truncation=True)
        return True, "✅ Local AI model loaded successfully!"
    except Exception as e:
        return False, f"❌ Local AI model error: {str(e)}"