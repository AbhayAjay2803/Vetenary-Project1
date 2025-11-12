# utils/helpers.py - Updated with better report structure
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
            temperature=0.8,
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
    
    prompt = f"""
Generate a professional veterinary medical report:

PATIENT INFORMATION:
- Species: {animal_info['animal']}
- Breed: {animal_info['breed']} 
- Age: {animal_info['age']}
- Weight: {animal_info['weight']} kg

PRESENTING SYMPTOMS:
{', '.join(symptoms)}

RISK ASSESSMENT:
- Level: {risk_level}
- Confidence: {confidence}
- Probability: {probability}

Please create a detailed veterinary report with these sections:

CLINICAL ASSESSMENT:

KEY FINDINGS:

RECOMMENDED ACTIONS:

TREATMENT CONSIDERATIONS:

FOLLOW-UP PLAN:

PROGNOSIS:

Write a clear, professional veterinary report:
"""
    return prompt

def format_structured_report(report):
    """Format the AI-generated report to ensure proper structure"""
    # Clean up any weird text patterns
    unwanted_patterns = [
        'FACT:', 'OF THE MONISCADORS', 'AGE:', 'COMMUNITY:', 'HIVING:', 'EMBODY:',
        'SUBJECT:', 'MEMPHIS:', 'MONISCA:', 'CHILD:', 'TALE:', 'THE DIVEST',
        'LATEST FACTS', 'ALL', 'FOR THE MALE', 'THIRD PARTIES', 'CHURCHIC REPORT'
    ]
    
    for pattern in unwanted_patterns:
        report = report.replace(pattern, '')
    
    # Ensure basic sections exist with proper content
    sections = {
        "CLINICAL ASSESSMENT:": "Based on the presented symptoms and patient information, this case requires professional veterinary evaluation.",
        "KEY FINDINGS:": "Multiple symptoms requiring assessment. Further diagnostic evaluation recommended.",
        "RECOMMENDED ACTIONS:": "Schedule veterinary consultation. Monitor vital signs. Provide supportive care.",
        "TREATMENT CONSIDERATIONS:": "Treatment should be determined by licensed veterinarian based on complete diagnostic workup.",
        "FOLLOW-UP PLAN:": "Re-evaluate in 24 hours or if condition changes. Maintain communication with veterinary provider.",
        "PROGNOSIS:": "Dependent on accurate diagnosis and timely treatment. Follow veterinary guidance for optimal outcomes."
    }
    
    formatted_report = report
    
    for section, default_content in sections.items():
        if section not in formatted_report:
            formatted_report += f"\n\n{section}\n{default_content}"
        else:
            # Ensure section has meaningful content
            section_start = formatted_report.find(section) + len(section)
            next_section_start = len(formatted_report)
            
            # Find the start of the next section
            for other_section in sections:
                if other_section != section:
                    other_pos = formatted_report.find(other_section, section_start)
                    if other_pos != -1 and other_pos < next_section_start:
                        next_section_start = other_pos
            
            section_content = formatted_report[section_start:next_section_start].strip()
            if not section_content or len(section_content) < 10:
                formatted_report = formatted_report.replace(section_content, f"\n{default_content}")
    
    return formatted_report

def generate_structured_fallback_report(prediction_result, animal_info, symptoms):
    """Generate a structured fallback report with table-like formatting"""
    
    risk_level = "HIGH - EMERGENCY" if prediction_result['ensemble']['dangerous'] else "LOW - MONITOR"
    confidence = prediction_result['ensemble']['confidence']
    
    # Calculate symptom analysis
    high_risk_symptoms = []
    medium_risk_symptoms = []
    low_risk_symptoms = []
    
    for symptom in symptoms:
        severity = prediction_result.get('symptom_severity', {}).get(symptom, 0.1)
        if severity > 0.7:
            high_risk_symptoms.append(symptom)
        elif severity > 0.4:
            medium_risk_symptoms.append(symptom)
        else:
            low_risk_symptoms.append(symptom)
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    VETERINARY MEDICAL REPORT                         ║
╚══════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────┐
│ PATIENT INFORMATION                                                  │
├──────────────────────────────────────────────────────────────────────┤
│ Species:    {animal_info['animal']:<50} │
│ Breed:      {animal_info['breed']:<50} │
│ Age:        {animal_info['age']:<50} │
│ Weight:     {animal_info['weight']} kg{'':<40} │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SYMPTOM ANALYSIS                                                    │
├──────────────────────────────────────────────────────────────────────┤
│ Total Symptoms: {len(symptoms):<45} │
│ High Risk:      {len(high_risk_symptoms):<45} │
│ Medium Risk:    {len(medium_risk_symptoms):<45} │
│ Low Risk:       {len(low_risk_symptoms):<45} │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ PRESENTING SYMPTOMS                                                 │
├──────────────────────────────────────────────────────────────────────┤
{'│ ' + ', '.join(symptoms) + ' ' * (70 - len(', '.join(symptoms))) + '│'}
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ RISK ASSESSMENT                                                     │
├──────────────────────────────────────────────────────────────────────┤
│ Level:       {risk_level:<47} │
│ Confidence:  {confidence:<47} │
│ Probability: {prediction_result['ensemble']['probability']:.1%}{'':<38} │
└──────────────────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════════════════
CLINICAL ASSESSMENT:
══════════════════════════════════════════════════════════════════════════
Based on the symptom presentation and AI analysis, this case has been 
classified as {risk_level.lower()}. The assessment indicates {confidence} 
confidence in this evaluation.

══════════════════════════════════════════════════════════════════════════
KEY FINDINGS:
══════════════════════════════════════════════════════════════════════════
• {len(symptoms)} symptom(s) identified requiring professional evaluation
• {f"{len(high_risk_symptoms)} high-risk symptom(s) present" if high_risk_symptoms else "No immediate high-risk symptoms"}
• Species-specific considerations for {animal_info['animal']}
• Age factor: {animal_info['age']} patient

══════════════════════════════════════════════════════════════════════════
RECOMMENDED ACTIONS:
══════════════════════════════════════════════════════════════════════════
{"• 🚨 CONTACT EMERGENCY VETERINARIAN IMMEDIATELY" if prediction_result['ensemble']['dangerous'] else "• Schedule veterinary consultation within 24-48 hours"}
• Monitor vital signs (temperature, respiration, heart rate)
• Ensure access to fresh water and comfortable environment
• {"Prepare for emergency transport to veterinary facility" if prediction_result['ensemble']['dangerous'] else "Observe for any changes in condition"}
• Document symptom progression and behavioral changes

══════════════════════════════════════════════════════════════════════════
TREATMENT CONSIDERATIONS:
══════════════════════════════════════════════════════════════════════════
• Complete physical examination by licensed veterinarian
• Diagnostic testing based on clinical presentation
• Species-appropriate medical interventions
• Supportive care and nutritional support
• Pain management if indicated

══════════════════════════════════════════════════════════════════════════
FOLLOW-UP PLAN:
══════════════════════════════════════════════════════════════════════════
• Re-evaluation within 24 hours or as condition changes
• Daily monitoring of symptoms and behavior
• Maintain detailed health records
• Follow veterinary discharge instructions precisely
• Emergency contact information readily available

══════════════════════════════════════════════════════════════════════════
PROGNOSIS:
══════════════════════════════════════════════════════════════════════════
• Outcome dependent on accurate diagnosis and timely intervention
• Better prognosis with early veterinary care
• Follow professional guidance for optimal recovery
• Regular monitoring improves long-term outcomes

══════════════════════════════════════════════════════════════════════════
EMERGENCY PROTOCOL:
══════════════════════════════════════════════════════════════════════════
• Contact emergency veterinary services immediately if condition worsens
• Have local emergency clinic information readily available
• Transport patient safely to veterinary facility
• Bring medical history and current medication information

══════════════════════════════════════════════════════════════════════════
DISCLAIMER:
══════════════════════════════════════════════════════════════════════════
This report is generated by an AI system for informational purposes only
and should not replace professional veterinary diagnosis and treatment.
Always consult a licensed veterinarian for medical advice.

Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
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