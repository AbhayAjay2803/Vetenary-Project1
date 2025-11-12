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

# Initialize the text generation pipeline
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
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
        return generator
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def generate_vet_report_local(prediction_result, animal_info, symptoms):
    """Generate a veterinary report using local AI model"""
    try:
        if not TORCH_AVAILABLE:
            return generate_fallback_report(prediction_result, animal_info, symptoms)
            
        # Set seed for reproducibility
        set_seed(42)
        
        # Get the generator
        generator = get_report_generator()
        if generator is None:
            return generate_fallback_report(prediction_result, animal_info, symptoms)
        
        # Create prompt
        prompt = create_report_prompt(prediction_result, animal_info, symptoms)
        
        # Generate report with fixed parameters to avoid warnings
        generated_text = generator(
            prompt,
            max_new_tokens=350,  # Reduced for faster generation
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=50256,
            truncation=True,  # Explicitly enable truncation
            no_repeat_ngram_size=2
        )[0]['generated_text']
        
        # Extract the generated part (remove the prompt)
        report = generated_text[len(prompt):].strip()
        
        # Format the report
        formatted_report = format_generated_report(report)
        
        return formatted_report
        
    except Exception as e:
        print(f"Error generating report with AI: {e}")
        return generate_fallback_report(prediction_result, animal_info, symptoms)

def create_report_prompt(prediction_result, animal_info, symptoms):
    """Create a detailed prompt for report generation"""
    
    risk_level = "HIGH - EMERGENCY" if prediction_result['ensemble']['dangerous'] else "LOW - MONITOR"
    confidence = prediction_result['ensemble']['confidence']
    probability = f"{prediction_result['ensemble']['probability']:.1%}"
    
    prompt = f"""
Generate a veterinary medical report:

PATIENT:
- Species: {animal_info['animal']}
- Breed: {animal_info['breed']}
- Age: {animal_info['age']}
- Weight: {animal_info['weight']} kg

SYMPTOMS: {', '.join(symptoms)}

RISK: {risk_level}
CONFIDENCE: {confidence}
PROBABILITY: {probability}

VETERINARY REPORT:

CLINICAL ASSESSMENT:
"""
    return prompt

def format_generated_report(report):
    """Format the generated report to ensure proper structure"""
    # Ensure the report has basic sections
    required_sections = [
        "CLINICAL ASSESSMENT:",
        "RECOMMENDATIONS:",
        "IMMEDIATE ACTIONS:",
        "FOLLOW-UP:"
    ]
    
    formatted_report = report
    
    # Add missing sections if needed
    for section in required_sections:
        if section not in formatted_report:
            formatted_report += f"\n\n{section}\n[Please consult a veterinarian for detailed guidance]"
    
    return formatted_report

def generate_fallback_report(prediction_result, animal_info, symptoms):
    """Generate a fallback report when AI generation fails"""
    
    risk_level = "HIGH - EMERGENCY" if prediction_result['ensemble']['dangerous'] else "LOW - MONITOR"
    confidence = prediction_result['ensemble']['confidence']
    
    report = f"""
===== VETERINARY MEDICAL REPORT =====

CLINICAL ASSESSMENT:
Based on the presented symptoms and patient information, this case has been assessed as {risk_level}. 
The system indicates {confidence} confidence in this assessment.

PRESENTING SYMPTOMS:
{', '.join(symptoms)}

IMMEDIATE ACTIONS REQUIRED:
{"**🚨 URGENT VETERINARY ATTENTION REQUIRED** - Contact emergency veterinarian immediately" if prediction_result['ensemble']['dangerous'] else "Continue monitoring and schedule veterinary consultation"}

RECOMMENDATIONS:
- Monitor vital signs closely
- Ensure access to fresh water
- Keep animal in comfortable, quiet environment
- Contact veterinarian if condition changes
- Follow professional veterinary advice

WHAT TO AVOID:
- Do not administer medications without veterinary guidance
- Do not attempt home treatments for serious symptoms
- Avoid stressing the animal unnecessarily

DIAGNOSTIC SUGGESTIONS:
- Complete physical examination by licensed veterinarian
- Blood work and laboratory tests as indicated
- Diagnostic imaging if warranted

FOLLOW-UP INSTRUCTIONS:
- Schedule veterinary appointment
- Monitor for any changes in condition
- Maintain communication with veterinary team

EMERGENCY PROTOCOL:
- Contact emergency veterinary services for worsening condition
- Have local emergency veterinary clinic information readily available

=== IMPORTANT DISCLAIMER ===
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
        return False, f"❌ Local AI model error: {str(e)}"s