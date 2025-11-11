import os
import json
from datetime import datetime
import google.generativeai as genai
from src.config import Config

def save_api_key(api_key, api_type='gemini'):
    """Save API key to .env file"""
    with open('.env', 'w') as f:
        if api_type == 'gemini':
            f.write(f'GEMINI_API_KEY={api_key}\n')
    
    # Update config
    if api_type == 'gemini':
        Config.GEMINI_API_KEY = api_key

def generate_vet_report(prediction_result, animal_info, symptoms):
    """Generate a veterinary report using Gemini AI"""
    if not Config.GEMINI_API_KEY:
        return "API key not configured. Please set up Gemini API key in settings."
    
    try:
        # Configure Gemini with correct model name
        genai.configure(api_key=Config.GEMINI_API_KEY)
        
        # Use the free model - gemini-1.5-flash (free tier)
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
        except:
            # Fallback to other free models
            try:
                model = genai.GenerativeModel('gemini-1.0-pro')
            except:
                return "Error: No available Gemini model found. Please check your API key and model availability."
        
        prompt = f"""
        Generate a comprehensive veterinary medical report and action plan based on the following assessment:

        PATIENT INFORMATION:
        - Species: {animal_info['animal']}
        - Breed: {animal_info['breed']}
        - Age: {animal_info['age']}
        - Weight: {animal_info['weight']} kg

        PRESENTING SYMPTOMS: {', '.join(symptoms)}

        RISK ASSESSMENT:
        - Overall Risk Level: {'HIGH - EMERGENCY' if prediction_result['ensemble']['dangerous'] else 'LOW - MONITOR'}
        - Confidence Level: {prediction_result['ensemble']['confidence']}
        - Risk Probability: {prediction_result['ensemble']['probability']:.1%}

        Please provide a detailed veterinary report in the following structure:

        ===== VETERINARY MEDICAL REPORT =====

        CLINICAL ASSESSMENT:
        [Provide overall clinical assessment based on symptoms and patient information]

        DIFFERENTIAL DIAGNOSES:
        [List possible conditions ranked by likelihood]

        IMMEDIATE ACTIONS REQUIRED:
        [Detailed step-by-step instructions on what to do immediately]

        WHAT TO DO:
        - [Specific actionable instructions]
        - [Monitoring guidelines]
        - [Home care instructions]
        - [When to seek emergency care]

        WHAT NOT TO DO:
        - [Specific warnings and contraindications]
        - [Medications to avoid]
        - [Actions that could worsen condition]

        DIAGNOSTIC RECOMMENDATIONS:
        [Recommended tests and examinations]

        TREATMENT CONSIDERATIONS:
        [Potential treatment options]

        PROGNOSIS:
        [Expected outcome and recovery timeline]

        FOLLOW-UP INSTRUCTIONS:
        [Monitoring schedule and when to recheck]

        EMERGENCY CONTACT PROTOCOL:
        [When and how to contact emergency services]

        Format this professionally as if from a licensed veterinary physician. Use clear sections and bullet points for readability.
        Keep the response concise but comprehensive, focusing on actionable advice.
        """

        response = model.generate_content(prompt)
        report = response.text
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"reports/generated_reports/vet_report_{timestamp}.txt"
        
        os.makedirs(os.path.dirname(report_filename), exist_ok=True)
        
        with open(report_filename, 'w') as f:
            f.write(report)
        
        return report
        
    except Exception as e:
        return f"Error generating report: {str(e)}"

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

def test_gemini_connection(api_key):
    """Test if Gemini API is working"""
    try:
        genai.configure(api_key=api_key)
        # Try the free model first
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("Say 'Connection successful' in one word.")
            return True, "✅ Gemini AI connection successful! (Using gemini-1.5-flash)"
        except:
            # Fallback to other model
            model = genai.GenerativeModel('gemini-1.0-pro')
            response = model.generate_content("Say 'Connection successful' in one word.")
            return True, "✅ Gemini AI connection successful! (Using gemini-1.0-pro)"
    except Exception as e:
        return False, f"❌ Gemini AI connection failed: {str(e)}"