# utils/helpers.py
import os
import json
from datetime import datetime
import google.generativeai as genai
from src.config import Config
import streamlit as st

def save_api_key(api_key):
    """Save API key to .env file"""
    try:
        env_vars = {}
        
        # Read existing .env file if it exists
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value
        
        # Update the Gemini API key
        env_vars['GEMINI_API_KEY'] = api_key
        
        # Write back to .env file
        with open('.env', 'w') as f:
            for key, value in env_vars.items():
                f.write(f'{key}={value}\n')
        
        # Update config immediately
        Config.GEMINI_API_KEY = api_key
            
        return True
    except Exception as e:
        st.error(f"Error saving API key: {e}")
        return False

def generate_vet_report(prediction_result, animal_info, symptoms):
    """Generate a veterinary report using Gemini AI"""
    if not Config.GEMINI_API_KEY:
        return "API key not configured. Please set up Gemini API key in the sidebar settings."
    
    try:
        # Configure Gemini
        genai.configure(api_key=Config.GEMINI_API_KEY)
        
        # Use the correct model name
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Generate a comprehensive veterinary medical report and action plan based on the following assessment:

        ANIMAL INFORMATION:
        - Species: {animal_info['animal']}
        - Breed: {animal_info['breed']}
        - Age: {animal_info['age']}
        - Weight: {animal_info['weight']} kg

        PRESENTING SYMPTOMS: {', '.join(symptoms)}

        RISK ASSESSMENT:
        - Overall Risk Level: {'HIGH - EMERGENCY' if prediction_result['ensemble']['dangerous'] else 'LOW - MONITOR'}
        - Confidence: {prediction_result['ensemble']['confidence']}
        - Model Agreement: {prediction_result['ensemble']['model_agreement']}
        - Risk Probability: {prediction_result['ensemble']['probability']:.1%}

        Please provide a detailed veterinary report structured as follows:

        ======== VETERINARY ACTION PLAN ========

        🚨 IMMEDIATE ACTIONS REQUIRED:
        [List critical immediate steps based on risk level]

        ✅ WHAT TO DO:
        [Detailed list of recommended actions, treatments, and monitoring]

        ❌ WHAT NOT TO DO:
        [Specific actions to avoid that could worsen the condition]

        🔍 CLINICAL ASSESSMENT:
        [Professional evaluation of the clinical presentation]

        📋 DIFFERENTIAL DIAGNOSES:
        [List of potential conditions in order of likelihood]

        🧪 RECOMMENDED DIAGNOSTIC TESTS:
        [Specific tests and examinations recommended]

        💊 TREATMENT RECOMMENDATIONS:
        [Detailed treatment plan and medications if applicable]

        📈 PROGNOSIS & MONITORING:
        [Expected outcome and monitoring schedule]

        🏠 HOME CARE INSTRUCTIONS:
        [Detailed instructions for home care and observation]

        Format this as a clear, actionable plan that a pet owner can follow immediately.
        Use clear sections with emojis for better readability.
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
    if not symptoms or not predictor.loaded:
        return ""
    
    try:
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
    except Exception as e:
        return f"Error analyzing symptoms: {str(e)}"