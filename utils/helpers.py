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
    """Generate a comprehensive veterinary report using Gemini AI"""
    if not Config.GEMINI_API_KEY:
        return "API key not configured. Please set up Gemini API key in the sidebar settings."
    
    try:
        # Configure Gemini
        genai.configure(api_key=Config.GEMINI_API_KEY)
        
        # Create the model
        model = genai.GenerativeModel('gemini-pro')
        
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
        - Confidence: {prediction_result['ensemble']['confidence']}
        - Model Agreement: {prediction_result['ensemble']['model_agreement']}
        - Risk Probability: {prediction_result['ensemble']['probability']:.1%}

        Please provide a DETAILED VETERINARY REPORT with clear, actionable advice in these two main sections:

        ## 🟢 WHAT TO DO - IMMEDIATE ACTIONS

        Provide specific, step-by-step instructions for:
        - Immediate first aid measures
        - Home care and monitoring
        - When to contact a veterinarian
        - Emergency procedures if needed
        - Comfort measures for the animal

        ## 🔴 WHAT NOT TO DO - IMPORTANT PRECAUTIONS

        List critical things to avoid:
        - Common dangerous mistakes
        - Unsafe home remedies
        - Medications to avoid
        - Foods or substances that could harm
        - Activities that could worsen the condition

        Format this as a clear, professional veterinary medical report that a pet owner can easily understand and follow.
        Use bullet points for clarity and emphasize safety above all.
        """
        
        # Generate content with Gemini
        response = model.generate_content(prompt)
        
        report = response.text
        
        # Add header with timestamp and case information
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""
VETERINARY HEALTH ASSESSMENT REPORT
Generated: {timestamp}
Case ID: {animal_info['animal']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}

PATIENT SUMMARY:
- Animal: {animal_info['animal'].title()}
- Breed: {animal_info['breed'].title()}
- Age: {animal_info['age'].title()}
- Weight: {animal_info['weight']} kg
- Symptoms: {', '.join(symptoms).title()}
- Risk Level: {'HIGH - EMERGENCY' if prediction_result['ensemble']['dangerous'] else 'LOW - MONITOR'}
- Assessment Confidence: {prediction_result['ensemble']['confidence']}

{'='*80}

"""
        
        full_report = header + report
        
        # Save report to file
        report_filename = f"reports/generated_reports/vet_report_{animal_info['animal']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        os.makedirs(os.path.dirname(report_filename), exist_ok=True)
        
        with open(report_filename, 'w') as f:
            f.write(full_report)
        
        return full_report
        
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
        medium_risk_count = 0
        
        for symptom in symptoms:
            severity = predictor.data_loader.symptom_severity_weights.get(symptom, 0.1)
            total_severity += severity
            
            if severity > 0.7:
                risk_level = "🔴 HIGH RISK"
                high_risk_count += 1
            elif severity > 0.4:
                risk_level = "🟡 MEDIUM RISK"
                medium_risk_count += 1
            else:
                risk_level = "🟢 LOW RISK"
            
            analysis += f"- **{symptom.title().replace('_', ' ')}**: {risk_level} (Severity: {severity:.2f})\n"
        
        # Summary
        analysis += f"\n**Summary:**\n"
        analysis += f"- Total Symptoms: {len(symptoms)}\n"
        if high_risk_count > 0:
            analysis += f"- 🔴 High-Risk Symptoms: {high_risk_count}\n"
        if medium_risk_count > 0:
            analysis += f"- 🟡 Medium-Risk Symptoms: {medium_risk_count}\n"
        
        if high_risk_count > 0:
            analysis += f"\n⚠️ **Critical Warning**: {high_risk_count} high-risk symptom(s) detected! Immediate veterinary attention recommended.\n"
        
        return analysis
    except Exception as e:
        return f"Error analyzing symptoms: {str(e)}"