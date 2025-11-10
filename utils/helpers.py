# utils/helpers.py
import os
import json
from datetime import datetime
import openai
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
        
        # Update the OpenAI API key
        env_vars['OPENAI_API_KEY'] = api_key
        
        # Write back to .env file
        with open('.env', 'w') as f:
            for key, value in env_vars.items():
                f.write(f'{key}={value}\n')
        
        # Update config immediately
        Config.OPENAI_API_KEY = api_key
            
        return True
    except Exception as e:
        st.error(f"Error saving API key: {e}")
        return False

def generate_vet_report(prediction_result, animal_info, symptoms):
    """Generate a veterinary report using AI"""
    if not Config.OPENAI_API_KEY:
        return "API key not configured. Please set up OpenAI API key in the sidebar settings."
    
    try:
        client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        
        prompt = f"""
        Generate a professional veterinary medical report based on the following assessment:
        
        Animal Information:
        - Species: {animal_info['animal']}
        - Breed: {animal_info['breed']}
        - Age: {animal_info['age']}
        - Weight: {animal_info['weight']} kg
        
        Presenting Symptoms: {', '.join(symptoms)}
        
        Risk Assessment:
        - Overall Risk Level: {'HIGH - Emergency' if prediction_result['ensemble']['dangerous'] else 'LOW - Monitor'}
        - Confidence: {prediction_result['ensemble']['confidence']}
        - Model Agreement: {prediction_result['ensemble']['model_agreement']}
        
        Please provide a structured veterinary report including:
        1. Clinical Assessment
        2. Differential Diagnoses
        3. Recommended Diagnostic Tests
        4. Treatment Recommendations
        5. Prognosis
        6. Client Instructions
        
        Format the report professionally as if from a licensed veterinarian.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a experienced veterinary physician. Provide detailed, professional medical reports."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        report = response.choices[0].message.content
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"reports/generated_reports/vet_report_{timestamp}.txt"
        
        os.makedirs(os.path.dirname(report_filename), exist_ok=True)
        
        with open(report_filename, 'w') as f:
            f.write(report)
        
        return report
        
    except openai.AuthenticationError:
        return "Error: Invalid API key. Please check your OpenAI API key in the settings."
    except openai.APIConnectionError:
        return "Error: Unable to connect to OpenAI API. Please check your internet connection."
    except openai.RateLimitError:
        return "Error: API rate limit exceeded. Please try again later."
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