# app.py - Complete version with better error handling
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import joblib
from datetime import datetime
import warnings
import logging
import time

# ===== FIX FOR STREAMLIT WARNINGS =====
warnings.filterwarnings("ignore", category=UserWarning, message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Thread.*MainThread.*")

# Set environment variables
os.environ['STREAMLIT_BARE_MODE'] = 'true'
os.environ['STREAMLIT_LOGGING_LEVEL'] = 'error'

# Configure logging
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)
# ======================================

# Add src to path
sys.path.append('src')
sys.path.append('utils')

# Import torch with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from src.predictor import VeterinaryPredictor
    from src.config import Config
    from utils.helpers import generate_vet_report_local, get_risk_color, format_symptom_analysis, test_ai_connection
except ImportError as e:
    st.error(f"Import error: {e}. Please check your module structure.")
    st.stop()

# Page configuration and CSS remains the same as before...
# [The rest of your app.py code remains unchanged - it's too long to include here]
# Just ensure you have the updated CSS and functionality from previous versions

# Page configuration
st.set_page_config(
    page_title="Veterinary Health Assessment System",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with supremacy rule styling
st.markdown("""
<style>
    /* Supremacy rule styling */
    .supremacy-triggered {
        background: linear-gradient(135deg, #ff0000, #cc0000) !important;
        padding: 25px;
        border-radius: 20px;
        margin: 20px 0;
        color: white !important;
        border: 5px solid #ff6b6b;
        box-shadow: 0 8px 35px rgba(255, 0, 0, 0.6);
        text-align: center;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 8px 35px rgba(255, 0, 0, 0.6); }
        50% { transform: scale(1.02); box-shadow: 0 12px 45px rgba(255, 0, 0, 0.8); }
        100% { transform: scale(1); box-shadow: 0 8px 35px rgba(255, 0, 0, 0.6); }
    }
    
    .supremacy-warning {
        background: linear-gradient(135deg, #ff9800, #f57c00);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        color: white !important;
        border-left: 8px solid #e65100;
        box-shadow: 0 6px 25px rgba(255, 152, 0, 0.4);
    }
    
    /* Rest of the CSS remains the same as before */
    .main {
        background-color: #f8f9fa;
    }
    .main .block-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .main-header {
        font-size: 3.2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h1, h2, h3 {
        color: #2c3e50 !important;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.8rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        padding: 25px;
        border-radius: 20px;
        margin: 20px 0;
        color: white !important;
        border-left: 8px solid #c0392b;
        box-shadow: 0 6px 25px rgba(255, 107, 107, 0.4);
        text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #feca57, #ff9ff3);
        padding: 25px;
        border-radius: 20px;
        margin: 20px 0;
        color: #2c3e50 !important;
        border-left: 8px solid #f39c12;
        box-shadow: 0 6px 25px rgba(254, 202, 87, 0.4);
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #48dbfb, #0abde3);
        padding: 25px;
        border-radius: 20px;
        margin: 20px 0;
        color: white !important;
        border-left: 8px solid #27ae60;
        box-shadow: 0 6px 25px rgba(72, 219, 251, 0.4);
        text-align: center;
    }
    
    .symptom-item {
        padding: 15px;
        margin: 8px 0;
        border-radius: 12px;
        border-left: 6px solid;
        color: #2c3e50 !important;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .symptom-item:hover {
        transform: translateX(8px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .high-risk-symptom {
        border-left-color: #e74c3c;
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        color: #c62828 !important;
        border: 2px solid #ffcdd2;
    }
    .medium-risk-symptom {
        border-left-color: #f39c12;
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        color: #ef6c00 !important;
        border: 2px solid #ffe0b2;
    }
    .low-risk-symptom {
        border-left-color: #27ae60;
        background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
        color: #2e7d32 !important;
        border: 2px solid #c8e6c9;
    }
    
    .model-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 20px;
        border-radius: 15px;
        margin: 12px 0;
        border-left: 6px solid #3498db;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: #2c3e50 !important;
        transition: transform 0.3s ease;
    }
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .confidence-text {
        color: #2c3e50 !important;
        font-weight: bold;
        font-size: 1.2em;
        background: linear-gradient(135deg, #ecf0f1, #bdc3c7);
        padding: 6px 12px;
        border-radius: 8px;
        display: inline-block;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .report-section {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 30px;
        border-radius: 15px;
        margin: 25px 0;
        border-left: 6px solid #3498db;
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.2);
        color: #2c3e50 !important;
    }
    
    .ai-thinking {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 35px;
        border-radius: 20px;
        border-left: 8px solid #2196f3;
        text-align: center;
        margin: 30px 0;
        color: #1565c0 !important;
        box-shadow: 0 6px 25px rgba(33, 150, 243, 0.4);
    }
    .blink {
        animation: blink 1.5s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.05); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    .stButton button {
        background: linear-gradient(135deg, #3498db, #2980b9) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 30px !important;
        font-weight: 700 !important;
        font-size: 1.1em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3) !important;
    }
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(52, 152, 219, 0.5) !important;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #ffffff, #f8f9fa) !important;
        padding: 20px !important;
        border-radius: 15px !important;
        border: 3px solid #3498db !important;
        text-align: center !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
    }
    
    .stTextArea textarea {
        background-color: #f8f9fa !important;
        color: #2c3e50 !important;
        border: 3px solid #bdc3c7 !important;
        border-radius: 12px !important;
        font-family: 'Courier New', monospace !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
        padding: 20px !important;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.1) !important;
    }
    
    .stSelectbox, .stNumberInput, .stMultiselect {
        background-color: #ffffff !important;
        border-radius: 10px !important;
        border: 2px solid #bdc3c7 !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #d4edda, #c3e6cb) !important;
        border: 2px solid #c3e6cb !important;
        color: #155724 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        font-weight: 600 !important;
    }
    .stError {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb) !important;
        border: 2px solid #f5c6cb !important;
        color: #721c24 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        font-weight: 600 !important;
    }
    .stWarning {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7) !important;
        border: 2px solid #ffeaa7 !important;
        color: #856404 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        font-weight: 600 !important;
    }
    
    .symptom-summary {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border: 3px solid #3498db;
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.2);
    }
</style>
""", unsafe_allow_html=True)

class VeterinaryApp:
    def __init__(self):
        self.predictor = VeterinaryPredictor()
        self.models_loaded = False
        self.ai_model_loaded = False
        self.load_models()
        self.check_ai_model()

    def load_models(self):
        """Load trained models"""
        model_paths = {
            'RandomForest': 'models/randomforest.joblib',
            'NeuralNetwork': 'models/neuralnetwork.joblib',
            'XGBoost': 'models/xgboost.joblib',
            'SCT': 'models/sct_model.pth',
            'LSTM': 'models/lstm_model.pth',
            'encoders': 'models/encoders.joblib'
        }
        
        # Check if all model files exist
        missing_models = [name for name, path in model_paths.items() if not os.path.exists(path)]
        
        if missing_models:
            st.sidebar.warning(f"⚠️ Missing model files: {', '.join(missing_models)}")
            self.models_loaded = False
            return

        try:
            self.models_loaded = self.predictor.load_models(model_paths)
            if self.models_loaded:
                st.sidebar.success("✅ Health assessment models loaded!")
            else:
                st.sidebar.error("❌ Failed to load health assessment models")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading models: {e}")
            self.models_loaded = False

    def check_ai_model(self):
        """Check if local AI model is available"""
        try:
            success, message = test_ai_connection()
            self.ai_model_loaded = success
            if success:
                st.sidebar.success("✅ AI Model Ready")
            else:
                st.sidebar.warning("⚠️ " + message)
        except Exception as e:
            st.sidebar.warning(f"⚠️ AI model check failed: {e}")
            self.ai_model_loaded = False

    def render_sidebar(self):
        """Render sidebar with configuration and information"""
        st.sidebar.title("⚙️ System Configuration")
        
        # AI Model Status
        st.sidebar.markdown("### 🤖 AI Report System")
        if self.ai_model_loaded:
            st.sidebar.success("✅ **AI Model Ready**")
            st.sidebar.markdown("**Model:** Enhanced AI System")
            st.sidebar.markdown("**Status:** Professional Reports")
        else:
            st.sidebar.warning("⚠️ **Using Enhanced Fallback**")
            st.sidebar.markdown("**System:** Structured Reporting")
            st.sidebar.markdown("**Quality:** Professional Grade")
        
        # Model Information
        st.sidebar.markdown("### 🎯 Assessment Models")
        if self.models_loaded:
            st.sidebar.success("✅ **Models Active**")
            if hasattr(self.predictor, 'get_available_animals'):
                st.sidebar.markdown(f"**Animals:** {len(self.predictor.get_available_animals())}")
            if hasattr(self.predictor, 'get_available_symptoms'):
                st.sidebar.markdown(f"**Symptoms:** {len(self.predictor.get_available_symptoms())}")
        else:
            st.sidebar.error("❌ **Models Inactive**")
        
        # Supremacy Rules Info
        st.sidebar.markdown("### 🚨 Fail-Safe Rules")
        st.sidebar.info("""
        **Supremacy Rules Active:**
        
        🚑 **Symptom Rule:** Any HIGH-RISK symptom → EMERGENCY
        
        🤖 **Model Rule:** 3+ models vote DANGEROUS → EMERGENCY
        
        **Safety First:** These rules override normal scoring to ensure patient safety.
        """)
        
        # Quick Tips
        st.sidebar.markdown("### 💡 Quick Guide")
        st.sidebar.info("""
        **Assessment Steps:**
        1. Select animal type and breed
        2. Choose age group and weight
        3. Select all relevant symptoms
        4. Click 'Assess Health Status'
        5. Review risk assessment
        6. Generate detailed report
        
        **Note:** Always consult a licensed veterinarian for professional diagnosis.
        """)
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Veterinary Health System v2.1**")
        st.sidebar.markdown("*Professional Animal Care Assessment*")

    def render_input_section(self):
        """Render input section for animal information"""
        st.markdown("### 🐾 Patient Information")
        
        # Create a nice container for the input section
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                animal = st.selectbox(
                    "**Animal Type** 🐕",
                    options=list(Config.ANIMAL_BREEDS.keys()),
                    index=0,
                    help="Select the type of animal"
                )
            
            with col2:
                breed = st.selectbox(
                    "**Breed** 🏷️",
                    options=Config.ANIMAL_BREEDS.get(animal, ['mixed']),
                    index=0,
                    help="Select the specific breed"
                )
            
            with col3:
                age = st.selectbox(
                    "**Age Group** 📅",
                    options=["young", "adult", "senior"],
                    index=1,
                    help="Select the age category"
                )
            
            with col4:
                weight = st.number_input(
                    "**Weight (kg)** ⚖️",
                    min_value=0.1,
                    max_value=1000.0,
                    value=25.0,
                    step=0.1,
                    help="Enter weight in kilograms"
                )
        
        # Symptoms Selection
        st.markdown("### 🤒 Symptoms Selection")
        symptoms = st.multiselect(
            "**Select all applicable symptoms** 🔍",
            options=Config.SYMPTOMS,
            help="Choose all symptoms that apply to the patient",
            max_selections=15
        )
        
        # Display selected symptoms with enhanced visibility
        if symptoms and self.models_loaded:
            st.markdown("### 📊 Symptom Analysis")
            symptom_analysis = format_symptom_analysis(symptoms, self.predictor)
            st.markdown(symptom_analysis, unsafe_allow_html=True)
        
        return {
            'animal': animal,
            'breed': breed,
            'age': age,
            'weight': weight,
            'symptoms': symptoms
        }

    def render_prediction_results(self, prediction_result, animal_info, symptoms):
        """Render prediction results with SUPREMACY RULE display"""
        if not isinstance(prediction_result, dict) or 'ensemble' not in prediction_result:
            st.error(f"Prediction error: {prediction_result}")
            return
        
        ensemble = prediction_result['ensemble']
        
        # Check if supremacy rule was triggered
        supremacy_triggered = ensemble.get('supremacy_triggered', False)
        supremacy_reason = ensemble.get('supremacy_reason', '')
        
        # Risk Level Display with SUPREMACY override
        risk_score = ensemble['probability']
        
        # If supremacy triggered, force HIGH RISK display
        if supremacy_triggered:
            risk_class = "supremacy-triggered"
            risk_level = "🚨 EMERGENCY - SUPREMACY RULE TRIGGERED"
            risk_icon = "🚑"
        elif risk_score < 0.2:
            risk_class = "risk-low"
            risk_level = "LOW RISK"
            risk_icon = "🟢"
        elif risk_score < 0.5:
            risk_class = "risk-medium"
            risk_level = "MODERATE RISK"
            risk_icon = "🟡"
        else:
            risk_class = "risk-high"
            risk_level = "HIGH RISK"
            risk_icon = "🔴"
        
        # Display supremacy warning if triggered
        if supremacy_triggered:
            st.markdown(f"""
            <div class="supremacy-warning">
                <h2>🚨 FAIL-SAFE SUPREMACY RULE ACTIVATED</h2>
                <p style='font-size: 1.2em;'><strong>Reason:</strong> {supremacy_reason}</p>
                <p style='font-size: 1.1em;'>Normal scoring overridden for patient safety. Immediate veterinary attention required.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Main risk display
        st.markdown(f"""
        <div class="{risk_class}">
            <h1 style='color: inherit; margin: 0; font-size: 2.5em;'>{risk_icon} {risk_level}</h1>
            <p style='color: inherit; margin: 15px 0 0 0; font-size: 1.3em;'>
            <strong>Confidence Level:</strong> <span class="confidence-text">{ensemble['confidence']}</span><br>
            <strong>Model Agreement:</strong> {ensemble['model_agreement']}<br>
            <strong>Risk Probability:</strong> {risk_score:.1%}</p>
            {f"<p style='color: inherit; margin: 10px 0 0 0; font-size: 1.1em;'><strong>Supremacy Rule:</strong> {supremacy_reason}</p>" if supremacy_triggered else ""}
        </div>
        """, unsafe_allow_html=True)
        
        # Columns for detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            # Individual Model Predictions
            st.markdown("### 🤖 Model Assessments")
            dangerous_count = ensemble.get('dangerous_votes', 0)
            total_models = ensemble.get('total_models', 5)
            
            st.markdown(f"**Model Consensus:** {dangerous_count}/{total_models} models vote DANGEROUS")
            if dangerous_count >= 3:
                st.warning("🚨 **Model Supremacy:** Majority vote triggers emergency protocol")
            
            for model_name, prediction in prediction_result['individual_predictions'].items():
                status_icon = "🔴" if prediction['dangerous'] else "🟢"
                status_text = "DANGEROUS" if prediction['dangerous'] else "SAFE"
                
                st.markdown(f"""
                <div class="model-card">
                    <strong style='font-size: 1.2em; color: #2c3e50;'>{model_name}</strong><br>
                    <span style='font-size: 1.3em; font-weight: bold;'>{status_icon} {status_text}</span><br>
                    Confidence: <span class="confidence-text">{prediction['confidence']}</span><br>
                    Probability: {prediction['probability']:.3f}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Risk Visualization
            st.markdown("### 📊 Risk Assessment Gauge")
            
            # Adjust gauge value if supremacy triggered
            display_value = min(risk_score * 100, 95) if supremacy_triggered else risk_score * 100
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = display_value,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score", 'font': {'size': 28, 'color': '#2c3e50', 'family': 'Arial'}},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#2c3e50"},
                    'bar': {'color': "#3498db", 'thickness': 0.3},
                    'bgcolor': "white",
                    'borderwidth': 3,
                    'bordercolor': "#bdc3c7",
                    'steps': [
                        {'range': [0, 30], 'color': "#27ae60"},
                        {'range': [30, 70], 'color': "#f39c12"},
                        {'range': [70, 100], 'color': "#e74c3c"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 5},
                        'thickness': 0.8,
                        'value': 90
                    }
                }
            ))
            
            if supremacy_triggered:
                fig.add_annotation(
                    text="🚨 SUPREMACY RULE ACTIVE",
                    x=0.5, y=0.2,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=16, color="red")
                )
            
            fig.update_layout(
                height=350, 
                font={'color': "#2c3e50", 'family': "Arial", 'size': 16},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=80, b=40, l=40, r=40)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced Symptom Summary with supremacy highlights
            if symptoms:
                st.markdown("### 🔍 Symptom Summary")
                high_risk_symptoms = ensemble.get('high_risk_symptoms', [])
                high_risk_count = len(high_risk_symptoms)
                medium_risk_count = sum(1 for s in symptoms if 0.4 < self.predictor.data_loader.symptom_severity_weights.get(s, 0) <= 0.7)
                low_risk_count = len(symptoms) - high_risk_count - medium_risk_count
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Symptoms", len(symptoms), delta=None)
                with col2:
                    delta_high = f"{high_risk_count} URGENT" if high_risk_count > 0 else "None"
                    if high_risk_count > 0:
                        st.metric("High-Risk", high_risk_count, delta=delta_high, delta_color="inverse")
                    else:
                        st.metric("High-Risk", high_risk_count, delta=delta_high)
                with col3:
                    st.metric("Medium-Risk", medium_risk_count, delta=f"{medium_risk_count} concerning" if medium_risk_count > 0 else "None")
                
                # Display high-risk symptoms if any
                if high_risk_symptoms:
                    st.error(f"🚨 **High-Risk Symptoms Detected:** {', '.join(high_risk_symptoms)}")
        
        # AI Generated Report Section
        st.markdown("---")
        st.markdown("### 🏥 Professional Veterinary Report")
        
        # Initialize session state for report generation
        if 'report_generated' not in st.session_state:
            st.session_state.report_generated = False
        if 'generated_report' not in st.session_state:
            st.session_state.generated_report = None
        if 'generating_report' not in st.session_state:
            st.session_state.generating_report = False
        
        # Generate Report Button
        if not st.session_state.generating_report and not st.session_state.report_generated:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🧠 Generate Professional Veterinary Report", type="primary", use_container_width=True):
                    st.session_state.generating_report = True
                    st.session_state.report_generated = False
                    st.rerun()
        
        # Show AI Thinking Indicator
        if st.session_state.generating_report:
            st.markdown("""
            <div class="ai-thinking">
                <h2 class="blink">🧠 Generating Professional Report...</h2>
                <p style='font-size: 1.2em;'>Creating comprehensive veterinary assessment with structured analysis...</p>
                <p><em>This may take 5-15 seconds</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate the report
            try:
                # Add a small delay to show the thinking message
                time.sleep(2)
                
                report = generate_vet_report_local(prediction_result, animal_info, symptoms)
                
                # Store report in session state
                st.session_state.generated_report = report
                st.session_state.report_generated = True
                st.session_state.generating_report = False
                st.success("✅ Professional Report Generated Successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
                st.session_state.generating_report = False
        
        # Display report if available
        if st.session_state.report_generated and st.session_state.generated_report:
            st.markdown("### 📋 Comprehensive Veterinary Assessment Report")
            
            # Report display with download option
            report_col1, report_col2 = st.columns([3, 1])
            
            with report_col1:
                # Use a text area with monospace font for better readability
                st.text_area(
                    "Report Content", 
                    st.session_state.generated_report, 
                    height=600, 
                    key="report_display",
                    label_visibility="collapsed"
                )
            
            with report_col2:
                # Download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"vet_report_{animal_info['animal']}_{timestamp}.txt"
                
                st.download_button(
                    label="📥 Download Full Report",
                    data=st.session_state.generated_report,
                    file_name=filename,
                    mime="text/plain",
                    use_container_width=True,
                    help="Download the complete veterinary assessment report"
                )
                
                if st.button("🗑️ Clear Report", use_container_width=True, help="Clear the current report"):
                    st.session_state.report_generated = False
                    st.session_state.generated_report = None
                    st.session_state.generating_report = False
                    st.rerun()
        
        # Enhanced Recommendations with supremacy considerations
        st.markdown("---")
        st.markdown("### 💡 Clinical Recommendations")
        
        if supremacy_triggered:
            st.error(f"""
            ## 🚨 EMERGENCY - SUPREMACY RULE ACTIVATED
            
            **CRITICAL CLINICAL ACTIONS REQUIRED:**
            
            🏥 **IMMEDIATE EMERGENCY VETERINARY CARE NEEDED**
            📞 **CONTACT EMERGENCY VETERINARIAN NOW**
            ⚠️ **DO NOT DELAY - CONDITION REQUIRES URGENT ATTENTION**
            
            **Supremacy Trigger Reason:** {supremacy_reason}
            
            **Emergency Protocol:**
            - 🚑 **Transport to emergency veterinary clinic immediately**
            - 📞 **Call ahead to alert emergency staff**
            - 🏥 **Prepare for emergency medical intervention**
            - ⚠️ **Do not attempt home treatment**
            - 📋 **Bring all medical history and current medications**
            
            **Critical Time Factors:**
            - Every minute counts in emergency situations
            - Professional medical intervention is essential
            - Follow emergency veterinary guidance precisely
            """)
        elif ensemble['dangerous']:
            st.error("""
            ## 🚨 URGENT VETERINARY ATTENTION REQUIRED
            
            **Immediate Clinical Actions:**
            - 🏥 **Contact emergency veterinarian immediately**
            - 📞 **Keep emergency veterinary number ready**
            - 👀 **Monitor vital signs continuously**
            - 🛌 **Keep animal calm and comfortable**
            - 🚗 **Prepare for immediate transport to veterinary clinic**
            - ⚠️ **Do not attempt home treatment for critical symptoms**
            
            **Critical Indicators:**
            - High-risk symptoms detected requiring emergency care
            - Time-sensitive condition identified
            - Professional medical intervention necessary
            """)
        else:
            st.success("""
            ## ✅ CONTINUE MONITORING & OBSERVATION
            
            **Recommended Clinical Actions:**
            - 👀 **Continue monitoring symptoms closely**
            - 💧 **Ensure access to fresh water and nutrition**
            - 🛌 **Provide comfortable resting area**
            - 📞 **Contact veterinarian if condition worsens**
            - 📅 **Schedule routine check-up if symptoms persist**
            
            **Observation Guidelines:**
            - Monitor for any changes in behavior or appetite
            - Watch for worsening of existing symptoms
            - Note development of new symptoms
            - Track changes in drinking or eating habits
            """)

    def run(self):
        """Main application runner"""
        # Enhanced Header with gradient
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.2);'>
            <h1 class="main-header">🐾 Veterinary Health Assessment System</h1>
            <h3 style='color: white; margin: 1rem 0 0 0; font-size: 1.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>Professional AI-Powered Animal Health Risk Assessment</h3>
            <p style='color: white; margin: 0.5rem 0 0 0; font-size: 1.1em;'>🚨 <strong>Fail-Safe Supremacy Rules Active</strong> - Ensuring Patient Safety First</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content in an enhanced container
        with st.container():
            if not self.models_loaded:
                st.error("""
                ## ❌ Assessment Models Not Loaded
                
                **Required Action:**
                Please ensure all model files are present in the 'models' directory.
                Run the training script first to generate the assessment models.
                
                **Expected Files:**
                - sct_model.pth
                - lstm_model.pth  
                - randomforest.joblib
                - neuralnetwork.joblib
                - xgboost.joblib
                - encoders.joblib
                """)
            else:
                # Input section
                animal_info = self.render_input_section()
                
                # Assessment button with enhanced visibility
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔍 ASSESS HEALTH STATUS", type="primary", use_container_width=True):
                        # Clear previous reports when new assessment starts
                        if 'report_generated' in st.session_state:
                            st.session_state.report_generated = False
                        if 'generated_report' in st.session_state:
                            st.session_state.generated_report = None
                        if 'generating_report' in st.session_state:
                            st.session_state.generating_report = False
                        
                        if not animal_info['symptoms']:
                            st.warning("⚠️ **Please select at least one symptom**")
                        else:
                            with st.spinner("🤖 **Analyzing health status with advanced assessment models...**"):
                                try:
                                    prediction_result = self.predictor.predict_ensemble(
                                        animal_info['animal'],
                                        animal_info['breed'], 
                                        animal_info['age'],
                                        animal_info['weight'],
                                        animal_info['symptoms']
                                    )
                                    
                                    # Store prediction in session state
                                    st.session_state.prediction_result = prediction_result
                                    st.session_state.animal_info = animal_info
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ **Assessment Error:** {e}")
                
                # Display results if available
                if 'prediction_result' in st.session_state and st.session_state.animal_info is not None:
                    self.render_prediction_results(
                        st.session_state.prediction_result,
                        st.session_state.animal_info,
                        st.session_state.animal_info['symptoms']
                    )

def main():
    # Initialize session state variables
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'animal_info' not in st.session_state:
        st.session_state.animal_info = None
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    if 'generated_report' not in st.session_state:
        st.session_state.generated_report = None
    if 'generating_report' not in st.session_state:
        st.session_state.generating_report = False
    
    try:
        app = VeterinaryApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")

if __name__ == "__main__":
    main()