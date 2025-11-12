# app.py - Enhanced with professional styling
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

# Page configuration
st.set_page_config(
    page_title="Veterinary Health Assessment System",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with improved color scheme
st.markdown("""
<style>
    /* Main background and text */
    .main {
        background-color: #f8f9fa;
    }
    .main .block-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Headers */
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Section headers */
    h1, h2, h3 {
        color: #2c3e50 !important;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Risk level cards */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        color: white !important;
        border-left: 6px solid #c0392b;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    .risk-medium {
        background: linear-gradient(135deg, #feca57, #ff9ff3);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        color: #2c3e50 !important;
        border-left: 6px solid #f39c12;
        box-shadow: 0 4px 15px rgba(254, 202, 87, 0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #48dbfb, #0abde3);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        color: white !important;
        border-left: 6px solid #27ae60;
        box-shadow: 0 4px 15px rgba(72, 219, 251, 0.3);
    }
    
    /* Model cards */
    .model-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 5px solid #3498db;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        color: #2c3e50 !important;
        transition: transform 0.2s ease;
    }
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Confidence text */
    .confidence-text {
        color: #2c3e50 !important;
        font-weight: bold;
        font-size: 1.1em;
        background: linear-gradient(135deg, #ecf0f1, #bdc3c7);
        padding: 4px 8px;
        border-radius: 6px;
        display: inline-block;
    }
    
    /* Symptom items */
    .symptom-item {
        padding: 12px;
        margin: 6px 0;
        border-radius: 8px;
        border-left: 5px solid;
        color: #2c3e50 !important;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .symptom-item:hover {
        transform: translateX(5px);
    }
    .high-risk-symptom {
        border-left-color: #e74c3c;
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        color: #c62828 !important;
    }
    .medium-risk-symptom {
        border-left-color: #f39c12;
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        color: #ef6c00 !important;
    }
    .low-risk-symptom {
        border-left-color: #27ae60;
        background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
        color: #2e7d32 !important;
    }
    
    /* Report sections */
    .report-section {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        border-left: 5px solid #3498db;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.2);
        color: #2c3e50 !important;
    }
    
    /* AI Thinking section */
    .ai-thinking {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 30px;
        border-radius: 15px;
        border-left: 6px solid #2196f3;
        text-align: center;
        margin: 25px 0;
        color: #1565c0 !important;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
    }
    .blink {
        animation: blink 1.5s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background: linear-gradient(135deg, #2c3e50, #34495e) !important;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #2c3e50, #34495e) !important;
    }
    
    /* Text area for reports */
    .stTextArea textarea {
        background-color: #f8f9fa !important;
        color: #2c3e50 !important;
        border: 2px solid #bdc3c7 !important;
        border-radius: 8px !important;
        font-family: 'Courier New', monospace !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #3498db, #2980b9) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4) !important;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #27ae60, #229954) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
    }
    
    /* Clear button */
    .stButton button:contains("Clear") {
        background: linear-gradient(135deg, #e74c3c, #c0392b) !important;
    }
    
    /* Input fields */
    .stSelectbox, .stNumberInput, .stMultiselect {
        background-color: #ffffff !important;
        border-radius: 8px !important;
    }
    
    /* Metrics */
    .stMetric {
        background: linear-gradient(135deg, #ffffff, #f8f9fa) !important;
        padding: 15px !important;
        border-radius: 10px !important;
        border: 2px solid #3498db !important;
        text-align: center !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ecf0f1;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda, #c3e6cb) !important;
        border: 1px solid #c3e6cb !important;
        color: #155724 !important;
        border-radius: 8px !important;
    }
    .stError {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb) !important;
        border: 1px solid #f5c6cb !important;
        color: #721c24 !important;
        border-radius: 8px !important;
    }
    .stWarning {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7) !important;
        border: 1px solid #ffeaa7 !important;
        color: #856404 !important;
        border-radius: 8px !important;
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
                st.sidebar.success("✅ Local AI Model Ready")
            else:
                st.sidebar.warning("⚠️ " + message)
        except Exception as e:
            st.sidebar.warning(f"⚠️ AI model check failed: {e}")
            self.ai_model_loaded = False

    def render_sidebar(self):
        """Render sidebar with configuration and information"""
        st.sidebar.title("⚙️ Configuration")
        
        # AI Model Status
        st.sidebar.markdown("### 🤖 Local AI Model")
        if self.ai_model_loaded:
            st.sidebar.success("✅ **Local AI Model Ready**")
            st.sidebar.markdown("**Model:** DistilGPT-2")
            st.sidebar.markdown("**Type:** Text Generation")
            if TORCH_AVAILABLE and torch is not None:
                hardware = "GPU" if torch.cuda.is_available() else "CPU"
            else:
                hardware = "CPU (PyTorch not available)"
            st.sidebar.markdown(f"**Hardware:** {hardware}")
        else:
            st.sidebar.warning("⚠️ **Using Fallback Reports**")
            st.sidebar.markdown("AI reports will use template-based generation")
        
        # Model Information
        st.sidebar.markdown("### 🎯 Health Assessment Models")
        if self.models_loaded:
            st.sidebar.success("✅ **Models loaded successfully!**")
            if hasattr(self.predictor, 'get_available_animals'):
                st.sidebar.markdown(f"**Animals:** {len(self.predictor.get_available_animals())}")
            if hasattr(self.predictor, 'get_available_symptoms'):
                st.sidebar.markdown(f"**Symptoms:** {len(self.predictor.get_available_symptoms())}")
        else:
            st.sidebar.error("❌ **Models not loaded**")
        
        # Quick Tips
        st.sidebar.markdown("### 💡 Quick Tips")
        st.sidebar.info("""
        - Select multiple symptoms for comprehensive assessment
        - High-risk symptoms significantly impact risk score
        - Model agreement indicates confidence in assessment
        - Generate AI report for detailed veterinary guidance
        - Always consult a veterinarian for professional diagnosis
        """)
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Veterinary Health Assessment System**")
        st.sidebar.markdown("*AI-Powered Animal Care*")

    def render_input_section(self):
        """Render input section for animal information"""
        st.markdown("### 🐾 Patient Information")
        
        # Create a nice container for the input section
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                animal = st.selectbox(
                    "**Animal Type**",
                    options=list(Config.ANIMAL_BREEDS.keys()),
                    index=0,
                    help="Select the type of animal"
                )
            
            with col2:
                breed = st.selectbox(
                    "**Breed**",
                    options=Config.ANIMAL_BREEDS.get(animal, ['mixed']),
                    index=0,
                    help="Select the specific breed"
                )
            
            with col3:
                age = st.selectbox(
                    "**Age Group**",
                    options=["young", "adult", "senior"],
                    index=1,
                    help="Select the age category"
                )
            
            with col4:
                weight = st.number_input(
                    "**Weight (kg)**",
                    min_value=0.1,
                    max_value=1000.0,
                    value=25.0,
                    step=0.1,
                    help="Enter weight in kilograms"
                )
        
        # Symptoms Selection
        st.markdown("### 🤒 Symptoms Selection")
        symptoms = st.multiselect(
            "**Select all applicable symptoms**",
            options=Config.SYMPTOMS,
            help="Choose all symptoms that apply",
            max_selections=10
        )
        
        # Display selected symptoms with risk levels
        if symptoms and self.models_loaded:
            st.markdown("### 📊 Selected Symptoms Analysis")
            for symptom in symptoms:
                severity = self.predictor.data_loader.symptom_severity_weights.get(symptom, 0.1)
                if severity > 0.7:
                    risk_class = "high-risk-symptom"
                    risk_icon = "🔴"
                elif severity > 0.4:
                    risk_class = "medium-risk-symptom"
                    risk_icon = "🟡"
                else:
                    risk_class = "low-risk-symptom"
                    risk_icon = "🟢"
                
                st.markdown(f"""
                <div class="symptom-item {risk_class}">
                    {risk_icon} <strong>{symptom}</strong> (Severity: {severity:.2f})
                </div>
                """, unsafe_allow_html=True)
        
        return {
            'animal': animal,
            'breed': breed,
            'age': age,
            'weight': weight,
            'symptoms': symptoms
        }

    def render_prediction_results(self, prediction_result, animal_info, symptoms):
        """Render prediction results"""
        if not isinstance(prediction_result, dict) or 'ensemble' not in prediction_result:
            st.error(f"Prediction error: {prediction_result}")
            return
        
        ensemble = prediction_result['ensemble']
        
        # Risk Level Display
        risk_score = ensemble['probability']
        if risk_score < 0.3:
            risk_class = "risk-low"
            risk_level = "LOW RISK"
            risk_icon = "🟢"
        elif risk_score < 0.7:
            risk_class = "risk-medium"
            risk_level = "MODERATE RISK"
            risk_icon = "🟡"
        else:
            risk_class = "risk-high"
            risk_level = "HIGH RISK"
            risk_icon = "🔴"
        
        st.markdown(f"""
        <div class="{risk_class}">
            <h2 style='color: inherit; margin: 0;'>{risk_icon} {risk_level}</h2>
            <p style='color: inherit; margin: 10px 0 0 0; font-size: 1.1em;'>
            <strong>Confidence:</strong> <span class="confidence-text">{ensemble['confidence']}</span> | 
            <strong>Model Agreement:</strong> {ensemble['model_agreement']} | 
            <strong>Risk Probability:</strong> {risk_score:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Columns for detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            # Individual Model Predictions
            st.markdown("### 🤖 Individual Model Predictions")
            for model_name, prediction in prediction_result['individual_predictions'].items():
                status_icon = "🔴" if prediction['dangerous'] else "🟢"
                status_text = "DANGEROUS" if prediction['dangerous'] else "SAFE"
                
                st.markdown(f"""
                <div class="model-card">
                    <strong style='font-size: 1.1em;'>{model_name}</strong><br>
                    <span style='font-size: 1.2em;'>{status_icon} {status_text}</span><br>
                    Confidence: <span class="confidence-text">{prediction['confidence']}</span><br>
                    Probability: {prediction['probability']:.3f}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Risk Visualization
            st.markdown("### 📊 Risk Visualization")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score", 'font': {'size': 24, 'color': '#2c3e50'}},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#2c3e50"},
                    'bar': {'color': "#3498db"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': "#27ae60"},
                        {'range': [30, 70], 'color': "#f39c12"},
                        {'range': [70, 100], 'color': "#e74c3c"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(
                height=300, 
                font={'color': "#2c3e50", 'family': "Arial"},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Symptom Summary
            if symptoms:
                st.markdown("### 🔍 Symptom Summary")
                high_risk_count = sum(1 for s in symptoms if self.predictor.data_loader.symptom_severity_weights.get(s, 0) > 0.7)
                medium_risk_count = sum(1 for s in symptoms if 0.4 < self.predictor.data_loader.symptom_severity_weights.get(s, 0) <= 0.7)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Symptoms", len(symptoms))
                with col2:
                    st.metric("High-Risk Symptoms", high_risk_count)
                with col3:
                    st.metric("Medium-Risk Symptoms", medium_risk_count)
        
        # AI Generated Report Section
        st.markdown("---")
        st.markdown("### 🏥 AI-Powered Veterinary Report")
        
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
                if st.button("🧠 Generate AI Veterinary Report", type="primary", use_container_width=True):
                    st.session_state.generating_report = True
                    st.session_state.report_generated = False
                    st.rerun()
        
        # Show AI Thinking Indicator
        if st.session_state.generating_report:
            st.markdown("""
            <div class="ai-thinking">
                <h3 class="blink">🧠 AI is analyzing...</h3>
                <p>Generating comprehensive veterinary report...</p>
                <p><em>This may take 10-30 seconds</em></p>
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
                st.success("✅ AI Report Generated Successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
                st.session_state.generating_report = False
        
        # Display report if available
        if st.session_state.report_generated and st.session_state.generated_report:
            st.markdown("### 📋 Detailed Veterinary Report")
            
            # Report display with download option
            report_col1, report_col2 = st.columns([3, 1])
            
            with report_col1:
                # Use a text area with monospace font for better readability
                st.text_area(
                    "Report Content", 
                    st.session_state.generated_report, 
                    height=500, 
                    key="report_display",
                    label_visibility="collapsed"
                )
            
            with report_col2:
                # Download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"vet_report_{animal_info['animal']}_{timestamp}.txt"
                
                st.download_button(
                    label="📥 Download Report",
                    data=st.session_state.generated_report,
                    file_name=filename,
                    mime="text/plain",
                    use_container_width=True
                )
                
                if st.button("🗑️ Clear Report", use_container_width=True):
                    st.session_state.report_generated = False
                    st.session_state.generated_report = None
                    st.session_state.generating_report = False
                    st.rerun()
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Immediate Recommendations")
        if ensemble['dangerous']:
            st.error("""
            ## 🚨 URGENT VETERINARY ATTENTION REQUIRED
            
            **Immediate Actions:**
            - 🏥 Contact emergency veterinarian immediately
            - 📞 Keep emergency veterinary number ready
            - 👀 Monitor vital signs continuously
            - 🛌 Keep animal calm and comfortable
            - 🚗 Prepare for immediate transport to veterinary clinic
            - ⚠️ Do not attempt home treatment for critical symptoms
            """)
        else:
            st.success("""
            ## ✅ CONTINUE MONITORING
            
            **Recommended Actions:**
            - 👀 Continue monitoring symptoms closely
            - 💧 Ensure access to fresh water and nutrition
            - 🛌 Provide comfortable resting area
            - 📞 Contact veterinarian if condition worsens
            - 📅 Schedule routine check-up if symptoms persist
            """)

    def run(self):
        """Main application runner"""
        # Header with gradient
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;'>
            <h1 class="main-header">🐾 Veterinary Health Assessment System</h1>
            <h3 style='color: white; margin: 0;'>AI-Powered Animal Health Risk Assessment</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content in a nice container
        with st.container():
            if not self.models_loaded:
                st.error("""
                ## ❌ Models Not Loaded
                
                Please ensure all model files are present in the 'models' directory.
                Run the training script first to generate the models.
                """)
            else:
                # Input section
                animal_info = self.render_input_section()
                
                # Assessment button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔍 Assess Health Status", type="primary", use_container_width=True):
                        # Clear previous reports when new assessment starts
                        if 'report_generated' in st.session_state:
                            st.session_state.report_generated = False
                        if 'generated_report' in st.session_state:
                            st.session_state.generated_report = None
                        if 'generating_report' in st.session_state:
                            st.session_state.generating_report = False
                        
                        if not animal_info['symptoms']:
                            st.warning("⚠️ Please select at least one symptom")
                        else:
                            with st.spinner("🤖 Analyzing health status..."):
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
                                    st.error(f"❌ Prediction error: {e}")
                
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