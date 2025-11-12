# app.py - Updated with better visibility
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

# Custom CSS with DARKER TEXT and better visibility
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
        margin: 10px 0;
        color: #000000 !important;
    }
    .risk-medium {
        background-color: #fff4cc;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffcc00;
        margin: 10px 0;
        color: #000000 !important;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00cc00;
        margin: 10px 0;
        color: #000000 !important;
    }
    .model-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
        color: #000000 !important;
    }
    .confidence-text {
        color: #2c3e50 !important;
        font-weight: bold;
        font-size: 1.1em;
        background-color: #ecf0f1;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .symptom-item {
        padding: 8px;
        margin: 4px 0;
        border-radius: 5px;
        border-left: 4px solid;
        color: #000000 !important;
    }
    .high-risk-symptom {
        border-left-color: #e74c3c;
        background-color: #ffebee;
        color: #000000 !important;
    }
    .medium-risk-symptom {
        border-left-color: #f39c12;
        background-color: #fff3e0;
        color: #000000 !important;
    }
    .low-risk-symptom {
        border-left-color: #27ae60;
        background-color: #e8f5e8;
        color: #000000 !important;
    }
    .report-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #3498db;
        color: #000000 !important;
    }
    .ai-thinking {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        text-align: center;
        margin: 20px 0;
        color: #000000 !important;
    }
    .blink {
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    /* DARK TEXT FOR ALL CONTENT */
    .main .block-container {
        color: #000000 !important;
    }
    .stMarkdown {
        color: #000000 !important;
    }
    .stText {
        color: #000000 !important;
    }
    .report-text {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    .stTextArea textarea {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    /* Table styling */
    .dataframe {
        color: #000000 !important;
    }
    .stDataFrame {
        color: #000000 !important;
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
        st.sidebar.subheader("🤖 Local AI Model")
        if self.ai_model_loaded:
            st.sidebar.success("✅ Local AI Model Ready")
            st.sidebar.write("**Model:** DistilGPT-2")
            st.sidebar.write("**Type:** Text Generation")
            if TORCH_AVAILABLE and torch is not None:
                hardware = "GPU" if torch.cuda.is_available() else "CPU"
            else:
                hardware = "CPU (PyTorch not available)"
            st.sidebar.write(f"**Hardware:** {hardware}")
        else:
            st.sidebar.warning("⚠️ Using Fallback Reports")
            st.sidebar.write("AI reports will use template-based generation")
        
        # Model Information
        st.sidebar.subheader("🎯 Health Assessment Models")
        if self.models_loaded:
            st.sidebar.success("✅ Models loaded successfully!")
            if hasattr(self.predictor, 'get_available_animals'):
                st.sidebar.write(f"**Animals:** {len(self.predictor.get_available_animals())}")
            if hasattr(self.predictor, 'get_available_symptoms'):
                st.sidebar.write(f"**Symptoms:** {len(self.predictor.get_available_symptoms())}")
        else:
            st.sidebar.error("❌ Models not loaded")
        
        # Quick Tips
        st.sidebar.subheader("💡 Quick Tips")
        st.sidebar.info("""
        - Select multiple symptoms for comprehensive assessment
        - High-risk symptoms significantly impact risk score
        - Model agreement indicates confidence in assessment
        - Generate AI report for detailed veterinary guidance
        - Always consult a veterinarian for professional diagnosis
        """)

    def render_input_section(self):
        """Render input section for animal information"""
        st.header("🐾 Patient Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            animal = st.selectbox(
                "Animal Type",
                options=list(Config.ANIMAL_BREEDS.keys()),
                index=0,
                help="Select the type of animal"
            )
        
        with col2:
            breed = st.selectbox(
                "Breed",
                options=Config.ANIMAL_BREEDS.get(animal, ['mixed']),
                index=0,
                help="Select the specific breed"
            )
        
        with col3:
            age = st.selectbox(
                "Age Group",
                options=["young", "adult", "senior"],
                index=1,
                help="Select the age category"
            )
        
        with col4:
            weight = st.number_input(
                "Weight (kg)",
                min_value=0.1,
                max_value=1000.0,
                value=25.0,
                step=0.1,
                help="Enter weight in kilograms"
            )
        
        # Symptoms Selection
        st.subheader("🤒 Symptoms Selection")
        symptoms = st.multiselect(
            "Select all applicable symptoms",
            options=Config.SYMPTOMS,
            help="Choose all symptoms that apply",
            max_selections=10
        )
        
        # Display selected symptoms with risk levels
        if symptoms and self.models_loaded:
            st.subheader("📊 Selected Symptoms Analysis")
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
            <h2>{risk_icon} {risk_level}</h2>
            <p><strong>Confidence:</strong> <span class="confidence-text">{ensemble['confidence']}</span> | 
            <strong>Model Agreement:</strong> {ensemble['model_agreement']} | 
            <strong>Risk Probability:</strong> {risk_score:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Columns for detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            # Individual Model Predictions
            st.subheader("🤖 Individual Model Predictions")
            for model_name, prediction in prediction_result['individual_predictions'].items():
                status_icon = "🔴" if prediction['dangerous'] else "🟢"
                status_text = "DANGEROUS" if prediction['dangerous'] else "SAFE"
                
                st.markdown(f"""
                <div class="model-card">
                    <strong>{model_name}</strong><br>
                    Status: {status_icon} {status_text}<br>
                    Confidence: <span class="confidence-text">{prediction['confidence']}</span><br>
                    Probability: {prediction['probability']:.3f}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Risk Visualization
            st.subheader("📊 Risk Visualization")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score", 'font': {'size': 24}},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
            st.plotly_chart(fig, use_container_width=True)
            
            # Symptom Summary
            if symptoms:
                st.subheader("🔍 Symptom Summary")
                high_risk_count = sum(1 for s in symptoms if self.predictor.data_loader.symptom_severity_weights.get(s, 0) > 0.7)
                medium_risk_count = sum(1 for s in symptoms if 0.4 < self.predictor.data_loader.symptom_severity_weights.get(s, 0) <= 0.7)
                
                st.metric("Total Symptoms", len(symptoms))
                st.metric("High-Risk Symptoms", high_risk_count)
                st.metric("Medium-Risk Symptoms", medium_risk_count)
        
        # AI Generated Report Section
        st.markdown("---")
        st.subheader("🏥 AI-Powered Veterinary Report")
        
        # Initialize session state for report generation
        if 'report_generated' not in st.session_state:
            st.session_state.report_generated = False
        if 'generated_report' not in st.session_state:
            st.session_state.generated_report = None
        if 'generating_report' not in st.session_state:
            st.session_state.generating_report = False
        
        # Generate Report Button
        if not st.session_state.generating_report and not st.session_state.report_generated:
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
        st.subheader("💡 Immediate Recommendations")
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
        # Header
        st.markdown('<h1 class="main-header">🐾 Veterinary Health Assessment System</h1>', 
                   unsafe_allow_html=True)
        st.markdown("### *AI-Powered Animal Health Risk Assessment*")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
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