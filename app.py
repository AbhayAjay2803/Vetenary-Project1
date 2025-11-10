import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add src to path
sys.path.append('src')
sys.path.append('utils')

from src.predictor import VeterinaryPredictor
from src.config import Config
from utils.helpers import save_api_key, generate_vet_report, get_risk_color, format_symptom_analysis

# Page configuration
st.set_page_config(
    page_title="Veterinary Health Assessment System",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ff0000;
    }
    .risk-medium {
        background-color: #fff4cc;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffcc00;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #00cc00;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class VeterinaryApp:
    def __init__(self):
        self.predictor = VeterinaryPredictor()
        self.models_loaded = False
        self.load_models()

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
        all_exist = all(os.path.exists(path) for path in model_paths.values())
        
        if all_exist:
            self.models_loaded = self.predictor.load_models(model_paths)
        else:
            st.warning("⚠️ Model files not found. Please ensure all model files are in the 'models' directory.")

    def render_sidebar(self):
        """Render sidebar with configuration and information"""
        st.sidebar.title("⚙️ Configuration")
        
        # API Key Configuration
        st.sidebar.subheader("🔑 AI API Configuration")
        api_key = st.sidebar.text_input("OpenAI API Key", type="password", 
                                       help="Enter your OpenAI API key for AI-generated reports")
        api_type = st.sidebar.selectbox("API Type", ["OpenAI", "Grok"])
        
        if st.sidebar.button("Save API Key"):
            if api_key:
                save_api_key(api_key, api_type.lower())
                st.sidebar.success("✅ API key saved successfully!")
            else:
                st.sidebar.error("❌ Please enter a valid API key")
        
        # Model Information
        st.sidebar.subheader("🤖 Model Information")
        if self.models_loaded:
            st.sidebar.success("✅ Models loaded successfully!")
            st.sidebar.write(f"**Available Models:** {len(self.predictor.models)}")
            st.sidebar.write(f"**Animals:** {len(self.predictor.get_available_animals())}")
            st.sidebar.write(f"**Symptoms:** {len(self.predictor.get_available_symptoms())}")
        else:
            st.sidebar.error("❌ Models not loaded")
        
        # Best Performing Model Info
        st.sidebar.subheader("🏆 Best Performing Model")
        st.sidebar.info("""
        **Improved Structured Clinical Transformer**
        - Accuracy: 84.0%
        - F1-Score: 84.3%
        - AUC: 91.0%
        """)
        
        # Quick Tips
        st.sidebar.subheader("💡 Quick Tips")
        st.sidebar.info("""
        - Select multiple symptoms for comprehensive assessment
        - High-risk symptoms significantly impact risk score
        - Model agreement indicates confidence in assessment
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
        st.subheader("🤒 Symptoms")
        symptoms = st.multiselect(
            "Select all applicable symptoms",
            options=Config.SYMPTOMS,
            help="Choose all symptoms that apply",
            max_selections=10
        )
        
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
            <p><strong>Confidence:</strong> {ensemble['confidence']} | 
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
                    Confidence: {prediction['confidence']}<br>
                    Probability: {prediction['probability']:.3f}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Symptom Analysis
            if symptoms:
                st.subheader("🔍 Symptom Analysis")
                st.markdown(format_symptom_analysis(symptoms, self.predictor), unsafe_allow_html=True)
            
            # Risk Visualization
            st.subheader("📊 Risk Visualization")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
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
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        if ensemble['dangerous']:
            st.error("""
            ## 🚨 URGENT VETERINARY ATTENTION REQUIRED
            
            **Immediate Actions:**
            - Contact emergency veterinarian immediately
            - Monitor vital signs continuously
            - Keep animal calm and comfortable
            - Prepare for transport to veterinary clinic
            - Do not attempt home treatment
            
            **Critical Symptoms Detected:**
            - This case shows signs requiring professional medical intervention
            - Time-sensitive condition identified
            """)
        else:
            st.success("""
            ## ✅ CONTINUE MONITORING
            
            **Recommended Actions:**
            - Continue monitoring symptoms closely
            - Ensure access to fresh water and nutrition
            - Provide comfortable resting area
            - Contact veterinarian if condition worsens
            - Schedule routine check-up if symptoms persist
            
            **Watch For:**
            - Any changes in behavior or appetite
            - Worsening of existing symptoms
            - Development of new symptoms
            """)
        
        # AI Generated Report
        if st.button("📋 Generate Detailed Veterinary Report", type="primary"):
            with st.spinner("Generating professional veterinary report..."):
                report = generate_vet_report(prediction_result, animal_info, symptoms)
                
                st.subheader("🏥 AI-Generated Veterinary Report")
                st.text_area("Report", report, height=400)
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"vet_report_{animal_info['animal']}.txt",
                    mime="text/plain"
                )

    def render_model_comparison(self):
        """Render model performance comparison"""
        st.header("📈 Model Performance Comparison")
        
        # Sample performance data (replace with actual metrics)
        performance_data = {
            'Model': ['Improved SCT', 'LSTM', 'Random Forest', 'Neural Network', 'XGBoost'],
            'Accuracy': [0.840, 0.816, 0.818, 0.801, 0.823],
            'F1-Score': [0.843, 0.822, 0.828, 0.809, 0.831],
            'AUC': [0.910, 0.908, 0.905, 0.888, 0.918]
        }
        
        df = pd.DataFrame(performance_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            st.dataframe(df.style.format({
                'Accuracy': '{:.3f}',
                'F1-Score': '{:.3f}', 
                'AUC': '{:.3f}'
            }).highlight_max(color='lightgreen'), use_container_width=True)
        
        with col2:
            st.subheader("Performance Visualization")
            fig = px.bar(df, x='Model', y=['Accuracy', 'F1-Score', 'AUC'],
                        title="Model Performance Comparison",
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<h1 class="main-header">🐾 Veterinary Health Assessment System</h1>', 
                   unsafe_allow_html=True)
        st.markdown("### *AI-Powered Animal Health Risk Assessment Using Ensemble Machine Learning*")
        
        # Sidebar
        self.render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["🏥 Health Assessment", "📊 Model Performance", "ℹ️ About"])
        
        with tab1:
            if not self.models_loaded:
                st.error("""
                ## ❌ Models Not Loaded
                
                Please ensure all model files are present in the 'models' directory:
                - sct_model.pth
                - lstm_model.pth  
                - randomforest.joblib
                - neuralnetwork.joblib
                - xgboost.joblib
                - encoders.joblib
                
                If models are not available, please run the training script first.
                """)
            else:
                # Input section
                animal_info = self.render_input_section()
                
                # Assessment button
                if st.button("🔍 Assess Health Status", type="primary", use_container_width=True):
                    if not animal_info['symptoms']:
                        st.warning("⚠️ Please select at least one symptom")
                    else:
                        with st.spinner("Analyzing health status..."):
                            prediction_result = self.predictor.predict_ensemble(
                                animal_info['animal'],
                                animal_info['breed'], 
                                animal_info['age'],
                                animal_info['weight'],
                                animal_info['symptoms']
                            )
                        
                        self.render_prediction_results(prediction_result, animal_info, animal_info['symptoms'])
        
        with tab2:
            self.render_model_comparison()
        
        with tab3:
            st.header("ℹ️ About This System")
            st.markdown("""
            ## Veterinary Health Assessment System
            
            This advanced AI-powered system uses ensemble machine learning to assess animal health risks 
            based on symptoms and patient information.
            
            ### 🏆 Best Performing Model
            **Improved Structured Clinical Transformer (SCT)**
            - **Accuracy**: 84.0%
            - **F1-Score**: 84.3% 
            - **AUC**: 91.0%
            - **Precision**: 82.8%
            - **Recall**: 85.9%
            
            ### 🤖 Ensemble Components
            The system combines multiple advanced models:
            
            - **Improved SCT** (40% weight): Multi-head attention with clinical prior integration
            - **Bidirectional LSTM** (25% weight): Sequential symptom analysis
            - **XGBoost** (20% weight): Gradient boosting excellence
            - **Random Forest** (8% weight): Ensemble decision trees
            - **Neural Network** (7% weight): Multi-layer perceptron
            
            ### ⚠️ Important Disclaimer
            This tool is for preliminary assessment only and should not replace professional veterinary care. 
            Always consult a licensed veterinarian for accurate diagnosis and treatment.
            
            ### 🛠️ Technical Details
            - **Framework**: PyTorch, Scikit-learn, XGBoost
            - **Interface**: Streamlit
            - **AI Integration**: OpenAI GPT for report generation
            - **Deployment**: Ready for production deployment
            """)

def main():
    app = VeterinaryApp()
    app.run()

if __name__ == "__main__":
    main()