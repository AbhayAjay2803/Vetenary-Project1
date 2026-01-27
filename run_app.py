# run_app.py
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except Exception as e:
        print(f"Installation failed: {e}")
        print("Please install requirements manually: pip install -r requirements.txt")

def check_torch():
    """Check if PyTorch is installed"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print("GPU is available!")
        else:
            print("Using CPU")
        return True
    except ImportError:
        print("PyTorch not installed.")
        return False

def run_streamlit():
    """Run Streamlit app using Python module"""
    print("Starting the application...")
    try:
        # Method 1: Try to run streamlit as module
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "app.py"])
    except Exception as e:
        print(f"Failed to run Streamlit as module: {e}")
        print("\nTrying alternative method...")
        try:
            # Method 2: Try to find streamlit executable
            streamlit_path = os.path.join(os.path.dirname(sys.executable), "streamlit")
            if os.path.exists(streamlit_path):
                subprocess.check_call([streamlit_path, "run", "app.py"])
            else:
                # Method 3: Try pip's bin directory
                import site
                streamlit_path = os.path.join(site.getusersitepackages(), "..", "..", "bin", "streamlit")
                if os.path.exists(streamlit_path):
                    subprocess.check_call([streamlit_path, "run", "app.py"])
                else:
                    print("Could not find Streamlit. Please ensure it's installed.")
                    print("Try: pip install streamlit")
        except Exception as e2:
            print(f"All methods failed: {e2}")
            print("\nPlease run manually:")
            print(f"1. Activate your Python environment")
            print(f"2. Run: python -m streamlit run app.py")

if __name__ == "__main__":
    # Install requirements
    install_requirements()
    
    # Check PyTorch
    if check_torch():
        run_streamlit()
    else:
        print("Please install PyTorch manually:")
        print("For CPU: pip install torch torchvision")
        print("For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")