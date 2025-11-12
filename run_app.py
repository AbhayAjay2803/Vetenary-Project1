# run_app.py
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

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
        print("PyTorch not installed. Please install it manually:")
        print("pip install torch torchvision")
        return False

if __name__ == "__main__":
    # Install requirements
    try:
        install_requirements()
    except Exception as e:
        print(f"Installation failed: {e}")
        print("Please install requirements manually: pip install -r requirements.txt")
    
    # Check PyTorch
    if check_torch():
        print("All dependencies installed successfully!")
        print("Starting the application...")
        os.system("streamlit run app.py")
    else:
        print("Please install PyTorch manually and then run: streamlit run app.py")