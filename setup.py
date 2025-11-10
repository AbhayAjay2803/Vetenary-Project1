from setuptools import setup, find_packages

setup(
    name="veterinary-health-assessment",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "scikit-learn>=1.2.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "joblib>=1.2.0",
        "python-dotenv>=1.0.0",
        "openai>=1.3.0",
        "xgboost>=1.7.0",
    ],
)