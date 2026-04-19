"""
This is the root entry point for Hugging Face Spaces.
Hugging Face expects an app.py in the root directory for Streamlit apps.
We simply import and run our main dashboard logic securely here.
"""
import sys
import os

# Add the project root to the python path so imports within streamlit_app work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Execute the main streamlit application natively
import app.streamlit_app
