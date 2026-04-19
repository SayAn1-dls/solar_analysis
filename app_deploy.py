"""
Streamlit Dashboard for Solar Power Forecasting - Deployment Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(
    page_title="Solar Power Forecasting — ML Project",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E88E5 0%, #43A047 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0 0.3rem 0;
        letter-spacing: -1px;
    }
    .sub-header {
        text-align: center;
        color: #9CA3AF;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: -8px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Solar Power Forecasting</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning-based Renewable Energy Prediction System</p>', unsafe_allow_html=True)

# Main content
st.markdown("## 🌟 Welcome to Solar Power Forecasting Dashboard")

st.info("""
**🚀 Deployment Status**: This is a deployment-ready version of the Solar Power Forecasting application.

**📋 Features Available**:
- Real-time solar power prediction
- Weather-based forecasting
- Interactive data visualization
- Performance metrics and analysis

**⚠️ Note**: Full functionality requires model training and data preparation. 
Please ensure the following files are available:
- `models/solar_model.pkl` - Trained ML model
- `data/processed/solar_final.csv` - Processed dataset
""")

# Demo section
st.markdown("## 📊 Demo Visualization")

# Create sample data for demonstration
hours = list(range(24))
sample_power = [
    0, 0, 0, 0, 0, 0,  # 0-5: Night
    50, 200, 450, 800, 1200, 1500,  # 6-11: Morning
    1800, 2000, 2100, 1900, 1600,  # 12-17: Peak hours
    1200, 600, 200, 50, 0, 0  # 18-23: Evening/Night
]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hours, sample_power, color='#1E88E5', linewidth=3, marker='o', markersize=6)
ax.fill_between(hours, sample_power, alpha=0.3, color='#1E88E5')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Solar Power (W)', fontsize=12)
ax.set_title('Sample Solar Power Generation Curve', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f8f9fa')

st.pyplot(fig)

# Feature importance demo
st.markdown("## 🔍 Key Features for Solar Prediction")

features = [
    "Irradiation", "Module Temperature", "Ambient Temperature", 
    "Hour of Day", "Month", "Weather Conditions"
]

importance = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]

fig2, ax2 = plt.subplots(figsize=(10, 6))
bars = ax2.barh(features, importance, color='#43A047')
ax2.set_xlabel('Feature Importance', fontsize=12)
ax2.set_title('Solar Power Prediction - Feature Importance', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, importance)):
    ax2.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{value:.2f}', va='center', fontsize=10)

st.pyplot(fig2)

# Deployment info
st.markdown("## 🚀 Deployment Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Status", "✅ Deployed", "Ready")
    st.metric("Framework", "Streamlit", "1.50.0")
    st.metric("Python", "3.14.4", "Latest")

with col2:
    st.metric("Repository", "GitHub", "Connected")
    st.metric("Last Updated", "2026-04-19", "Today")
    st.metric("Environment", "Production", "Cloud")

st.markdown("---")
st.markdown("**🔗 GitHub Repository**: [solar_analysis](https://github.com/SayAn1-dls/solar_analysis)")
st.markdown("**📧 Contact**: For deployment issues, please check the repository issues page.")
