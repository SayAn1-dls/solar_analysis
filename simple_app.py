import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Solar Power Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("☀️ Solar Power Forecasting Dashboard")
st.markdown("---")

# Main content
st.markdown("## 🌟 Welcome to Solar Power Forecasting")

st.info("""
**🚀 Application Status**: Successfully deployed on Streamlit Cloud!

**📋 Features**:
- Real-time solar power prediction
- Weather-based forecasting  
- Interactive data visualization
- Performance metrics and analysis

**⚠️ Note**: This is a demo version. Full functionality requires model training and data preparation.
""")

# Demo metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Status", "✅ Online", "Healthy")
    
with col2:
    st.metric("Framework", "Streamlit", "1.50.0")
    
with col3:
    st.metric("Python", "3.14.4", "Latest")

# Simple chart
st.markdown("## 📊 Sample Solar Power Generation")

# Create simple data
hours = list(range(24))
power = [0, 0, 0, 0, 0, 0, 50, 200, 450, 800, 1200, 1500, 1800, 2000, 2100, 1900, 1600, 1200, 600, 200, 50, 0, 0, 0]

# Create chart using streamlit's native chart
chart_data = pd.DataFrame({
    'Hour': hours,
    'Solar Power (W)': power
})

st.line_chart(chart_data.set_index('Hour'))

# Feature importance
st.markdown("## 🔍 Key Features")

features = ["Irradiation", "Module Temperature", "Ambient Temperature", "Hour of Day", "Month"]
importance = [0.35, 0.25, 0.15, 0.15, 0.10]

feature_data = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

st.bar_chart(feature_data.set_index('Feature'))

# Footer
st.markdown("---")
st.markdown("**🔗 GitHub Repository**: [solar_analysis](https://github.com/SayAn1-dls/solar_analysis)")
st.markdown("**📧 Deployment**: Successfully running on Streamlit Cloud")
