import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Import from custom modules
from config.streamlit_config import setup_page
from modules.liquid_lipstick import render_liquid_lipstick_dashboard
from modules.solid_lipstick import render_solid_lipstick_dashboard

# --- 1. App Configuration ---
setup_page()

# --- 2. Main Application ---
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <h1 style='font-size: 3rem; font-weight: 700; color: #1f2937; margin-bottom: 0.5rem;'>
        💄 QualiFormula
    </h1>
    <h2 style='font-size: 1.5rem; font-weight: 400; color: #6b7280; margin-bottom: 1rem;'>
        Data Quality & Visualization Platform
    </h2>
    <p style='font-size: 1.1rem; color: #9ca3af; max-width: 600px; margin: 0 auto;'>
        Comprehensive data quality assurance and anomaly detection for cosmetic formulations
    </p>
</div>
""", unsafe_allow_html=True)

# --- Enhanced Sidebar ---
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <img src='https://images.seeklogo.com/logo-png/49/2/loreal-groupe-logo-png_seeklogo-499550.png' width='140' style='margin-bottom: 1rem;'>
    <h2 style='color: #374151; font-weight: 600; margin: 0;'>Dataset Selection</h2>
</div>
""", unsafe_allow_html=True)

lipstick_type = st.sidebar.radio(
    "Choose Lipstick Type:",
    ("💧 Liquid Lipstick", "💄 Solid Lipstick"),
    index=0,
    help="Select which lipstick dataset to analyze"
)

# Add some spacing
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.9rem;'>
    <p>Switch between datasets to explore different lipstick dataset's audit.</p>
</div>
""", unsafe_allow_html=True)

# --- Dashboard Rendering ---
if "Liquid" in lipstick_type:
    render_liquid_lipstick_dashboard()
else:
    render_solid_lipstick_dashboard()
