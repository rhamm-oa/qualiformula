import streamlit as st

def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="QualiFormula",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
