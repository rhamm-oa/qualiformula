import streamlit as st
import pandas as pd
import numpy as np
import os
import openpyxl

def load_data(file_path):
    """Load data from the given file path into a pandas DataFrame."""
    if not file_path or not os.path.exists(file_path):
        st.warning(f"File not found or path is incorrect: {file_path}")
        return None
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, sep=';', decimal=',')
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path, engine='openpyxl', decimal=',')
        else:
            st.error("Unsupported file format. Please provide a path to a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def get_column_types(df):
    """Categorize columns into numeric, text, and mixed types."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    text_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    truly_text = []
    mixed_cols = []
    for col in text_cols:
        try:
            pd.to_numeric(df[col], errors='raise')
            if col not in numeric_cols:
                 numeric_cols.append(col)
        except (ValueError, TypeError):
            is_mixed = df[col].dropna().astype(str).str.contains(r'[A-Za-z]').any() and \
                       df[col].dropna().astype(str).str.contains(r'[0-9]').any()
            if is_mixed:
                mixed_cols.append(col)
            else:
                truly_text.append(col)

    return numeric_cols, truly_text, mixed_cols
