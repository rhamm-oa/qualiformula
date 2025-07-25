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
    """Categorize columns into numeric, text, and mixed types based on content."""
    numeric_cols = []
    text_cols = []
    mixed_cols = []

    for col in df.columns:
        # Attempt to convert to numeric silently
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        
        # Check if the column is completely numeric (no NaNs were created from non-numeric strings)
        if numeric_series.notna().all():
            numeric_cols.append(col)
            continue

        # If not completely numeric, check original non-null values
        original_not_null = df[col].notna()
        
        # Check if there are any numbers in the coerced series
        has_numbers = numeric_series.notna().any()
        
        # Check if there were original text values that are now NaN
        # This identifies values that couldn't be converted
        has_text = numeric_series.isna()[original_not_null].any()

        if has_numbers and has_text:
            # The column contains a mix of values that can be numbers and values that are just text
            mixed_cols.append(col)
        elif has_text:
            # The column only contains non-numeric values (like '4U2', 'L'Oreal')
            text_cols.append(col)
        elif has_numbers:
            # This case handles columns with numbers and NaNs, which are considered numeric
            numeric_cols.append(col)
        else:
            # Column is likely all NaNs, treat as text by default
            text_cols.append(col)
            
    return numeric_cols, text_cols, mixed_cols
