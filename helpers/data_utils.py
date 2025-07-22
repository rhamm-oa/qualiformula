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


def analyze_color_aberrations(df, color_family=None):
    """
    Analyze color space aberrations in the DataFrame, optionally filtered by color family.
    
    Args:
        df: pandas DataFrame with color data
        color_family: Optional string to filter by specific color family
    
    Returns:
        dict: Dictionary containing aberration analysis results
    """
    
    # Define the color space columns
    color_columns = ['L* BULK', 'a* BULK', 'b* BULK', 'C* BULK', 'h BULK']
    lab_columns = ['L* BULK', 'a* BULK', 'b* BULK']
    
    # Filter by color family if specified
    if color_family and 'COLOR FAMILY MED LIP' in df.columns:
        analysis_df = df[df['COLOR FAMILY MED LIP'] == color_family].copy()
    else:
        analysis_df = df.copy()
    
    # Initialize results
    results = {
        'color_family': color_family,
        'total_products': len(analysis_df),
        'aberrations': {},
        'critical_rows': [],
        'summary': {}
    }
    
    # Track all critical aberrant indices
    critical_indices = set()
    
    for col in color_columns:
        if col not in analysis_df.columns:
            continue
            
        results['aberrations'][col] = {
            'complete_nan_sets': [],
            'partial_nan_values': [],
            'non_lab_nan_values': [],
            'non_numeric': [],
            'negative_values': [],
            'out_of_range': [],
            'extreme_outliers': []
        }
        
        # Handle NaN values
        nan_mask = analysis_df[col].isnull()
        
        if col in lab_columns:
            # Check for complete vs partial NaN sets
            nan_indices = analysis_df[nan_mask].index.tolist()
            
            for idx in nan_indices:
                lab_values_for_row = analysis_df.loc[idx, lab_columns]
                all_lab_nan = lab_values_for_row.isnull().all()
                
                if all_lab_nan:
                    results['aberrations'][col]['complete_nan_sets'].append(idx)
                else:
                    results['aberrations'][col]['partial_nan_values'].append(idx)
                    critical_indices.add(idx)
        
        else:  # For C* and h columns
            if nan_mask.sum() > 0:
                nan_indices = analysis_df[nan_mask].index.tolist()
                
                for idx in nan_indices:
                    lab_values_for_row = analysis_df.loc[idx, lab_columns]
                    has_lab_data = lab_values_for_row.notna().any()
                    
                    if has_lab_data:
                        results['aberrations'][col]['non_lab_nan_values'].append(idx)
                        critical_indices.add(idx)
        
        # Check for non-numeric values
        cleaned_series = analysis_df[col].astype(str).str.replace(',', '.', regex=False)  # Replace comma decimals
        cleaned_series = cleaned_series.str.strip()  # Remove whitespace
        # Handle Excel errors and common non-numeric strings
        cleaned_series = cleaned_series.replace([
            '', 'N/A', 'NULL', 'null', 'NaN', 'nan', '#N/A', '-', 
            '#DIV/0!', '#VALUE!', '#REF!', '#NAME?', '#NUM!', '#NULL!',  # Excel errors
            'None', 'none', 'NONE'
        ], pd.NA)

        numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
        non_numeric_mask = analysis_df[col].notna() & numeric_series.isna()
        
        if non_numeric_mask.sum() > 0:
            non_numeric_indices = analysis_df[non_numeric_mask].index.tolist()
            results['aberrations'][col]['non_numeric'] = non_numeric_indices
            critical_indices.update(non_numeric_indices)
        
        # Work with numeric values only for range checks
        valid_numeric = numeric_series.dropna()
        
        # Column-specific range checks
        if 'L*' in col:
            negative_mask = valid_numeric < 0
            over_100_mask = valid_numeric > 100
            
            if negative_mask.sum() > 0:
                neg_indices = negative_mask[negative_mask].index.tolist()
                results['aberrations'][col]['negative_values'] = neg_indices
                critical_indices.update(neg_indices)
            
            if over_100_mask.sum() > 0:
                over_indices = over_100_mask[over_100_mask].index.tolist()
                results['aberrations'][col]['out_of_range'] = over_indices
                critical_indices.update(over_indices)
        
        elif col in ['a* BULK', 'b* BULK']:
            extreme_negative = valid_numeric < -128
            extreme_positive = valid_numeric > 127
            
            extreme_indices = []
            if extreme_negative.sum() > 0:
                extreme_indices.extend(extreme_negative[extreme_negative].index.tolist())
            if extreme_positive.sum() > 0:
                extreme_indices.extend(extreme_positive[extreme_positive].index.tolist())
            
            if extreme_indices:
                results['aberrations'][col]['extreme_outliers'] = extreme_indices
        
        elif 'C*' in col:
            negative_mask = valid_numeric < 0
            if negative_mask.sum() > 0:
                neg_indices = negative_mask[negative_mask].index.tolist()
                results['aberrations'][col]['negative_values'] = neg_indices
                critical_indices.update(neg_indices)
        
        elif 'h' in col.lower():
            negative_mask = valid_numeric < 0
            over_360_mask = valid_numeric > 360
            
            if negative_mask.sum() > 0:
                neg_indices = negative_mask[negative_mask].index.tolist()
                results['aberrations'][col]['negative_values'] = neg_indices
                critical_indices.update(neg_indices)
            
            if over_360_mask.sum() > 0:
                over_indices = over_360_mask[over_360_mask].index.tolist()
                results['aberrations'][col]['out_of_range'] = over_indices
                critical_indices.update(over_indices)
    
    # Create summary of critical rows
    if critical_indices:
        results['critical_rows'] = list(critical_indices)
    
    # Generate summary statistics
    total_issues = sum(len(indices) for col_aberrations in results['aberrations'].values() 
                      for aberration_type, indices in col_aberrations.items() 
                      if indices and aberration_type != 'complete_nan_sets')
    
    results['summary'] = {
        'total_issues': total_issues,
        'critical_rows_count': len(critical_indices),
        'percentage_affected': (len(critical_indices) / len(analysis_df) * 100) if len(analysis_df) > 0 else 0
    }
    
    return results

def get_color_families(df):
    """Get list of available color families from DataFrame."""
    if 'COLOR FAMILY MED LIP' in df.columns:
        return sorted(df['COLOR FAMILY MED LIP'].dropna().unique().tolist())
    return []