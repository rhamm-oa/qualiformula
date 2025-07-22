import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Import from custom modules
from config.streamlit_config import setup_page
from helpers.data_utils import load_data, get_column_types,get_color_families, analyze_color_aberrations

# --- 1. App Configuration ---
setup_page()

# --- 2. Main Application ---
st.title("🧪 QualiFormula: Data Quality & Visualization")
st.markdown("""
Welcome to QualiFormula! This tool is designed for comprehensive data quality assurance and anomaly detection.
""")

# --- 3. Data Loading ---
# IMPORTANT: Replace this with the actual path to your data file
file_path = "data/lipsticks/solid/Solid_lipstick_database.csv"  # <--- FURNISH YOUR FILE PATH HERE

df = load_data(file_path) 

# --- 4. Main Panel ---
if df is not None:
    st.success(f"Successfully loaded data from: {file_path}")
    
    # --- Data Overview Section ---
    st.header("📊 Data Overview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of Rows", df.shape[0])
    with col2:
        st.metric("Number of Columns", df.shape[1])

    st.subheader("Column Summary")
    missing_values = df.isnull().sum()
    column_summary = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': missing_values,
        'Missing (%)': (missing_values / df.shape[0]) * 100
    })
    st.dataframe(column_summary)

    # --- Column Type Analysis ---
    numeric_cols, text_cols, mixed_cols = get_column_types(df)
    st.subheader("Column Type Analysis")
    st.write(f"**Numeric Columns ({len(numeric_cols)}):**")
    st.info(', '.join(numeric_cols))
    st.write(f"**Textual Columns ({len(text_cols)}):**")
    st.info(', '.join(text_cols))
    st.write(f"**Mixed-Type Columns ({len(mixed_cols)}):**")
    st.info(', '.join(mixed_cols) if mixed_cols else "None detected")

    # --- Visualizations ---
    st.header("🔬 Visualizations")

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    if len(numeric_cols) > 1:
        corr_df = df[numeric_cols].corr()
        fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Numeric Column Correlation")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Not enough numeric columns to generate a correlation matrix.")

    # Sankey Diagram
    st.subheader("Sankey Diagram")
    categorical_cols = text_cols + mixed_cols
    if len(categorical_cols) >= 2:
        sankey_cols = st.multiselect("Select 2 or 3 columns for the Sankey diagram:", options=categorical_cols, default=categorical_cols[:2], max_selections=3)
        if sankey_cols and len(sankey_cols) >= 2:
            sankey_df = df[sankey_cols].dropna().groupby(sankey_cols).size().reset_index(name='value')
            if not sankey_df.empty:
                labels = pd.concat([sankey_df[col] for col in sankey_cols]).unique() # type: ignore
                label_map = {label: i for i, label in enumerate(labels)}
                source = sankey_df[sankey_cols[0]].map(label_map)
                target = sankey_df[sankey_cols[1]].map(label_map)
                value = sankey_df['value']
                if len(sankey_cols) == 3:
                    source = pd.concat([source, sankey_df[sankey_cols[1]].map(label_map)])
                    target = pd.concat([target, sankey_df[sankey_cols[2]].map(label_map)])
                    value = pd.concat([value, value])
                fig_sankey = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, label=labels), link=dict(source=source, target=target, value=value))])
                fig_sankey.update_layout(title_text=f"Flow between {', '.join(sankey_cols)}", font_size=10)
                st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.warning("Not enough categorical columns for a Sankey diagram.")

    # Sunburst Chart
    st.subheader("Sunburst Chart")
    if len(categorical_cols) >= 1:
        sunburst_path = st.multiselect("Select columns for the Sunburst hierarchy:", options=categorical_cols, default=categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols[:1])
        if sunburst_path:
            sunburst_df = df[sunburst_path].dropna()
            fig_sunburst = px.sunburst(sunburst_df, path=sunburst_path, title="Hierarchical Data Distribution")
            st.plotly_chart(fig_sunburst, use_container_width=True)

# --- Color Space Aberration Analysis ---
    st.header("🎨 Color Space Aberration Analysis")
    st.markdown("""
    This section analyzes color measurement aberrations in the L*a*b*C*h color space.
    It detects various types of errors including missing values, out-of-range measurements, and statistical outliers.
    """)
    
    # Load specific dataset for color analysis
    color_analysis_path = "data/lipsticks/solid/Solid_lipstick_database.csv"
    
    if os.path.exists(color_analysis_path):
        color_df = load_data(color_analysis_path)
        
        if color_df is not None:
            st.success(f"Loaded color analysis dataset: {len(color_df)} products")
            
            # Get available color families
            color_families = get_color_families(color_df)
            
            if color_families:
                # Color family selection
                selected_family = st.selectbox(
                    "Select Color Family for Analysis:",
                    options=['All Families'] + color_families,
                    index=0
                )
                
                # Analyze button
                if st.button("🔍 Analyze Color Aberrations", type="primary"):
                    analysis_family = None if selected_family == 'All Families' else selected_family
                    
                    with st.spinner('Analyzing color space aberrations...'):
                        results = analyze_color_aberrations(color_df, analysis_family)
                    
                    # Display results
                    st.subheader(f"📊 Analysis Results for: {selected_family}")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Products", results['total_products'])
                    with col2:
                        st.metric("Total Issues", results['summary']['total_issues'])
                    with col3:
                        st.metric("Affected Rows", results['summary']['critical_rows_count'])
                    with col4:
                        st.metric("Affected %", f"{results['summary']['percentage_affected']:.1f}%")
                    
                    # Detailed aberration breakdown
                    if results['summary']['total_issues'] > 0:
                        st.subheader("🔍 Detailed Aberration Breakdown")
                        
                        # Create aberration summary table
                        aberration_summary = []
                        for col, aberrations in results['aberrations'].items():
                            for aberration_type, indices in aberrations.items():
                                if indices and aberration_type != 'complete_nan_sets':
                                    aberration_summary.append({
                                        'Column': col,
                                        'Aberration Type': aberration_type.replace('_', ' ').title(),
                                        'Count': len(indices),
                                        'Severity': '🔴 Critical' if aberration_type in ['partial_nan_values', 'non_lab_nan_values', 'non_numeric', 'negative_values', 'out_of_range'] else '🟡 Warning'
                                    })
                        
                        if aberration_summary:
                            summary_df = pd.DataFrame(aberration_summary)
                            # Ensure all columns are proper types for display
                            summary_display = summary_df.copy()
                            for col in summary_display.columns:
                                if summary_display[col].dtype == 'object':
                                    summary_display[col] = summary_display[col].astype(str)

                            st.dataframe(summary_display, use_container_width=True)
                            
                            # Show critical rows details
                            if results['critical_rows']:
                                st.subheader("🚨 Critical Aberrant Rows")
                                
                                # Prepare columns to display
                                display_cols = ['FORMULA', 'PRODUCT NAME', 'COLOR FAMILY MED LIP'] + \
                                             ['L* BULK', 'a* BULK', 'b* BULK', 'C* BULK', 'h BULK']
                                available_cols = [col for col in display_cols if col in color_df.columns]
                                
                                # Get critical rows data
                                critical_df = color_df.loc[results['critical_rows'], available_cols].copy()
                                
                                # Add aberration details for each row
                                aberration_details = []
                                for idx in results['critical_rows']:
                                    details = []
                                    for col, aberrations in results['aberrations'].items():
                                        for aberration_type, indices in aberrations.items():
                                            if idx in indices and aberration_type != 'complete_nan_sets':
                                                details.append(f"{col}: {aberration_type.replace('_', ' ')}")
                                    aberration_details.append("; ".join(details))
                                
                                critical_df['Aberration Details'] = aberration_details
                                
                                # Sort by color family and display
                                if 'COLOR FAMILY MED LIP' in critical_df.columns:
                                    critical_df = critical_df.sort_values('COLOR FAMILY MED LIP')
                                
                                # Convert object columns to string to avoid PyArrow issues
                                display_df = critical_df.copy()
                                for col in display_df.columns:
                                    if display_df[col].dtype == 'object':
                                        display_df[col] = display_df[col].astype(str)

                                st.dataframe(display_df, use_container_width=True)
                                
                                # Download option for critical rows
                                csv_data = critical_df.to_csv(index=True).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Critical Rows CSV",
                                    data=csv_data,
                                    file_name=f"color_aberrations_{selected_family.replace(' ', '_')}.csv",
                                    mime='text/csv'
                                )
                                
                                # Visualization of aberrations by type
                                st.subheader("📈 Aberration Distribution")
                                
                                # Count aberrations by type
                                aberration_counts = {}
                                for col, aberrations in results['aberrations'].items():
                                    for aberration_type, indices in aberrations.items():
                                        if indices and aberration_type != 'complete_nan_sets':
                                            clean_type = aberration_type.replace('_', ' ').title()
                                            aberration_counts[clean_type] = aberration_counts.get(clean_type, 0) + len(indices)
                                
                                if aberration_counts:
                                    fig_aberrations = px.bar(
                                        x=list(aberration_counts.keys()),
                                        y=list(aberration_counts.values()),
                                        labels={'x': 'Aberration Type', 'y': 'Count'},
                                        title=f"Aberration Types Distribution - {selected_family}",
                                        color_discrete_sequence=['#2649B2', '#4A74F3', '#8E7DE3', '#9D5CE6', '#D4D9F0']
                                    )
                                    fig_aberrations.update_layout(xaxis_tickangle=-45)
                                    st.plotly_chart(fig_aberrations, use_container_width=True)
                    
                    else:
                        st.success("🎉 No critical color space aberrations found! Your color data looks clean.")
                        
                        # Still show complete NaN sets as info
                        complete_nan_info = []
                        for col, aberrations in results['aberrations'].items():
                            if aberrations['complete_nan_sets']:
                                complete_nan_info.append(f"{col}: {len(aberrations['complete_nan_sets'])} complete NaN sets")
                        
                        if complete_nan_info:
                            st.info("ℹ️ Complete NaN sets (acceptable missing data):\n" + "\n".join(complete_nan_info))
            
            else:
                st.warning("No 'COLOR FAMILY MED LIP' column found in the dataset.")
    
    else:
        st.error(f"Color analysis dataset not found at: {color_analysis_path}")
        st.info("Please ensure the file exists at the specified path.")
else:
    st.info("Please provide a valid file path in the script to load data.")
