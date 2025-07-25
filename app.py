import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import from custom modules
from config.streamlit_config import setup_page
from helpers.data_utils import load_data, get_column_types

# --- 1. App Configuration ---
setup_page()

# --- 2. Main Application ---
st.title("🧪 QualiFormula: Data Quality & Visualization")
st.markdown("""
Welcome to QualiFormula! This tool is designed for comprehensive data quality assurance and anomaly detection.
""")

# --- 3. Data Loading ---
file_path = "data/lipsticks/liquid/Liquid_lipstick_database.csv"  # <--- FURNISH YOUR FILE PATH HERE

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
                # Generate a color palette for Sankey nodes
                import plotly.colors as pc
                palette = pc.sample_colorscale('Viridis', [i/(len(labels)-1) for i in range(len(labels))]) if len(labels) > 1 else ['#636EFA']
                node_colors = palette
                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(pad=15, thickness=20, label=labels, color=node_colors),
                    link=dict(source=source, target=target, value=value)
                )])
                fig_sankey.update_layout(
                    title_text=f"Flow between {', '.join(sankey_cols)}",
                    font=dict(size=16, color='black'),
                    height=1200,
                    width=1200,
                    margin=dict(l=0, r=0, t=80, b=0)
                )
                st.plotly_chart(fig_sankey, use_container_width=False)
    else:
        st.warning("Not enough categorical columns for a Sankey diagram.")

    # Sunburst Chart
    st.subheader("Sunburst Chart")
    if len(categorical_cols) >= 1:
        sunburst_path = st.multiselect("Select columns for the Sunburst hierarchy:", options=categorical_cols, default=categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols[:1])
        if sunburst_path:
            sunburst_df = df[sunburst_path].dropna().copy()
            fig_sunburst = px.sunburst(
                sunburst_df,
                path=sunburst_path,
                title="Hierarchical Data Distribution",
                color=None,
                color_discrete_sequence=px.colors.qualitative.Set3,
                maxdepth=-1 # show all levels
            )
            fig_sunburst.update_layout(
                height=1200,
                width=1200,
                font=dict(size=16),
                margin=dict(l=0, r=0, t=80, b=0),
                showlegend=False
            )
            fig_sunburst.update_traces(
                textinfo='label+percent entry',
                insidetextorientation='radial',
                hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Percent: %{percentParent:.2%}<extra></extra>',
                marker=dict(line=dict(color='white', width=1))
            )
            st.plotly_chart(fig_sunburst, use_container_width=False)
            # Drilldown table for all paths
            st.markdown("**Drilldown Table:** You can search for any value below.")
            # Build a table of all unique paths and their counts
            path_counts = sunburst_df.value_counts(subset=sunburst_path).reset_index(name='Count')
            # Add a filter box
            filter_text = st.text_input("Filter drilldown table (case-insensitive):", "")
            if filter_text:
                mask = path_counts.apply(lambda row: filter_text.lower() in ' '.join(row.astype(str)).lower(), axis=1)
                st.dataframe(path_counts[mask].reset_index(drop=True))
            else:
                st.dataframe(path_counts)
else:
    st.info("Please provide a valid file path in the script to load data.")
