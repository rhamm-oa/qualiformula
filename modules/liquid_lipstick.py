import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from difflib import SequenceMatcher
from helpers.data_utils import load_data, get_column_types, get_color_families, analyze_color_aberrations

# File path constant
FILE_PATH = "data/lipsticks/liquid/Liquid_lipstick_database.csv"

def render_liquid_lipstick_dashboard():
    df = load_data(FILE_PATH)
    if df is None:
        st.error(f"Failed to load the liquid lipstick dataset at: {FILE_PATH}")
        st.info("Please check that the file exists and the path is correct.")
        return
    st.success(f"Successfully loaded data from: {FILE_PATH}")
    
    # Get column types once for use across all tabs
    numeric_cols, text_cols, mixed_cols = get_column_types(df)
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🔗 Correlation Analysis", 
        "🌊 Interactive Visualizations", 
        "🎨 Color Space Analysis",
        "🔍 Text Similarity Analysis"
    ])
    
    with tab1:
        st.header("📊 Data Overview")
        st.dataframe(df.head(), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Rows", df.shape[0])
        with col2:
            st.metric("Number of Columns", df.shape[1])
        
        st.header("🧾 Column Summary")
        missing_values = df.isnull().sum()
        column_summary = pd.DataFrame({
            'Data Type': df.dtypes.astype(str),  # Convert to string to avoid Arrow issues
            'Missing Values': missing_values,
            'Missing (%)': (missing_values / df.shape[0]) * 100
        })
        st.dataframe(column_summary, use_container_width=True)
        
        st.header("🔢 Column Type Analysis")
        st.markdown(f"**Numeric Columns ({len(numeric_cols)}):**")
        st.info(', '.join(numeric_cols))
        st.markdown(f"**Textual Columns ({len(text_cols)}):**")
        st.info(', '.join(text_cols))
        st.markdown(f"**Mixed-Type Columns ({len(mixed_cols)}):**")
        st.info(', '.join(mixed_cols) if mixed_cols else "None detected")
    
    with tab2:
        st.header("🔗 Correlation Matrix")
        if len(numeric_cols) > 1:
            corr_df = df[numeric_cols].corr()
            fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Numeric Column Correlation")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Not enough numeric columns to generate a correlation matrix.")
    
    with tab3:
        st.header("🌊 Interactive Visualizations")
        
        # Create sub-columns for Sankey and Sunburst
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌊 Sankey Diagram")
            categorical_cols = text_cols + mixed_cols
            if len(categorical_cols) >= 2:
                sankey_cols = st.multiselect("Select 2 or 3 columns for the Sankey diagram:", options=categorical_cols, default=categorical_cols[:2], max_selections=3)
                
                if sankey_cols and len(sankey_cols) >= 2:
                    # Filter options to reduce clutter
                    st.markdown("**Filter Options:**")
                    col_filter1, col_filter2 = st.columns(2)
                    
                    with col_filter1:
                        min_flow_value = st.number_input("Minimum flow value:", min_value=1, value=2, help="Hide flows below this threshold to reduce clutter")
                    
                    with col_filter2:
                        top_n_values = st.number_input("Show top N values per column:", min_value=5, max_value=50, value=15, help="Limit values to most common ones")
                    
                    # Prepare data with filtering
                    filtered_df = df[sankey_cols].dropna().copy()
                    
                    # Limit top N values per column to reduce complexity
                    for col in sankey_cols:
                        top_values = filtered_df[col].value_counts().head(top_n_values).index
                        filtered_df = filtered_df[filtered_df[col].isin(top_values)]
                    
                    sankey_df = filtered_df.groupby(sankey_cols).size().reset_index(name='value')
                    sankey_df = sankey_df[sankey_df['value'] >= min_flow_value]
                    
                    if not sankey_df.empty:
                        labels = pd.concat([sankey_df[col] for col in sankey_cols]).unique().tolist() # type: ignore
                        label_map = {label: i for i, label in enumerate(labels)}
                        
                        source = sankey_df[sankey_cols[0]].map(label_map)
                        target = sankey_df[sankey_cols[1]].map(label_map)
                        value = sankey_df['value']
                        
                        if len(sankey_cols) == 3:
                            source = pd.concat([source, sankey_df[sankey_cols[1]].map(label_map)])
                            target = pd.concat([target, sankey_df[sankey_cols[2]].map(label_map)])
                            value = pd.concat([value, value])
                        
                        # Enhanced color palette
                        import plotly.colors as pc
                        color_palettes = {
                            'Viridis': pc.qualitative.Plotly,
                            'Set3': pc.qualitative.Set3,
                            'Pastel': pc.qualitative.Pastel,
                            'Bold': pc.qualitative.Bold
                        }
                        
                        selected_palette = st.selectbox("Choose color palette:", list(color_palettes.keys()), index=0)
                        colors = color_palettes[selected_palette]
                        node_colors = [colors[i % len(colors)] for i in range(len(labels))]
                        
                        # Create enhanced Sankey diagram
                        fig_sankey = go.Figure(data=[go.Sankey(
                            node=dict(
                                pad=20,
                                thickness=25,
                                label=labels,
                                color=node_colors,
                                line=dict(color="rgba(0,0,0,0.3)", width=1)
                            ),
                            link=dict(
                                source=source,
                                target=target,
                                value=value,
                                color="rgba(0,0,0,0.2)",
                                hovertemplate='%{source.label} → %{target.label}<br>Flow: %{value}<extra></extra>'
                            )
                        )])
                        
                        fig_sankey.update_layout(
                            title=dict(
                                text=f"Flow Analysis: {' → '.join(sankey_cols)}",
                                font=dict(size=18, color='#2c3e50')
                            ),
                            font=dict(size=14, family='Inter, Helvetica Neue, Arial'),
                            height=800,
                            margin=dict(l=10, r=10, t=80, b=10),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig_sankey, use_container_width=True)
                        
                        # Summary statistics
                        st.markdown(f"**Summary:** Showing {len(sankey_df)} flows with {len(labels)} unique values")
                    else:
                        st.warning("No data meets the current filter criteria. Try reducing the minimum flow value or increasing the top N values.")
            else:
                st.warning("Not enough categorical columns for a Sankey diagram.")
        
        with col2:
            st.subheader("🌞 Sunburst Chart")
            if len(categorical_cols) >= 1:
                sunburst_path = st.multiselect("Select columns for the Sunburst hierarchy:", options=categorical_cols, default=categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols[:1])
                
                if sunburst_path:
                    # Prepare data
                    sunburst_df = df[sunburst_path].dropna().copy()
                    
                    if not sunburst_df.empty:
                        sunburst_df['count'] = 1
                        grouped = sunburst_df.groupby(sunburst_path)['count'].sum().reset_index()
                        
                        # Enhanced color schemes
                        color_schemes = {
                            'Sunset': px.colors.sequential.Sunset,
                            'Viridis': px.colors.sequential.Viridis,
                            'Plasma': px.colors.sequential.Plasma,
                            'Turbo': px.colors.sequential.Turbo,
                            'Rainbow': px.colors.qualitative.Set3
                        }
                        
                        selected_scheme = st.selectbox("Choose color scheme:", list(color_schemes.keys()), index=0, key="sb_colors")
                        
                        # Create enhanced Sunburst
                        fig_sunburst = px.sunburst(
                            grouped,
                            path=sunburst_path,
                            values='count',
                            color='count',
                            color_continuous_scale=color_schemes[selected_scheme] if selected_scheme != 'Rainbow' else None,
                            color_discrete_sequence=color_schemes[selected_scheme] if selected_scheme == 'Rainbow' else None,
                            hover_data={'count': True}
                        )
                        
                        fig_sunburst.update_traces(
                            textinfo="label+percent parent",
                            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent}<extra></extra>'
                        )
                        
                        fig_sunburst.update_layout(
                            title=dict(
                                text=f"Hierarchical Distribution: {' → '.join(sunburst_path)}",
                                font=dict(size=18, color='#2c3e50'),
                                x=0.5
                            ),
                            font=dict(size=12, family='Inter, Helvetica Neue, Arial'),
                            height=800,
                            margin=dict(l=10, r=10, t=80, b=10),
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig_sunburst, use_container_width=True)
                        
                        # Summary statistics
                        total_segments = len(grouped)
                        total_items = grouped['count'].sum()
                        st.markdown(f"**Summary:** {total_segments} segments representing {total_items} items")
                    else:
                        st.warning("No data available for the selected hierarchy.")
    
    with tab4:
        st.header("🎨 Color Space Aberration Analysis")
        st.markdown("""
        This section analyzes color measurement aberrations in the L*a*b*C*h color space.
        It detects various types of errors including missing values, out-of-range measurements, and statistical outliers.
        """)
        
        # Use the same dataset for color analysis
        if os.path.exists(FILE_PATH):
            # Use the already loaded df for color analysis
            color_df = df
            st.success(f"Using liquid lipstick dataset for color analysis: {len(color_df)} products")
            
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
                    
                    st.subheader(f"📊 Analysis Results for: {selected_family}")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Products", results['total_products'])
                    with col2:
                        st.metric("Total Issues", results['summary']['total_issues'])
                    with col3:
                        st.metric("Affected Rows", results['summary']['critical_rows_count'])
                    with col4:
                        st.metric("Affected %", f"{results['summary']['percentage_affected']:.1f}%")
                    
                    if results['summary']['total_issues'] > 0:
                        st.subheader("🔍 Detailed Aberration Breakdown")
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
                            summary_display = summary_df.copy()
                            for col in summary_display.columns:
                                if summary_display[col].dtype == 'object':
                                    summary_display[col] = summary_display[col].astype(str)
                            st.dataframe(summary_display, use_container_width=True)
                            if results['critical_rows']:
                                st.subheader("🚨 Critical Aberrant Rows")
                                display_cols = ['FORMULA', 'PRODUCT NAME', 'COLOR FAMILY MED LIP', 'L* BULK', 'a* BULK', 'b* BULK', 'C* BULK', 'h BULK']
                                available_cols = [col for col in display_cols if col in color_df.columns]
                                critical_df = color_df.loc[results['critical_rows'], available_cols].copy()
                                aberration_details = []
                                for idx in results['critical_rows']:
                                    details = []
                                    for col, aberrations in results['aberrations'].items():
                                        for aberration_type, indices in aberrations.items():
                                            if idx in indices and aberration_type != 'complete_nan_sets':
                                                details.append(f"{col}: {aberration_type.replace('_', ' ')}")
                                    aberration_details.append("; ".join(details))
                                critical_df['Aberration Details'] = aberration_details
                                if 'COLOR FAMILY MED LIP' in critical_df.columns:
                                    critical_df = critical_df.sort_values('COLOR FAMILY MED LIP')
                                display_df = critical_df.copy()
                                for col in display_df.columns:
                                    if display_df[col].dtype == 'object':
                                        display_df[col] = display_df[col].astype(str)
                                st.dataframe(display_df, use_container_width=True)
                                csv_data = critical_df.to_csv(index=True).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Critical Rows CSV",
                                    data=csv_data,
                                    file_name=f"color_aberrations_{selected_family.replace(' ', '_')}.csv",
                                    mime='text/csv'
                                )
                                st.subheader("📈 Aberration Distribution")
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
                        complete_nan_info = []
                        for col, aberrations in results['aberrations'].items():
                            if aberrations['complete_nan_sets']:
                                complete_nan_info.append(f"{col}: {len(aberrations['complete_nan_sets'])} complete NaN sets")
                        if complete_nan_info:
                            st.info("ℹ️ Complete NaN sets (acceptable missing data):\n" + "\n".join(complete_nan_info))
            else:
                st.warning("No 'COLOR FAMILY MED LIP' column found in the dataset.")
        else:
            st.error(f"Dataset not found at: {FILE_PATH}")
            st.info("Please ensure the file exists at the specified path.")
    
    with tab5:
        st.header("🔍 Text Similarity Analysis")
        st.markdown("""
        This section analyzes textual columns to detect similar values using **difflib.SequenceMatcher**.
        It finds potential data quality issues like:
        - Extra spaces: "YSL" vs "YSL "
        - Case variations: "Dior" vs "DIOR"
        - Typos: "L'Oreal" vs "L'Oréal"
        
        **Shows exact row numbers** where similar values appear and **highlights the differences**.
        """)
        
        if len(text_cols) > 0:
            # Column selection
            selected_text_col = st.selectbox(
                "Select a text column to analyze:",
                options=text_cols,
                index=0
            )
            
            # Similarity threshold
            similarity_threshold = st.slider(
                "Similarity threshold (higher = more strict):",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Values closer to 1.0 will only find very similar strings"
            )
            
            if st.button("🔎 Analyze Text Similarities", type="primary"):
                with st.spinner('Analyzing text similarities...'):
                    # Get unique values from the selected column
                    unique_values = df[selected_text_col].dropna().unique()
                    
                    # Function to calculate similarity ratio
                    def similarity_ratio(a, b):
                        return SequenceMatcher(None, str(a).lower().strip(), str(b).lower().strip()).ratio()
                    
                    # Find similar pairs
                    similar_pairs = []
                    processed = set()
                    
                    for i, val1 in enumerate(unique_values):
                        if val1 in processed:
                            continue
                            
                        similar_group = [val1]
                        for j, val2 in enumerate(unique_values[i+1:], i+1):
                            if val2 in processed:
                                continue
                                
                            similarity = similarity_ratio(val1, val2)
                            if similarity >= similarity_threshold and val1 != val2:
                                similar_group.append(val2)
                                processed.add(val2)
                        
                        if len(similar_group) > 1:
                            # Calculate counts and row numbers for each value in the group
                            group_info = []
                            for val in similar_group:
                                mask = df[selected_text_col] == val
                                count = mask.sum()
                                row_numbers = df[mask].index.tolist()
                                group_info.append((val, count, row_numbers))
                            
                            similar_pairs.append(group_info)
                            processed.add(val1)
                    
                    if similar_pairs:
                        st.success(f"🔍 Found {len(similar_pairs)} groups of similar values!")
                        
                        # Display results
                        for idx, group in enumerate(similar_pairs, 1):
                            with st.expander(f"Group {idx}: {len(group)} similar values"):
                                # Create a DataFrame for this group
                                group_df = pd.DataFrame(group, columns=['Value', 'Count', 'Row_Numbers'])
                                group_df = group_df.sort_values('Count', ascending=False)
                                
                                # Show the group with row numbers
                                display_df = group_df.copy()
                                display_df['Row_Numbers'] = display_df['Row_Numbers'].apply(lambda x: ', '.join(map(str, x[:10])) + ('...' if len(x) > 10 else ''))
                                st.dataframe(display_df, use_container_width=True)
                                
                                # Suggest the most common value as canonical
                                most_common = group_df.iloc[0]
                                total_count = group_df['Count'].sum()
                                st.info(f"💡 **Suggestion:** Use '{most_common['Value']}' as the canonical value ({total_count} total occurrences)")
                                
                                # Show detailed analysis for each value
                                st.markdown("**Detailed Analysis:**")
                                canonical = most_common['Value']
                                
                                for _, row in group_df.iterrows():
                                    val = row['Value']
                                    if val != canonical:
                                        sim_score = similarity_ratio(canonical, val)
                                        
                                        # Highlight differences
                                        def highlight_differences(s1, s2):
                                            if s1 == s2:
                                                return f"'{s1}' (identical)"
                                            
                                            # Check for common differences
                                            s1_clean = s1.strip().lower()
                                            s2_clean = s2.strip().lower()
                                            
                                            differences = []
                                            if s1 != s1.strip() or s2 != s2.strip():
                                                differences.append("whitespace")
                                            if s1.lower() != s1 or s2.lower() != s2:
                                                differences.append("case")
                                            if s1_clean != s2_clean:
                                                differences.append("content")
                                            
                                            diff_str = ", ".join(differences) if differences else "unknown"
                                            return f"'{s1}' vs '{s2}' (diff: {diff_str})"
                                        
                                        diff_analysis = highlight_differences(val, canonical)
                                        st.write(f"• **{sim_score:.1%} similar:** {diff_analysis}")
                                        st.write(f"  📍 Rows: {', '.join(map(str, row['Row_Numbers'][:5]))}{'...' if len(row['Row_Numbers']) > 5 else ''}")
                        
                        # Summary statistics
                        total_groups = len(similar_pairs)
                        total_similar_values = sum(len(group) for group in similar_pairs)
                        total_affected_rows = sum(sum(count for _, count, _ in group) for group in similar_pairs)
                        
                        st.markdown("---")
                        st.markdown("### 📊 Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Similar Groups", total_groups)
                        with col2:
                            st.metric("Similar Values", total_similar_values)
                        with col3:
                            st.metric("Affected Rows", total_affected_rows)
                        
                        # Data cleaning suggestions
                        st.markdown("### 🧽 Data Cleaning Suggestions")
                        st.markdown("""
                        **Recommended actions:**
                        1. **Standardize values**: Replace similar values with their canonical form
                        2. **Trim whitespace**: Remove leading/trailing spaces from all text values
                        3. **Normalize case**: Consider standardizing to Title Case or UPPER CASE
                        4. **Validate entries**: Review unusual values that might be typos
                        """)
                        
                    else:
                        st.success("🎉 No similar values found! Your text data appears to be clean.")
                        st.info(f"Analyzed {len(unique_values)} unique values in '{selected_text_col}' column.")
        else:
            st.warning("No text columns available for similarity analysis.")
            st.info("Text similarity analysis requires columns with textual data.")
