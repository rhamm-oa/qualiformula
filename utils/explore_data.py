import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def create_cosmetic_dashboard(csv_file_path):
    """
    Creates a comprehensive dashboard for cosmetic products data analysis
    """
    
    # Load the data
    print("Loading data...")
    df = pd.read_csv(csv_file_path, sep=';')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Initialize HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>L'Oréal Cosmetic Products Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .header { background: linear-gradient(135deg, #2649B2, #4A74F3); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; text-align: center; }
            .section { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .metric-card { display: inline-block; background: linear-gradient(135deg, #8E7DE3, #9D5CE6); color: white; padding: 20px; margin: 10px; border-radius: 10px; text-align: center; min-width: 150px; }
            .metric-value { font-size: 28px; font-weight: bold; }
            .metric-label { font-size: 14px; opacity: 0.9; }
            .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .chart-container { margin: 20px 0; }
            h1, h2, h3 { color: #2649B2; }
            .filter-info { background-color: #D4D9F0; padding: 15px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎨 L'Oréal Cosmetic Products Dashboard</h1>
            <p>Comprehensive Analysis of Product Data & Quality Metrics</p>
        </div>
    """
    
    # Basic Dataset Overview
    total_products = len(df)
    brands = df['BRAND'].nunique() if 'BRAND' in df.columns else 0
    franchises = df['FRANCHISE'].nunique() if 'FRANCHISE' in df.columns else 0
    inhouse_count = len(df[df['INHOUSE / COMPETITOR'].str.contains('INHOUSE', case=False, na=False)]) if 'INHOUSE / COMPETITOR' in df.columns else 0
    competitor_count = len(df[df['INHOUSE / COMPETITOR'].str.contains('COMPETITOR', case=False, na=False)]) if 'INHOUSE / COMPETITOR' in df.columns else 0
    
    html_content += f"""
        <div class="section">
            <h2>📊 Dataset Overview</h2>
            <div class="metric-card">
                <div class="metric-value">{total_products:,}</div>
                <div class="metric-label">Total Products</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{brands}</div>
                <div class="metric-label">Brands</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{franchises}</div>
                <div class="metric-label">Franchises</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{inhouse_count:,}</div>
                <div class="metric-label">In-House Products</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{competitor_count:,}</div>
                <div class="metric-label">Competitor Products</div>
            </div>
        </div>
    """
    
    # Missing Data Analysis
    missing_data = df.isnull().sum()
    missing_percentages = (missing_data / len(df) * 100).round(2)
    critical_missing = missing_percentages[missing_percentages > 50]
    
    # Missing Formula Analysis
    formula_missing = 0
    if 'FORMULA' in df.columns:
        formula_missing = df['FORMULA'].isnull().sum()
    
    html_content += f"""
        <div class="section">
            <h2>⚠️ Data Quality Alerts</h2>
            <div class="warning">
                <strong>Missing Formulas:</strong> {formula_missing:,} products ({(formula_missing/total_products*100):.1f}%)
            </div>
            <div class="warning">
                <strong>Critical Missing Data:</strong> {len(critical_missing)} columns have >50% missing values
            </div>
    """
    
    if len(critical_missing) > 0:
        html_content += "<ul>"
        for col, pct in critical_missing.head(10).items():
            html_content += f"<li>{col}: {pct}% missing</li>"
        html_content += "</ul>"
    
    html_content += "</div>"
    
    # Create Missing Data Heatmap
    missing_data_top = missing_percentages.nlargest(20)
    
    fig_missing = go.Figure(data=go.Bar(
        x=missing_data_top.values,
        y=missing_data_top.index,
        orientation='h',
        marker_color=['#2649B2', '#4A74F3', '#8E7DE3', '#9D5CE6', '#D4D9F0'] * 4,
        text=[f'{val}%' for val in missing_data_top.values],
        textposition='auto'
    ))
    
    fig_missing.update_layout(
        title="Top 20 Columns with Missing Data",
        xaxis_title="Missing Percentage (%)",
        yaxis_title="Columns",
        height=600,
        template="plotly_white"
    )
    
    html_content += f"""
        <div class="section">
            <h2>🔍 Missing Data Analysis</h2>
            <div class="chart-container">
                <div id="missing-data-chart"></div>
            </div>
        </div>
        <script>
            Plotly.newPlot('missing-data-chart', {fig_missing.to_json()});
        </script>
    """
    
    # Color Analysis (L*, a*, b* values)
    color_columns = [col for col in df.columns if any(x in col for x in ['L*', 'a*', 'b*', 'C*', 'h°']) and 'BULK' not in col]
    
    if color_columns:
        # Outlier detection for color values
        outliers_summary = {}
        
        for col in color_columns[:10]:  # Analyze first 10 color columns
            if df[col].dtype in ['float64', 'int64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outliers_summary[col] = len(outliers)
        
        # Create outliers chart
        if outliers_summary:
            fig_outliers = go.Figure(data=go.Bar(
                x=list(outliers_summary.keys()),
                y=list(outliers_summary.values()),
                marker_color=['#2649B2', '#4A74F3', '#8E7DE3', '#9D5CE6', '#6C8BE0'] * 2,
                text=list(outliers_summary.values()),
                textposition='auto'
            ))
            
            fig_outliers.update_layout(
                title="Outliers Detected in Color Measurements",
                xaxis_title="Color Parameters",
                yaxis_title="Number of Outliers",
                height=400,
                template="plotly_white",
                xaxis_tickangle=-45
            )
            
            html_content += f"""
                <div class="section">
                    <h2>🎯 Outlier Detection</h2>
                    <div class="chart-container">
                        <div id="outliers-chart"></div>
                    </div>
                </div>
                <script>
                    Plotly.newPlot('outliers-chart', {fig_outliers.to_json()});
                </script>
            """
    
    # Brand Distribution Analysis
    if 'BRAND' in df.columns:
        brand_counts = df['BRAND'].value_counts().head(10)
        
        fig_brands = go.Figure(data=go.Pie(
            labels=brand_counts.index,
            values=brand_counts.values,
            hole=0.4,
            marker_colors=['#2649B2', '#4A74F3', '#8E7DE3', '#9D5CE6', '#D4D9F0', '#6C8BE0', '#B55CE6'] * 2
        ))
        
        fig_brands.update_layout(
            title="Top 10 Brands Distribution",
            height=500,
            template="plotly_white"
        )
        
        html_content += f"""
            <div class="section">
                <h2>🏷️ Brand Analysis</h2>
                <div class="chart-container">
                    <div id="brands-chart"></div>
                </div>
            </div>
            <script>
                Plotly.newPlot('brands-chart', {fig_brands.to_json()});
            </script>
        """
    
    # In-house vs Competitor Analysis
    if 'INHOUSE / COMPETITOR' in df.columns:
        inhouse_competitor = df['INHOUSE / COMPETITOR'].value_counts()
        
        fig_inhousecomp = go.Figure(data=go.Bar(
            x=inhouse_competitor.index,
            y=inhouse_competitor.values,
            marker_color=['#2649B2', '#9D5CE6'],
            text=inhouse_competitor.values,
            textposition='auto'
        ))
        
        fig_inhousecomp.update_layout(
            title="In-house vs Competitor Products",
            xaxis_title="Product Type",
            yaxis_title="Number of Products",
            height=400,
            template="plotly_white"
        )
        
        html_content += f"""
            <div class="section">
                <h2>🏢 In-house vs Competitor Analysis</h2>
                <div class="chart-container">
                    <div id="inhouse-competitor-chart"></div>
                </div>
            </div>
            <script>
                Plotly.newPlot('inhouse-competitor-chart', {fig_inhousecomp.to_json()});
            </script>
        """
    
    # Spectral Data Analysis (wavelength columns)
    wavelength_cols = [col for col in df.columns if col.split()[0].isdigit() and any(x in col for x in ['WB', 'BB'])]
    
    if wavelength_cols:
        # Sample spectral analysis for first few products
        sample_size = min(20, len(df))
        sample_df = df.head(sample_size)
        
        fig_spectral = go.Figure()
        
        colors = ['#2649B2', '#4A74F3', '#8E7DE3', '#9D5CE6', '#6C8BE0', '#B55CE6'] * 4
        
        for i, idx in enumerate(sample_df.index[:6]):
            wavelengths = []
            values = []
            for col in wavelength_cols[:10]:  # First 10 wavelength columns
                try:
                    wavelength = int(col.split()[0])
                    value = sample_df.loc[idx, col]
                    if pd.notna(value):
                        wavelengths.append(wavelength)
                        values.append(value)
                except:
                    continue
            
            if wavelengths and values:
                fig_spectral.add_trace(go.Scatter(
                    x=wavelengths,
                    y=values,
                    mode='lines+markers',
                    name=f'Product {idx}',
                    line=dict(color=colors[i % len(colors)])
                ))
        
        fig_spectral.update_layout(
            title="Sample Spectral Data Analysis",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Reflectance Value",
            height=500,
            template="plotly_white"
        )
        
        html_content += f"""
            <div class="section">
                <h2>🌈 Spectral Data Analysis</h2>
                <div class="chart-container">
                    <div id="spectral-chart"></div>
                </div>
            </div>
            <script>
                Plotly.newPlot('spectral-chart', {fig_spectral.to_json()});
            </script>
        """
    
    # Color Family Analysis
    if 'COLOR FAMILY MED LIP' in df.columns:
        color_family_counts = df['COLOR FAMILY MED LIP'].value_counts().head(10)
        
        fig_color_family = go.Figure(data=go.Bar(
            x=color_family_counts.values,
            y=color_family_counts.index,
            orientation='h',
            marker_color=['#2649B2', '#4A74F3', '#8E7DE3', '#9D5CE6', '#D4D9F0', '#6C8BE0', '#B55CE6'] * 2,
            text=color_family_counts.values,
            textposition='auto'
        ))
        
        fig_color_family.update_layout(
            title="Color Family Distribution",
            xaxis_title="Number of Products",
            yaxis_title="Color Family",
            height=500,
            template="plotly_white"
        )
        
        html_content += f"""
            <div class="section">
                <h2>🎨 Color Family Analysis</h2>
                <div class="chart-container">
                    <div id="color-family-chart"></div>
                </div>
            </div>
            <script>
                Plotly.newPlot('color-family-chart', {fig_color_family.to_json()});
            </script>
        """
    
    # Statistical Summary for Key Numeric Columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    key_numeric_cols = [col for col in numeric_cols if any(x in col for x in ['L*', 'a*', 'b*', 'C*']) and 'BULK' not in col][:5]
    
    if key_numeric_cols:
        stats_summary = df[key_numeric_cols].describe().round(2)
        
        html_content += """
            <div class="section">
                <h2>📈 Statistical Summary</h2>
                <div class="filter-info">
                    <strong>Key Color Measurements Statistics</strong>
                </div>
                <table border="1" style="width:100%; border-collapse: collapse; margin-top: 15px;">
                    <thead style="background-color: #2649B2; color: white;">
                        <tr>
                            <th style="padding: 10px;">Statistic</th>
        """
        
        for col in key_numeric_cols:
            html_content += f"<th style='padding: 10px;'>{col}</th>"
        
        html_content += "</tr></thead><tbody>"
        
        for stat in stats_summary.index:
            html_content += f"<tr><td style='padding: 8px; background-color: #f8f9fa;'><strong>{stat}</strong></td>"
            for col in key_numeric_cols:
                html_content += f"<td style='padding: 8px; text-align: center;'>{stats_summary.loc[stat, col]}</td>"
            html_content += "</tr>"
        
        html_content += "</tbody></table></div>"
    
    # Data Quality Summary
    html_content += f"""
        <div class="section">
            <h2>✅ Data Quality Summary</h2>
            <div class="filter-info">
                <h3>Key Findings:</h3>
                <ul>
                    <li><strong>Total Records:</strong> {total_products:,}</li>
                    <li><strong>Columns:</strong> {len(df.columns)}</li>
                    <li><strong>Missing Formulas:</strong> {formula_missing:,} ({(formula_missing/total_products*100):.1f}%)</li>
                    <li><strong>Columns with >50% Missing:</strong> {len(critical_missing)}</li>
                    <li><strong>In-house Products:</strong> {inhouse_count:,}</li>
                    <li><strong>Competitor Products:</strong> {competitor_count:,}</li>
                </ul>
            </div>
        </div>
    """
    
    # Recommendations
    html_content += """
        <div class="section">
            <h2>💡 Recommendations</h2>
            <div class="warning">
                <h3>Priority Actions:</h3>
                <ol>
                    <li><strong>Formula Completion:</strong> Address missing formula data for better product tracking</li>
                    <li><strong>Data Standardization:</strong> Review columns with high missing percentages</li>
                    <li><strong>Outlier Investigation:</strong> Examine detected outliers in color measurements</li>
                    <li><strong>Quality Control:</strong> Implement data validation for future entries</li>
                    <li><strong>Spectral Data:</strong> Ensure complete wavelength measurements for all products</li>
                </ol>
            </div>
        </div>
    """
    
    html_content += """
        <div class="section" style="text-align: center; background: linear-gradient(135deg, #2649B2, #4A74F3); color: white;">
            <h2>🎯 Dashboard Generated Successfully</h2>
            <p>This comprehensive analysis provides insights into your cosmetic products database.</p>
            <p><em>Generated by L'OréalGPT - Your AI Assistant for Data Analysis</em></p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    output_file = "loreal_cosmetic_dashboard.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Dashboard created successfully: {output_file}")
    print(f"📊 Analyzed {total_products:,} products across {len(df.columns)} columns")
    print(f"⚠️  Found {formula_missing:,} missing formulas")
    print(f"🔍 Detected {len(critical_missing)} columns with >50% missing data")
    
    return output_file

# Usage
if __name__ == "__main__":
    # Replace 'your_file.csv' with the actual path to your CSV file
    csv_file_path = '/home/user/qualiformula/data/lipsticks/solid/Solid_lipstick_database.csv'
    
    try:
        dashboard_file = create_cosmetic_dashboard(csv_file_path)
        print(f"\n🎉 Open {dashboard_file} in your web browser to view the dashboard!")
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        print("Please ensure your CSV file path is correct and the file is accessible.")