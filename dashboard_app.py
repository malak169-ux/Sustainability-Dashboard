import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import warnings

# Ignore Matplotlib and Seaborn warnings related to styles
warnings.filterwarnings("ignore")

# ====================================================================
# ⚠️ STREAMLIT PAGE CONFIGURATION (MUST BE FIRST STREAMLIT COMMAND)
# ====================================================================
st.set_page_config(page_title="Sustainability Dashboard", layout="wide") 

# ====================================================================
# 0. Helper Functions and Data Loading (Initial Data Prep)
# ====================================================================
@st.cache_data
def load_raw_data():
    try:
        df = pd.read_csv('Sustainability_Raw_Data.csv')
        # Standardize column names for safe processing
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
        # Ensure 'year' is numeric and handle potential errors
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
            
        
        if 'certification' in df.columns:
            df['certification'] = df['certification'].fillna('None').astype(str)

        return df
    except FileNotFoundError:
        st.error("❌ Error: 'Sustainability_Raw_Data.csv' not found. Please ensure it's in the same folder.")
        return pd.DataFrame() 
    except Exception as e:
        st.error(f"❌ An error occurred during data loading: {e}")
        return pd.DataFrame() 

# --- Helper Function for Safe KPI Calculation ---
def safe_kpi_calc(series, func, rounding=2):
    """Calculates KPI safely, returning 'N/A' on error or empty data."""
    if series.empty or not pd.api.types.is_numeric_dtype(series):
        return "N/A"
    try:
        result = func(series.dropna())
        if isinstance(result, (int, float)):
            return round(result, rounding)
        return "N/A"
    except Exception:
        return "N/A"

# ====================================================================
# 1. Visualization Functions (Returning fig instead of plt.show())
# ====================================================================

# Helper function to create a small figure (for better dashboard fit)

def create_figure(df, title, func, figsize=(6, 3.5)): 
    if df.empty: 
        return None
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        
        fig.patch.set_alpha(0.0) 
        
         
        ax.patch.set_alpha(0.0) 
        
        func(fig, ax, df) 
        ax.set_title(title, fontsize=12, color='darkgreen', fontweight='bold', pad=10) 
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        return fig
    except Exception as e:
        # st.error(f"Error in creating figure: {e}") 
        return None
# 1. Top 10 Sustainable Brands
def plot_top_brands(df_brands): 
    
    return create_figure(df_brands, "Top 10 Sustainable Brands", figsize=(7, 4), 
        func=lambda fig, ax, df: (
            sns.set_theme(style="whitegrid"),
            norm := plt.Normalize(df["sustainability_rating"].min(), df["sustainability_rating"].max()),
            colors := plt.cm.Greens_r(norm(df["sustainability_rating"])),
            bars := ax.bar(df["brand_name"], df["sustainability_rating"], color=colors, edgecolor="#2E8B57", linewidth=1.0),
            [ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkgreen') for bar in bars], # تصغير حجم نص القيم
            ax.set_xlabel("Brand Name", fontsize=10),
            ax.set_ylabel("Avg. Rating", fontsize=10),
            rotation_angle := 45 if max([len(str(label)) for label in df["brand_name"]]) > 8 else 30,
            ax.set_xticklabels(df["brand_name"], rotation=rotation_angle, ha='right', fontsize=8),
            sns.despine(ax=ax)
        )
    )

# 2. Top 5 Product lines
def plot_top_product_lines(df_categories): 
    return create_figure(df_categories, "Top 5 Product Lines", 
        func=lambda fig, ax, df: (
            sns.set_theme(style="whitegrid"),
            norm := plt.Normalize(df["sustainability_rating"].min(), df["sustainability_rating"].max()),
            colors := plt.cm.Greens_r(norm(df["sustainability_rating"])),
            bars := ax.bar(df["product_line"], df["sustainability_rating"], color=colors, edgecolor="#2E8B57", linewidth=1.0),
            [ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen') for bar in bars],
            ax.set_xlabel("Product Line", fontsize=12),
            ax.set_ylabel("Avg. Rating", fontsize=12),
            rotation_angle := 45 if max([len(str(label)) for label in df["product_line"]]) > 10 else 30,
            ax.set_xticklabels(df["product_line"], rotation=rotation_angle, ha='right', fontsize=10),
            sns.despine(ax=ax)
        )
    )

# 3. Top 5 Countries
def plot_top_countries(df_countries):
    return create_figure(df_countries, "Top 5 Countries by Avg. Rating",figsize=(7, 5), 
        func=lambda fig, ax, df: (
            sns.set_theme(style="whitegrid"),
            norm := plt.Normalize(df["sustainability_rating"].min(), df["sustainability_rating"].max()),
            colors := plt.cm.Greens_r(norm(df["sustainability_rating"])),
            bars := ax.bar(df["country_name"], df["sustainability_rating"], color=colors, edgecolor="#2E8B57", linewidth=1.0),
            [ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen') for bar in bars],
            ax.set_xlabel("Country", fontsize=12),
            ax.set_ylabel("Avg. Rating", fontsize=12),
            ax.set_xticklabels(df["country_name"], rotation=25, ha='right', fontsize=10),
            sns.despine(ax=ax)
        )
    )

# 4. Number of Certifications per Product line
def plot_certifications_per_product(df_cert_counts): 
    return create_figure(df_cert_counts, "Certifications per Product Line", figsize=(5, 3.5), # 
        func=lambda fig, ax, df: (
            sns.barplot(data=df, x='product_line', y='num_certification', palette='Greens_r', ax=ax),
            [ax.text(index, value + 0.5, str(value), ha='center', va='bottom', fontsize=8, color='#1B5E20', fontweight='bold') for index, value in enumerate(df['num_certification'])],
            ax.set_xlabel('Product Line', fontsize=10),
            ax.set_ylabel('Number of Certifications', fontsize=10),
            ax.tick_params(axis='x', rotation=30, labelsize=8)
        )
    )

# 5. Average Environmental Metrics per Product Line
def plot_environmental_metrics(df_melted):
    return create_figure(df_melted, "Environmental Metrics by Product Line", figsize=(7, 4), 
        func=lambda fig, ax, df: (
            sns.barplot(data=df, x='product_line', y='Average Value', hue='Metric', palette=['#1B5E20', '#388E3C', '#66BB6A'], ax=ax),
            [ax.bar_label(container, fmt='%.1f', label_type='edge', fontsize=7, color='#1B5E20', padding=2) for container in ax.containers],
            ax.set_xlabel('Product Line', fontsize=10),
            ax.set_ylabel('Average Value', fontsize=10),
            ax.tick_params(axis='x', rotation=30, labelsize=8),
            ax.legend(title='Metric', title_fontsize=9, fontsize=8, loc='upper right', bbox_to_anchor=(1.35, 1))
        )
    )

# 6. Sustainability Improvements Over Time
def plot_time_improvement(df_time_avg): 
    return create_figure(df_time_avg, "Sustainability Rating Over Time", 
        func=lambda fig, ax, df: (
            sns.lineplot(data=df, x='year', y='avg_rating', marker='o', color='#2E7D32', linewidth=2.0, ax=ax),
            [ax.text(x, y + 0.002, f"{y:.2f}", ha='center', fontsize=8, fontweight='bold', color='#1B5E20') for x, y in zip(df['year'], df['avg_rating'])],
            ax.set_xlabel("Year", fontsize=10),
            ax.set_ylabel("Avg. Rating", fontsize=10),
            ax.set_xticks(df['year']),
            ax.tick_params(axis='x', rotation=30, labelsize=8)
        )
    )

# 7. Average Sustainability Rating by Target Audience
def plot_audience_sustainability(df_audience_sus): 
    return create_figure(df_audience_sus, "Avg. Rating by Target Audience", 
        func=lambda fig, ax, df: (
            sns.barplot(data=df, x='target_audience', y='sustainability_rating', palette=['#1B5E20', '#2E7D32', '#66BB6A', '#A5D6A7'], ax=ax),
            [ax.text(index, row['sustainability_rating'] + 0.01, f"{row['sustainability_rating']:.3f}", ha='center', fontsize=10, fontweight='bold', color='#1B5E20') for index, row in df.iterrows()],
            ax.set_xlabel('Target Audience', fontsize=12),
            ax.set_ylabel('Avg. Rating', fontsize=12),
            ax.tick_params(axis='x', rotation=15, labelsize=8)
        )
    )

# 8. Average Sustainability Rating by Material Status (Donut Chart)
def plot_material_status(df_material_sus): 
    if df_material_sus.empty or len(df_material_sus) < 2 or 'sustainability_rating' not in df_material_sus.columns:
        return None
        
    fig, ax = plt.subplots(figsize=(4, 4)) # Pie/Donut Chart

    fig.patch.set_alpha(0.0)
    
    
    ax.patch.set_alpha(0.0) 
    
    colors = ['#1B5E20', '#4CAF50', '#A5D6A7']
    colors_to_use = colors[:len(df_material_sus)]
    

    wedges, texts = ax.pie(df_material_sus['sustainability_rating'], 
                           labels=df_material_sus['label'], 
                           startangle=90, colors=colors_to_use, 
                           textprops={'color': '#1B5E20', 'fontsize': 9, 'fontweight': 'bold'}) 

    centre_circle = plt.Circle((0, 0), 0.70, fc='none') 
    fig.gca().add_artist(centre_circle)
    
    ax.axis('off') 

    ax.set_title('Avg. Rating by Material Status', fontsize=12, color='#1B5E20', fontweight='bold', pad=10)
    plt.tight_layout()
    return fig

# 9. Eco-friendly vs Non Eco-friendly Brands #(Pie Chart)
def plot_eco_friendly_counts(eco_counts_series): 
    if eco_counts_series.empty or len(eco_counts_series) < 2: return None

    fig, ax = plt.subplots(figsize=(4, 4)) # Pie/Donut Chart
    fig.patch.set_alpha(0.0)
    ax.pie(eco_counts_series.values, 
           labels=['Non Eco-friendly', 'Eco-friendly'][:len(eco_counts_series)], 
           autopct='%1.1f%%', 
           colors=['#A5D6A7', '#1B5E20'][:len(eco_counts_series)], 
           startangle=90,
           textprops={'fontsize': 11}) 
    ax.set_title("Eco-friendly vs Non Eco-friendly Mfg", fontsize=12, color='darkgreen', fontweight='bold', pad=10)
    plt.tight_layout()
    return fig

# 10. Relationship Between Price and Sustainability Rating (Scatter Plot)
def plot_price_vs_sustainability(df_price_sus): 
    return create_figure(df_price_sus, "Price vs. Sustainability Rating", figsize=(7, 4),
        func=lambda fig, ax, df: (
            sns.scatterplot(data=df, x="average_price", y="sustainability_rating", hue="brand_category", palette="Greens_r", alpha=0.7, s=50, edgecolor="black", ax=ax),
            ax.set_xlabel("Average Price", fontsize=12),
            ax.set_ylabel("Sustainability Rating", fontsize=12),
            ax.legend(title="Brand Category", title_fontsize=8, fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
        )
    )

# 11. Impact of Certification on Sustainability Rating
def plot_certification_impact(df_certification_avg): 
    return create_figure(df_certification_avg, "Impact of Certification on Rating", figsize=(7, 4), 
        func=lambda fig, ax, df: (
            green_palette := sns.color_palette("Greens", n_colors=len(df)),
            barplot := sns.barplot(data=df, x="certification", y="avg_rating", palette=green_palette, ax=ax),
            [barplot.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8, fontweight='medium', color='black', xytext=(0, 3), textcoords='offset points') for p in barplot.patches],
            ax.set_xlabel("Certification", fontsize=10),
            ax.set_ylabel("Avg. Rating", fontsize=10),
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        )
    )

# 12. Market Trend vs Sustainability Rating (Donut Chart)
def plot_market_trend(df_trend_avg): 
    if df_trend_avg.empty or len(df_trend_avg) < 2 or 'sustainability_rating' not in df_trend_avg.columns:
        return None
        
    fig, ax = plt.subplots(figsize=(4, 4)) # Pie/Donut Chart
    
    
    fig.patch.set_alpha(0.0)
    
    
    # ax.patch.set_alpha(0.0) 
    # ax.set_facecolor('none')  

    colors = sns.color_palette("Greens", n_colors=len(df_trend_avg))
    
    wedges, texts, autotexts = ax.pie(
        df_trend_avg["sustainability_rating"],
        labels=df_trend_avg["market_trend"],
        autopct=lambda p: f'{p:.1f} ({p*sum(df_trend_avg["sustainability_rating"])/100:.2f})', 
        startangle=140,
        colors=colors,
        pctdistance=0.85, 
        textprops={"fontsize": 9, "color": "black"}
    )
    
   
    centre_circle = plt.Circle((0, 0), 0.70, fc="#F3F9E8") 
    fig.gca().add_artist(centre_circle)
    
   
    ax.axis('off')

    ax.set_title("Market Trend vs Sustainability Rating", fontsize=12, color="green", fontweight="bold", pad=10)
    plt.tight_layout()
    return fig
# ====================================================================
# 2. Main Streamlit App
# ====================================================================

def main():
    # 1. Load Raw Data
    df = load_raw_data()

    if df.empty:
        return

    # --- Header with Logos ---
    
    col_logo1, col_title, col_logo2 = st.columns([1, 4, 1])
    with col_logo1:
        st.image("logo_ministry.png", width=100) #  "logo_ministry.png" 
    with col_title:
        st.title("Sustainability Dashboard")
    with col_logo2:
        st.image("logo_project.png", width=100) #  "logo_project.png" 
    st.markdown("---")
    
    # ----------------------------------------------------
    # 2. Filtering (Creates df_filtered)
    # ----------------------------------------------------
    
    st.sidebar.title("Filter") 
    
    # Initialize df_filtered with a copy of the original df
    df_filtered = df.copy() 
    
    # Helper to get unique options and set default
    def get_filter_options(column_name):
        if column_name in df.columns:
            options = sorted(df[column_name].unique().tolist())
            return options, options # Default to all options
        return [], [] # Return empty lists if column not found

    # 1. Countries Filter
    country_options, default_countries = get_filter_options('country_name')
   
    selected_countries = st.sidebar.multiselect(
        "Select Country:", options=country_options, default=default_countries
    )
    if selected_countries:
        df_filtered = df_filtered[df_filtered['country_name'].isin(selected_countries)]
    
    # 2. Years Filter
    year_options, default_years = get_filter_options('year')
 
    selected_years = st.sidebar.multiselect(
        "Select Year:", options=year_options, default=default_years
    )
    if selected_years:
        df_filtered = df_filtered[df_filtered['year'].isin(selected_years)]

    # 3. Certifications Filter
    certification_options, default_certifications = get_filter_options('certification')
   
    selected_certifications = st.sidebar.multiselect(
        "Select Certification:", options=certification_options, default=default_certifications
    )
    if selected_certifications:
        df_filtered = df_filtered[df_filtered['certification'].isin(selected_certifications)]
    
    # 4. Product Lines Filter
    product_line_options, default_product_lines = get_filter_options('product_line')
   
    selected_product_lines = st.sidebar.multiselect(
        "Select Product Line:", options=product_line_options, default=default_product_lines
    )
    if selected_product_lines:
        df_filtered = df_filtered[df_filtered['product_line'].isin(selected_product_lines)]

    # 5. Brands Filter
    brand_options, default_brands = get_filter_options('brand_name')
   
    selected_brands = st.sidebar.multiselect(
        "Select Sustainable Brand:", options=brand_options, default=default_brands
    )
    if selected_brands:
        df_filtered = df_filtered[df_filtered['brand_name'].isin(selected_brands)]

    # Fallback if no data matches filters (for KPIs and charts)
    if df_filtered.empty:
        st.warning("⚠️ No data matches the current filters. Please adjust your selections. Displaying global KPIs and charts with full data as fallback.")
        df_to_use_for_insights = df.copy() # Use full data for insights if filter results in empty
    else:
        df_to_use_for_insights = df_filtered.copy() # Use filtered data for insights


    # ----------------------------------------------------
    # 3. Calculate all 12 Insights based on df_to_use_for_insights
    # ----------------------------------------------------
    available_cols_for_insights = df_to_use_for_insights.columns.tolist()

    # 1. Top 10 Sustainable Brands 
    df_top_brands = pd.DataFrame()
    if 'brand_name' in available_cols_for_insights and 'sustainability_rating' in available_cols_for_insights and not df_to_use_for_insights.empty:
        df_top_brands = (
            df_to_use_for_insights.groupby("brand_name")["sustainability_rating"].mean().round(2).reset_index()
            .sort_values(by="sustainability_rating", ascending=False).head(10))

    # 2. Top 5 Product lines
    df_top_categories = pd.DataFrame()
    if 'product_line' in available_cols_for_insights and 'sustainability_rating' in available_cols_for_insights and not df_to_use_for_insights.empty:
        df_top_categories = (
            df_to_use_for_insights.groupby("product_line")["sustainability_rating"].mean().round(2).reset_index()
            .sort_values(by="sustainability_rating", ascending=False).head(5))

    # 3. Top 5 Countries
    df_top_countries = pd.DataFrame()
    if 'country_name' in available_cols_for_insights and 'sustainability_rating' in available_cols_for_insights and not df_to_use_for_insights.empty:
        df_top_countries = (
            df_to_use_for_insights.groupby("country_name")["sustainability_rating"].mean().round(2).reset_index()
            .sort_values(by="sustainability_rating", ascending=False).head(5))
    
    # 4. Certifications per Product line
    df_category_cert = pd.DataFrame()
    if 'product_line' in available_cols_for_insights and 'certification' in available_cols_for_insights and not df_to_use_for_insights.empty:
        df_category_cert = (
            df_to_use_for_insights.groupby("product_line")["certification"].count().reset_index()
            .rename(columns={"certification": "num_certification"})
            .sort_values(by="num_certification", ascending=False))
    
    # 5. Environmental Metrics
    df_melted = pd.DataFrame()
    env_cols = ['waste_production', 'water_usage', 'carbon_footprint']
    if all(col in available_cols_for_insights for col in env_cols) and 'product_line' in available_cols_for_insights and not df_to_use_for_insights.empty:
        df_avg = df_to_use_for_insights.groupby('product_line')[env_cols].mean().reset_index()
        df_melted = df_avg.melt(id_vars='product_line', value_vars=env_cols, var_name='Metric', value_name='Average Value')
    
    # 6. Time Improvement
    df_avg_time = pd.DataFrame()
    if 'year' in available_cols_for_insights and 'sustainability_rating' in available_cols_for_insights and not df_to_use_for_insights.empty:
        # 'year' is already processed to int in load_raw_data()
        df_avg_time = df_to_use_for_insights.groupby('year')['sustainability_rating'].mean().reset_index(name='avg_rating')
    
    # 7. Audience Sustainability
    audience_sustainability = pd.DataFrame()
    if 'target_audience' in available_cols_for_insights and 'sustainability_rating' in available_cols_for_insights and not df_to_use_for_insights.empty:
        audience_sustainability = df_to_use_for_insights.groupby('target_audience')['sustainability_rating'].mean().round(3).reset_index().sort_values(by='sustainability_rating', ascending=False)
    
    # 8. Material Status
    material_sustainability = pd.DataFrame(columns=['material_status', 'sustainability_rating', 'label'])
    if 'material_status' in available_cols_for_insights and 'sustainability_rating' in available_cols_for_insights and not df_to_use_for_insights.empty:
        material_sustainability = df_to_use_for_insights.groupby('material_status')['sustainability_rating'].mean().round(3).reset_index().sort_values(by='sustainability_rating', ascending=False)
        if not material_sustainability.empty:
             material_sustainability['label'] = (material_sustainability['material_status'] + ' (' + material_sustainability['sustainability_rating'].astype(str) + ')')
    
    # 9. Eco-Friendly Counts
    eco_counts = pd.Series()
    if 'eco_friendly_manufacturing' in available_cols_for_insights and not df_to_use_for_insights.empty:
        eco_counts = df_to_use_for_insights['eco_friendly_manufacturing'].value_counts()
    
    # 10. Price vs Sustainability
    df_price_vs_sus = pd.DataFrame()
    if all(col in available_cols_for_insights for col in ['average_price', 'sustainability_rating', 'brand_category']) and not df_to_use_for_insights.empty:
        df_price_vs_sus = df_to_use_for_insights.copy() 
    
    # 11. Certification Impact
    certification_avg = pd.DataFrame()
    if 'certification' in available_cols_for_insights and 'sustainability_rating' in available_cols_for_insights and not df_to_use_for_insights.empty:
        certification_avg = df_to_use_for_insights.groupby("certification", as_index=False).agg(avg_rating=("sustainability_rating", "mean")).round(3).sort_values("avg_rating", ascending=False)
    
    # 12. Market Trend 
    trend_avg = pd.DataFrame()
    if 'market_trend' in available_cols_for_insights and 'sustainability_rating' in available_cols_for_insights and not df_to_use_for_insights.empty:
        trend_avg = (
            df_to_use_for_insights.groupby("market_trend", as_index=False)["sustainability_rating"]
            .mean()
            .sort_values("sustainability_rating", ascending=False)
        )
    # ----------------------------------------------------


    # ----------------------------------------------------
    # 4. Key Performance Indicators (KPIs) - Now based on df_filtered
    # ----------------------------------------------------
    
    # Calculate KPIs based on df_to_use_for_insights
    avg_price = safe_kpi_calc(df_to_use_for_insights.get('average_price', pd.Series()), np.mean)
    avg_carbon = safe_kpi_calc(df_to_use_for_insights.get('carbon_footprint', pd.Series()), np.mean)
    avg_water = safe_kpi_calc(df_to_use_for_insights.get('water_usage', pd.Series()), np.mean, rounding=0)
    avg_waste = safe_kpi_calc(df_to_use_for_insights.get('waste_production', pd.Series()), np.mean)
    min_sus_rating = safe_kpi_calc(df_to_use_for_insights.get('sustainability_rating', pd.Series()), np.min)
    max_sus_rating = safe_kpi_calc(df_to_use_for_insights.get('sustainability_rating', pd.Series()), np.max)


    # Display KPIs in a single row using st.columns
    kpi_cols = st.columns(6)

    kpi_cols[0].metric(
        label="💰 **AVG PRICE** ",
        value=f"{avg_price:,.2f}" if isinstance(avg_price, (int, float)) else avg_price
    )
    kpi_cols[1].metric(
        label="🏭 **AVG CARBON** ",
        value=f"{avg_carbon:.2f}" if isinstance(avg_carbon, (int, float)) else avg_carbon
    ) 
    kpi_cols[2].metric(
        label="💧 **AVG WATER** ",
        value=f"{avg_water:,.0f}" if isinstance(avg_water, (int, float)) else avg_water
    ) 
    kpi_cols[3].metric(
        label="🗑️ **AVG WASTE** ",
        value=f"{avg_waste:.2f}" if isinstance(avg_waste, (int, float)) else avg_waste
    ) 
    kpi_cols[4].metric(
        label="⭐️ **MIN SUS RATING**",
        value=f"{min_sus_rating}"
    )
    kpi_cols[5].metric(
        label="🌟 **MAX SUS RATING**",
        value=f"{max_sus_rating}"
    )
    
    st.markdown("---")

    # --- Tabs for Organization (6 Tabs) ---
    tab1, tab2, tab3, tab4, tab5 , tab6 = st.tabs([
        " Top Performance", 
        "Geographic & Material Impact", 
        " Trends Over Time", 
        " Environmental Metrics", 
        " Price & Audience", 
        "Certifications "
    ])
    
    # ====================================================
    # Tab 1: Top Performance (Brands, Product Lines, Certifications)
    # ====================================================
    with tab1:
        st.header(" Top Sustainable Performers")
        colA, colB = st.columns([1.5, 1])
        
        with colA:
            st.subheader(" Top 10 Sustainable Brands")
            fig = plot_top_brands(df_top_brands) # Pass the dynamically calculated df_top_brands
            if fig: st.pyplot(fig)
        
        with colB:
            st.subheader(" Top 5 Product Lines")
            fig = plot_top_product_lines(df_top_categories) # Pass the dynamically calculated df_top_categories
            if fig: st.pyplot(fig)
            
        
    # ====================================================
    # Tab 2: Geographic & Material Impact
    # ====================================================
    with tab2:
        st.header(" Geographic & Material Impact Analysis")
        
        colC,  colE = st.columns([2, 1])
        
        with colC:
            st.subheader("Top 5 Countries by Avg. Rating")
            fig = plot_top_countries(df_top_countries) # Pass the dynamically calculated df_top_countries
            if fig: st.pyplot(fig)

        with colE:
            st.subheader("Eco-friendly vs. Non Eco-friendly")
            fig = plot_eco_friendly_counts(eco_counts) # Pass the dynamically calculated eco_counts
            if fig: st.pyplot(fig)

    # ====================================================
    # Tab 3: Trends Over Time
    # ====================================================
    with tab3:
        st.header("Sustainability Trends")
        colF, colG = st.columns([1.5, 1]) 

        with colF:
            st.subheader("Avg. Rating Improvement Over Time")
            fig = plot_time_improvement(df_avg_time) # Pass the dynamically calculated df_avg_time
            if fig: st.pyplot(fig)
            
        with colG:
            st.subheader("Market Trend vs. Avg. Rating")
            fig = plot_market_trend(trend_avg) # Pass the dynamically calculated trend_avg
            if fig: st.pyplot(fig)

    # ====================================================
    # Tab 4: Environmental Metrics
    # ====================================================
    with tab4:
        st.header(" Core Environmental Metrics")
        st.subheader("Average Waste, Water, and Carbon Footprint per Product Line")
        fig = plot_environmental_metrics(df_melted) # Pass the dynamically calculated df_melted
        if fig: st.pyplot(fig)
        
       
        
    # ====================================================
    # Tab 5: Price & Audience
    # ====================================================
    with tab5:
        st.header(" Market and Customer Analysis")
        colI, colJ = st.columns(2)
        
        with colI:
            st.subheader("Price vs. Sustainability Rating")
            fig = plot_price_vs_sustainability(df_price_vs_sus) # Pass the dynamically calculated df_price_vs_sus
            if fig: st.pyplot(fig)
            
        with colJ:
            st.subheader("Avg. Rating by Target Audience")
            fig = plot_audience_sustainability(audience_sustainability) # Pass the dynamically calculated audience_sustainability
            if fig: st.pyplot(fig)
    # ====================================================
    # Tab 6: Certifications 
    # ====================================================
    with tab6:
        st.header(" Certification Analysis")
        
        
        st.subheader("Impact of Certification on Rating")
        fig = plot_certification_impact(certification_avg) # Pass the dynamically calculated certification_avg
        if fig: st.pyplot(fig)
        
        st.markdown("---")
        
       
        st.subheader("Number of Certifications per Product Line")
        fig = plot_certifications_per_product(df_category_cert) # Pass the dynamically calculated df_category_cert
        if fig: st.pyplot(fig)


# Run the main function
if __name__ == "__main__":
    main()