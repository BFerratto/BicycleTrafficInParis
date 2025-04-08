import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image
import requests
from io import BytesIO
import os
import tempfile

# Scikit-learn - Regression & Classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    classification_report
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn import tree

# Joblib for saving/loading models
import joblib

# Imbalanced data handling
from imblearn.over_sampling import RandomOverSampler



@st.cache_data
def load_main_data():
    url = "https://storage.googleapis.com/bicycle-traffic-data/df_processed_1023_0924.csv"
    return pd.read_csv(url)

@st.cache_data
def load_directional_data():
    url = "https://storage.googleapis.com/bicycle-traffic-data/df_processed_one_direction.csv"
    return pd.read_csv(url)

df_main = load_main_data()
df_directional = load_directional_data()

# Page Title & Sidebar
st.title("🚲 Analysis of Bicycle Traffic in Paris")
st.sidebar.title("Contents")
pages = ["Introduction", "Data Exploration", "Data Visualization","Machine Learning", "Conclusions"]
page = st.sidebar.radio("Navigate to", pages)

if page == pages[0] : 
    st.markdown("""
        ### ⭐ Introduction & City Map
        """)
    def load_image(path):
        return Image.open(path)

    img = load_image("img/bikepic.jpg")
    st.image(img, use_column_width=True)
    st.markdown("""
                Paris has made remarkable progress in recent years toward establishing itself as a bicycle-friendly metropolis. 
                With growing emphasis on sustainable urban mobility, city planners and policymakers are increasingly relying on data-driven insights
                to guide infrastructure development and policy decisions.
                
                This project, Bicycling in Paris, leverages data from automated bicycle counters distributed across the city to uncover key usage patterns. 
                The analysis focuses on two core aspects: understanding hourly and seasonal fluctuations in bicycle traffic, 
                and identifying directional imbalances along popular cycling routes. These insights aim to support more efficient infrastructure planning, 
                targeted investments, and an overall improvement in the cycling experience throughout Paris.
                
        """)
    st.subheader("Bicycle traffic map")


    # OKTOBER 2023 BIS SEPTEMBER 2024
    MAP_select_2324 = ["2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09"]
    df_1023_0924 = df_main[df_main["month_year_count"].str.contains("|".join(MAP_select_2324))]
    MAP_one_year = df_1023_0924
    # Removing of the two outliers
    MAP_one_year = MAP_one_year[MAP_one_year["hourly_count"] != 8190]
    MAP_one_year = MAP_one_year[MAP_one_year["hourly_count"] != 2047]   
    filtered_meterIDs = MAP_one_year.drop_duplicates(subset="technical_meter_identifier")
    tecID_geo = filtered_meterIDs[["technical_meter_identifier", "geographical_coordinates"]]
    Latitudes = tecID_geo["geographical_coordinates"].apply(lambda b: b.split(',')[0])
    Longitudes = tecID_geo["geographical_coordinates"].apply(lambda l: l.split(',')[1])
    lat_lon = pd.DataFrame({"latitude": Latitudes, "longitude": Longitudes})
    df_tecID_geo = pd.concat([tecID_geo, lat_lon], axis=1)
    df_tecID_geo = df_tecID_geo.drop(columns=["geographical_coordinates"])
    df_tecID_geo = df_tecID_geo.rename(columns={"technical_meter_identifier": "Meter ID (technical)"})
    df_tecID_geo = df_tecID_geo.reset_index(drop=True)
    df_tecID_geo["latitude"] = df_tecID_geo["latitude"].astype(float)
    df_tecID_geo["longitude"] = df_tecID_geo["longitude"].astype(float)
    summed_counts = MAP_one_year.groupby("technical_meter_identifier")["hourly_count"].sum().reset_index()
    summed_counts = summed_counts.rename(columns={"technical_meter_identifier": "Meter ID (technical)"})
    summed_counts.reset_index(drop=True)
    df = df_tecID_geo.merge(summed_counts[["Meter ID (technical)", "hourly_count"]], on="Meter ID (technical)", how='left')
    df["hourly_count"] = df["hourly_count"].fillna(0)
    df_MAP = df.rename(columns={"hourly_count": "Total counts"})    
    # Plotly
    fig = px.scatter_mapbox(
        df_MAP,
        lat="latitude",
        lon="longitude",
        hover_name="Meter ID (technical)", 
        color="Total counts",
        size="Total counts",
        size_max=14,
        color_continuous_scale=px.colors.sequential.Plasma,
        zoom=11,        
        mapbox_style="carto-positron")  
    fig.update_layout(title='Counter locations in Paris recording the number of bycicles from October 2023 up to September 2024', width=1000, height=600,

    coloraxis_colorbar=dict(title="Total counts"))

    # Anzeige des Diagramms in Streamlit
    st.plotly_chart(fig)
    st.markdown("""
        ## 🎯 Objectives

        This project aims to extract meaningful insights from bicycle traffic data in Paris to inform data-driven decisions in urban planning 
        and sustainable transportation policy. The key objectives are to:
        
        - ✅ Analyze cycling patterns across different hours, weekdays, and seasons to identify temporal trends and usage peaks.
        - ✅ Develop predictive models using machine learning to estimate hourly bicycle counts based on time and location factors.
        - ✅ Identify directional imbalances on bidirectional routes to highlight potential inefficiencies or infrastructure gaps.
        - ✅ Evaluate model performance in forecasting both total cycling volume and directional flow accuracy.
        - ✅ Deliver actionable recommendations to guide infrastructure development, improve bike-sharing logistics, and enhance the overall cycling experience in Paris.
    """)

if page == pages[1]:
    st.write('### 🔍 Exploration & Description')    
    st.write("The underlying data set contains detailed records of the bicycle traffic in Paris, collected from various automated counters installed across the city. "
            "The data is provided as numerical, categorical and cyclical time-based information in the form of a table as a "
            "CSV file. The original language is French and provided by the city of Paris. The primary focus of the data set is to track the number of bicycles passing a specific counting "
            "station at a given time. The raw data set has 943512 entries subdivided into 16 columns. The analysis focuses on the bicycle traffic volume, represented by the 'Hourly count' variable."
            "  \nKey attributes of the raw data set include:")


    st.markdown("""

        - **Counter Identification**: Unique identifiers for each bicycle counter 

        - **Location Information**: Names and identifiers of the counting sites

        - **Time-based Data**: Exact timestamps of each measurement

        - **Traffic Count**: Number of bicycles recorded per time interval

        - **Metadata**: Additional attributes such as installation dates, photo links, and image types
        """)
    
    st.markdown('<a href="https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name" target="_blank" style="color:blue; text-decoration:none;">Link to Data</a>', unsafe_allow_html=True)
    st.write("Exemplary data extract")
    data_example = {
    "Meter identifier": ["100003096-353242251", "100003096-353242251", "100003096-353242251", "100003096-353242251"],
    "Meter name": ["97 avenue Denfert Rochereau SO-NE", "97 avenue Denfert Rochereau SO-NE", "97 avenue Denfert Rochereau SO-NE", "97 avenue Denfert Rochereau SO-NE"],
    "Metering site identifier": ["100003096", "100003096", "100003096", "100003096"],
    "Name of metering site": ["97 avenue Denfert Rochereau", "97 avenue Denfert Rochereau", "97 avenue Denfert Rochereau", "97 avenue Denfert Rochereau"],
    "Hourly count": [4, 63, 16, 225],
    "Metering date and time": ["2023-09-01T05:00:00+02:00", "2023-09-01T07:00:00+02:00", "2023-09-01T06:00:00+02:00", "2023-09-01T08:00:00+02:00"],
    "Metering site installation date": ["2012-02-22", "2012-02-22", "2012-02-22", "2012-02-22"],
    "Link to photo of metering site": ["https://filer.eco-counter-tools.com/file/10/6d...", "https://filer.eco-counter-tools.com/file/10/6d...", "https://filer.eco-counter-tools.com/file/10/6d...", "https://filer.eco-counter-tools.com/file/10/6d..."],
    "Geographical coordinates": ["48.83504,2.33314", "48.83504,2.33314", "48.83504,2.33314", "48.83504,2.33314"],
    "Technical meter identifier": ["Y2H21111072", "Y2H21111072", "Y2H21111072", "Y2H21111072"],
    "Photo ID": ["https://filer.eco-counter-tools.com/file/10/6d...", "https://filer.eco-counter-tools.com/file/10/6d...", "https://filer.eco-counter-tools.com/file/10/6d...", "https://filer.eco-counter-tools.com/file/10/6d..."],
    "Test link to photos of counting site": ["https://filer.eco-counter-tools.com/file/10/6d...", "https://filer.eco-counter-tools.com/file/10/6d...", "https://filer.eco-counter-tools.com/file/10/6d...", "https://filer.eco-counter-tools.com/file/10/6d..."],
    "ID photo 1": ["https://filer.eco-counter-tools.com/file/10/6d...", "https://filer.eco-counter-tools.com/file/10/6d...", "https://filer.eco-counter-tools.com/file/10/6d...", "https://filer.eco-counter-tools.com/file/10/6d..."],
    "URL website": ["https://www.eco-visio.net/Photos/100003096", "https://www.eco-visio.net/Photos/100003096", "https://www.eco-visio.net/Photos/100003096", "https://www.eco-visio.net/Photos/100003096"],
    "Image type": ["jpg", "jpg", "jpg", "jpg"],
    "Month year count": ["2023-09", "2023-09", "2023-09", "2023-09"]
    }
    df_ex = pd.DataFrame(data_example)
    st.dataframe(df_ex)
    st.markdown("""
                Two extreme values of `hourly_count` (8190 and 2047) have been filtered out. These high counts were both generated on October 22, 2023. On this date, bicycle traffic in Paris was particularly heavy, as the "Fête du Vélo", an annual festival in honor of the bicycle, took place on this day. These two values are not representative of the normal bicycle traffic.
    """)

if page == pages[2] :
    st.write("### 📈 Data Visualization")
    st.markdown("""
                To gain an initial understanding of bicycle behavior:

                - Daily traffic trends revealed a significant drop in weekend cycling, consistent with reduced commuting.
                - Fall had the highest total bike counts overall, followed closely by Summer and Spring, while Winter had significantly lower traffic — likely due to weather-related factors.
                """)
    # Convert to datetime
    df_directional["date_time_utc_plus_2"] = pd.to_datetime(df_directional["date_time_utc_plus_2"])
    df_main["date_time_utc_plus_2"] = pd.to_datetime(df_main["date_time_utc_plus_2"])

    # Create 'season' column from month
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    df_directional["season"] = df_directional["date_time_utc_plus_2"].dt.month.map(get_season)
    # Sum hourly counts per day (not hour), across all counters
    daily_counts = (
        df_directional.groupby(["date_time_utc_plus_2", "weekday", "season"])["hourly_count"]
        .sum()
        .reset_index(name="daily_total")
    )

    # Now group by weekday and season, and compute total and average per week
    counts_by_weekday_season = (
        daily_counts
        .groupby(["weekday", "season"])
        .agg(total_counts=("daily_total", "sum"))
        .reset_index()
    )

    # Divide by 52 to get the average per weekday (assuming 52 weeks)
    counts_by_weekday_season["average_per_weekday"] = (
        counts_by_weekday_season["total_counts"] / 52
    )

    # Order weekdays
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    counts_by_weekday_season["weekday"] = pd.Categorical(
        counts_by_weekday_season["weekday"], categories=weekday_order, ordered=True
    )
    counts_by_weekday_season = counts_by_weekday_season.sort_values("weekday")

    # Plot
    fig_seasonal = px.line(
        counts_by_weekday_season,
        x="weekday",
        y="average_per_weekday",
        color="season",
        title="📅 Average Bicycle Counts per Weekday by Season (Oct 2023 – Sep 2024)",
        labels={
            "average_per_weekday": "Avg Counts per Weekday",
            "total_counts": "Total Counts",
            "weekday": "Day of Week",
            "season": "Season"
        },
        hover_data=["total_counts", "average_per_weekday"]
        ) 
    st.plotly_chart(fig_seasonal, use_container_width=True)
    
    #Preprocessing plot Total Counts per month per season
    # Convert datetime and extract month-year and season
    df_main["date_time_utc_plus_2"] = pd.to_datetime(df_main["date_time_utc_plus_2"])
    df_main["month_year"] = df_main["date_time_utc_plus_2"].dt.strftime("%Y-%m")
    df_main["season"] = df_main["date_time_utc_plus_2"].dt.month.map(get_season)
    
    # Group by month-year
    monthly_counts = (
        df_main.groupby("month_year")["hourly_count"]
        .sum()
        .reset_index(name="total_counts")
    )
    monthly_counts["month_dt"] = pd.to_datetime(monthly_counts["month_year"])
    monthly_counts = monthly_counts.sort_values("month_dt")
    # Group by season
    seasonal_counts = (
        df_main.groupby("season")["hourly_count"]
        .sum()
        .reset_index(name="total_counts")
    )
    seasonal_counts["season"] = pd.Categorical(seasonal_counts["season"], 
                                            categories=["Winter", "Spring", "Summer", "Fall"],
                                            ordered=True)

    # Custom colors (light blue for July and August 2024)
    highlight_months = ['2024-07', '2024-08']
    colors = ['#1f77b4' if month not in highlight_months else '#A6C8FF' for month in monthly_counts['month_year']]
    seasonal_counts = seasonal_counts[seasonal_counts["season"].isin(["Summer", "Winter"])]

    import plotly.graph_objects as go

    # Create subplot with 2 plots side by side
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Monthly Counts", "Seasonal Counts")
    )

    # Monthly bar chart
    fig.add_trace(
        go.Bar(
            x=monthly_counts["month_year"],
            y=monthly_counts["total_counts"],
            marker=dict(color=colors)
        ),
        row=1, col=1
    )

    # Seasonal bar chart
    fig.add_trace(
        go.Bar(
            x=seasonal_counts["season"],
            y=seasonal_counts["total_counts"],
            marker=dict(color='#ff7f0e')
        ),
        row=1, col=2
    )

    # Layout and formatting
    fig.update_layout(
        title_text="Total Counts per Month and per Season 🌻❄️🍂🌱",
        showlegend=False,
        height=500,
        width=1000,
        xaxis=dict(title="Month"),
        xaxis2=dict(title="Season"),
        yaxis=dict(title="Counts"),
        yaxis2=dict(title="Counts"),
        template="plotly_white",
        yaxis_tickformat='.2e'
    )

    # Display in Streamlit
    st.plotly_chart(fig)


    st.markdown(""" 
                - Monthly trends highlighted a dip in August due to holiday periods.    
                - Bike traffic is most active during the day, with over half of all rides occurring in the Afternoon (40.5%) and Morning (33.7%) together, while nighttime periods (Late Night + Midnight) account for only ~11% of total traffic.
                """)
    
    #Integer Hour
    time_bins = [0, 6, 12, 18, 21, 24]  
    time_labels = ['Midnight', 'Morning', 'Afternoon', 'Evening', 'Late Night']
    df_main['metering_date_and_time'] = pd.to_datetime(df_main['metering_date_and_time'], errors='coerce', utc=True)
    df_main['hour'] = df_main['metering_date_and_time'].dt.hour
    #Timeperiod 
    df_main['time_period'] = pd.cut(
        df_main['hour'], 
        bins=time_bins, 
        labels=time_labels, 
        right=False  
    )
    df_pie_time = df_main.groupby('time_period', observed=True).agg({'hourly_count': 'sum'}).reset_index().sort_values(by='hourly_count')
    fig = px.pie(df_pie_time, values='hourly_count', names='time_period', title='🕔 Bike Traffic per each Time period')
    st.plotly_chart(fig)
    st.markdown("""
                Additional visualizations focused on directional flow differences:
                """)
    # --- Prepare directional flow data ---
    @st.cache_data
    def prepare_directional_flow(df_directional, top_n=20):
        grouped = df_directional.groupby("base_route").agg({
            "so-ne": "sum",
            "ne-so": "sum",
            "se-no": "sum",
            "no-se": "sum"
        }).reset_index()
    
        # Net flow per direction pair
        grouped["Flow_SO_NE"] = grouped["so-ne"] - grouped["ne-so"]
        grouped["Flow_SE_NO"] = grouped["se-no"] - grouped["no-se"]

        # Total net flow
        grouped["Total_Flow_Imbalance"] = grouped["Flow_SO_NE"] + grouped["Flow_SE_NO"]

        # Absolute net imbalance for cleaner plot
        grouped["Net_Imbalance"] = grouped["Total_Flow_Imbalance"].abs()

        # Sort by absolute value
        grouped = grouped.sort_values("Net_Imbalance", ascending=False).head(top_n)

        return grouped  
    # --- Streamlit UI ---
    st.write("#### Directional Flow Imbalance by Route")
    st.write("This chart shows the top routes in Paris with the largest directional bicycle traffic imbalance.")

    top_n = st.slider("Number of top routes to display", 5, 30, 20)
    df_flow = prepare_directional_flow(df_directional, top_n=top_n)

    # --- Plot ---
    fig = px.bar(
        df_flow,
        x="Net_Imbalance",
        y="base_route",
        orientation="h",
        title="🔄 Top Routes by Net Directional Flow Imbalance",
        labels={
            "Net_Imbalance": "Absolute Net Flow Imbalance",
            "base_route": "Route"
        }
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        margin=dict(l=200, r=20, t=60, b=40),
        font=dict(family="Arial", size=14),
        xaxis_tickformat=".2s"
    )

    st.plotly_chart(fig, use_container_width=True)


    st.markdown("""

              - Major flow imbalances are concentrated along a few key routes like **quai de la Tournelle** and **boulevard Masséna**, suggesting these corridors have highly directional commuting patterns (likely tied to peak-hour flows in and out of central Paris).

    """)
if page == pages[3] :
    st.write("### ⚙️ Machine Learning")
    st.markdown("""
                Two separate ML pipelines were developed to address distinct goals.
            """)
    col1, col2, col3 = st.columns(3)  
    if col1.button("Hourly Count Analysis"):
        st.write("#### 1. Hourly Count Prediction")
        st.markdown("""
                    - Goal: Prediction of hourly counts bicycle traffic volume
                    - Variable: 'hourly count' is a continuous variable \u2192 regression problem
                    - Models: RandomForestRegressor (**RFR**) and XGBRegressor (**XGBR**)
                    """)
        st.write("#### Preprocessing:")
        st.markdown("""
                    - Fragmentation of "Metering date and time" variable into years, months, days, hours and seasons
                    - Definition of target ("Hourly count") and feature variables
                    - Split data into train (80%) and test (20%) sets
                    - Definition of numerical and categorical variables
                    - Encoding (OneHotEncoder) of categorical variables
                    - Standard scaling of numerical variables
                """)
        st.write("#### Results")
        # Define your data
        data = {
            "R² train": [0.98, 0.78],
            "R² test": [0.87, 0.78],
            "MAE train": [6.79, 26.63],
            "MAE test": [18.29, 26.78],
            "MSE train": [208.09, 2435.52],
            "MSE test": [1489.51, 2468.63]
        }
        index = ["RFR", "XGBR"]

        df = pd.DataFrame(data, index=index)

        # Optional: apply styling
        styled_df = df.style.set_table_styles([
            {"selector": "th", "props": [("background-color", "#42A5F5"), ("color", "white"), ("font-weight", "bold")]},
            {"selector": "td", "props": [("text-align", "center")]}
        ]).format("{:.2f}")

        # Display it in Streamlit
        st.dataframe(styled_df, use_container_width=True)
        st.markdown("""

                    - Random Forest outperformed XGBoost with an R² of 0.87 on the test set
                    - Fivefold cross-validation: \n
                        **RFR** MSE = 1722.2 \n
                        **XGBR** MSE = 2575.2
                    """)
        st.subheader("Feature importance")
        # GRAPHIC RFR
        # Daten für die Features und ihre Wichtigkeit
        features = ["hour", "day", "month"]
        importance = [0.297465, 0.156254, 0.087202]

        # Erstellen der Plotly Barplot
        fig_RFR = go.Figure(data=[go.Bar(
            x=features,
            y=importance,
            marker_color='royalblue'
        )])

        # Titel und Achsenbeschriftungen hinzufügen
        fig_RFR.update_layout(
            title="Top 3 features of RandomForestRegressor",
            xaxis_title="Feature",
            yaxis_title="Importance",
            yaxis=dict(range=[0.00, 0.30])  # Y-Achsenbereich festlegen
        )

        # Anzeige der Graphik in Streamlit
        st.plotly_chart(fig_RFR)

        # GRAPHIC XGBR
        # Daten für die Features und ihre Wichtigkeit
        features = ["64 Rue de Rivoli", "38 Rue Turbigo", "73 Boulevard de Sébastopol"]
        importance = [0.125025, 0.095337, 0.089949]
        # Erstellen der Plotly Barplot
        fig_XGB = go.Figure(data=[go.Bar(
            x=features,
            y=importance,
            marker_color='royalblue'
        )])
        # Titel und Achsenbeschriftungen hinzufügen
        fig_XGB.update_layout(
            title="Top 3 features of XGBRegressor",
            xaxis_title="Feature",
            yaxis_title="Importance",
            yaxis=dict(range=[0.00, 0.126])  # Y-Achsenbereich festlegen
        )
        # Anzeige der Graphik in Streamlit
        st.plotly_chart(fig_XGB)
        
    if col2.button("Peak Hour Analysis"):
        # Use st.cache_data for data preprocessing
        @st.cache_data
        def preprocess_data(df_main):
        
            # Convert date columns to datetime
            df_main['metering_date_and_time'] = pd.to_datetime(df_main['metering_date_and_time'], errors='coerce', utc=True)

            # Extract date and time components
            df_main['date_time'] = pd.to_datetime(df_main['metering_date_and_time'], errors='coerce', utc=True)
            df_main['hour'] = df_main['metering_date_and_time'].dt.hour
            df_main['hour_time'] = df_main['metering_date_and_time'].dt.strftime('%H:%M:%S')

            return df_main

        # Use st.cache_data for data filtering and transformation
        @st.cache_data
        def filter_and_transform_data(df_main):
            # Timeperiods based on hour
            time_bins = [0, 6, 12, 18, 21, 24]  
            time_labels = ['Midnight', 'Morning', 'Afternoon', 'Evening', 'Late Night']
            df_main.loc[:, 'time_period'] = pd.cut(df_main['hour'], bins=time_bins, labels=time_labels, right=False)

            # Weekday and Month
            df_main.loc[:, 'month'] = df_main['date_time'].dt.month
            df_main.loc[:, 'weekday_name'] = df_main['date_time'].dt.day_name()
            df_main.loc[:, 'weekday_number'] = df_main['date_time'].dt.weekday
            
            # Latitude and Longitude from 'Coordinates'
            df_main.loc[:, 'latitude'] = df_main['geographical_coordinates'].apply(lambda x: x.split(',')[0]).astype(float)
            df_main.loc[:, 'longitude'] = df_main['geographical_coordinates'].apply(lambda x: x.split(',')[1]).astype(float)

            # Categorize zones (North/South/Center)
            def categorize_zone(latitude):
                if latitude > 48.86:   
                    return "North"
                elif latitude < 48.84:
                    return "South"
                else:
                    return "Center"

            df_main.loc[:, 'region'] = df_main['latitude'].apply(categorize_zone)

            return df_main

        # Use st.cache_data for peak hour determination
        @st.cache_data
        def add_peak_hour(df_main):
            df_North = df_main[df_main['region'] == 'North'].copy()
            threshold = df_North['hourly_count'].quantile(0.75)  # Top 25% of traffic is Peak
            df_North['peak_hour'] = (df_North['hourly_count'] > threshold).astype(int)

            df_South = df_main[df_main['region'] == 'South'].copy()
            threshold = df_South['hourly_count'].quantile(0.75)
            df_South['peak_hour'] = (df_South['hourly_count'] > threshold).astype(int)

            df_Center = df_main[df_main['region'] == 'Center'].copy()
            threshold = df_Center['hourly_count'].quantile(0.75)
            df_Center['peak_hour'] = (df_Center['hourly_count'] > threshold).astype(int)

            return pd.concat([df_North, df_South, df_Center])
        
        df_main = preprocess_data(df_main)
        df_main = filter_and_transform_data(df_main)
        df_combined = add_peak_hour(df_main)
        new_df = df_combined[['time_period', 'month', 'weekday_name', 'hour', 'latitude', 'longitude', 'peak_hour', 'region']]
         # Prepare data
        X = new_df.drop(['peak_hour'], axis=1)
        y = new_df['peak_hour']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        cat_cols = ['time_period', 'region']
        num_cols = ['latitude', 'longitude']
        circular_cols = ['hour', 'weekday_name', 'month']

        # Preprocessing circular features and encoding
        def preprocess_circular_features(circular_train, circular_test):
            # Hour
            circular_train.loc[:, 'sin_hour'] = circular_train['hour'].apply(lambda h: np.sin(2 * np.pi * h / 24))
            circular_train.loc[:, 'cos_hour'] = circular_train['hour'].apply(lambda h: np.cos(2 * np.pi * h / 24))

            circular_test.loc[:, 'sin_hour'] = circular_test['hour'].apply(lambda h: np.sin(2 * np.pi * h / 24))
            circular_test.loc[:, 'cos_hour'] = circular_test['hour'].apply(lambda h: np.cos(2 * np.pi * h / 24))

            # Weekday
            circular_train.loc[:, 'sin_weekday'] = circular_train['weekday_name'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}).apply(lambda h: np.sin(2 * np.pi * h / 7))
            circular_train.loc[:, 'cos_weekday'] = circular_train['weekday_name'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}).apply(lambda h: np.cos(2 * np.pi * h / 7))

            circular_test.loc[:, 'sin_weekday'] = circular_test['weekday_name'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}).apply(lambda h: np.sin(2 * np.pi * h / 7))
            circular_test.loc[:, 'cos_weekday'] = circular_test['weekday_name'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}).apply(lambda h: np.cos(2 * np.pi * h / 7))

            # Month
            circular_train.loc[:, 'sin_month'] = circular_train['month'].apply(lambda m: np.sin(2 * np.pi * m / 12))
            circular_train.loc[:, 'cos_month'] = circular_train['month'].apply(lambda m: np.cos(2 * np.pi * m / 12))

            circular_test.loc[:, 'sin_month'] = circular_test['month'].apply(lambda m: np.sin(2 * np.pi * m / 12))
            circular_test.loc[:, 'cos_month'] = circular_test['month'].apply(lambda m: np.cos(2 * np.pi * m / 12))

            return circular_train.drop(['hour', 'weekday_name', 'month'], axis=1), circular_test.drop(['hour', 'weekday_name', 'month'], axis=1)
        circular_train = X_train[circular_cols]
        circular_test = X_test[circular_cols]

        # Apply preprocessing
        circular_train, circular_test = preprocess_circular_features(circular_train, circular_test)

        # Imputation of missing values
        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        num_train_imputed = numeric_imputer.fit_transform(X_train[num_cols])
        num_test_imputed = numeric_imputer.transform(X_test[num_cols])

        cat_train_imputed = categorical_imputer.fit_transform(X_train[cat_cols])
        cat_test_imputed = categorical_imputer.transform(X_test[cat_cols])

        num_train_imputed = pd.DataFrame(num_train_imputed, columns=num_cols)
        cat_train_imputed = pd.DataFrame(cat_train_imputed, columns=cat_cols)
        num_test_imputed = pd.DataFrame(num_test_imputed, columns=num_cols)
        cat_test_imputed = pd.DataFrame(cat_test_imputed, columns=cat_cols)

        X_train_imputed = pd.concat([num_train_imputed, cat_train_imputed, circular_train], axis=1)
        X_test_imputed = pd.concat([num_test_imputed, cat_test_imputed, circular_test], axis=1)

        # One-Hot Encoding of categorical features
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        cat_train_encoded = encoder.fit_transform(cat_train_imputed)
        cat_test_encoded = encoder.transform(cat_test_imputed)

        cat_train_encoded = pd.DataFrame(cat_train_encoded, columns=encoder.get_feature_names_out(cat_cols))
        cat_test_encoded = pd.DataFrame(cat_test_encoded, columns=encoder.get_feature_names_out(cat_cols))

        X_train_encoded = pd.concat([num_train_imputed, cat_train_encoded], axis=1)
        X_test_encoded = pd.concat([num_test_imputed, cat_test_encoded], axis=1)

        # Standardizing the data
        scaler = StandardScaler()
        X_train_encoded = scaler.fit_transform(X_train_encoded)
        X_test_encoded = scaler.transform(X_test_encoded)

        # Model Training - Decision Tree
        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train_encoded, y_train)
        
    
    
        st.write("#### Peak Hour Prediction")
        st.markdown("""
                    - Goal: Predict if a given hour is a peak hour

                    - Models: Decision Tree and Random Forest
                    """)
        st.write("#### Preprocessing:")
        st.markdown("""
                    - The dataset is divided into three regions: North, South, and Center
                    - Peak hour is defined as the top 25% of hourly count for each region
                    - A binary feature is created: 1 for peak hour, 0 non-peak hour
                """)
        st.write("#### Results")
        st.subheader("Decision Tree")

        st.write('Decision Tree Score on Train Set:', clf.score(X_train_encoded, y_train))
        st.write('Decision Tree Score on Test Set:', clf.score(X_test_encoded, y_test))

        y_pred = clf.predict(X_test_encoded)
        st.write("Confusion Matrix for Decision Tree:")
        st.write(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Prediction']))

        st.write("Classification Report for Decision Tree:")
        st.text(classification_report(y_test, y_pred, zero_division=1))
        # Model Training - Random Forest
        st.subheader("Random Forest")

        # Use class weights to handle the class imbalance instead of oversampling
        rf = RandomForestClassifier(class_weight='balanced')  # Automatically adjusts weights based on class distribution

        # Fit the model to the original data (without resampling)
        rf.fit(X_train_encoded, y_train)

        # Display the model performance on the train set
        st.write('Random Forest Score on Train Set:', rf.score(X_train_encoded, y_train))

        # Display the model performance on the test set
        st.write('Random Forest Score on Test Set:', rf.score(X_test_encoded, y_test))

        # Make predictions on the test set
        y_pred = rf.predict(X_test_encoded)

        # Display confusion matrix for the Random Forest model
        st.write("Confusion Matrix for Random Forest:")
        confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Prediction'])
        st.write(confusion_matrix)

        # Display classification report for the Random Forest model
        st.write("Classification Report for Random Forest:")
        st.text(classification_report(y_test, y_pred, zero_division=1))
        st.markdown("""
                    - The Decision Tree model performs well in predicting peak hours but is more accurate in identifying non-peak hours. The model's accuracy on the test set is strong, with a balanced precision and recall for non-peak hours. However, it struggles with predicting peak hours, as shown by the lower recall and F1-score for the peak hour class.
                """)
    # Initialize session state
    if "show_directional_ml" not in st.session_state:
        st.session_state.show_directional_ml = False
    
    # Button to toggle section visibility
    if col3.button("Directional Flow and Route-Level Imbalance Analysis", key="toggle_directional_section"):
        st.session_state.show_directional_ml = not st.session_state.show_directional_ml  # Toggle the state
        
    # Show the section only if state is True
    if st.session_state.show_directional_ml:
        st.write("#### Directional Flow Difference Prediction")
        st.markdown("""
            - Goal: Predict the difference in traffic volume between opposite directions on the same route. 
            - Models: RandomForestRegressor and Linear Regression
        """)
        st.write("##### Preprocessing")
        st.markdown("""
            - Split Counter Name into Base Route and Direction  
            - Retained only routes with valid bidirectional data    
            - Created a new feature Difference  
        """)
        st.write("##### Results")

        # Hugging Face URLs
        lr_url = "https://huggingface.co/BFerratto/bicycle-models/resolve/main/lr_model.joblib"
        rf_url = "https://huggingface.co/BFerratto/bicycle-models/resolve/main/rf_model_light.joblib"
        test_data_url = "https://huggingface.co/BFerratto/bicycle-models/resolve/main/test_data.joblib"

        # Load function
        @st.cache_resource
        def load_joblib_from_url(url):
            response = requests.get(url, stream=True)
            buffer = BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                buffer.write(chunk)
            buffer.seek(0)
            return joblib.load(buffer)

        # Initialize session state
        for key in ["rf_model", "lr_model", "X_test", "y_test"]:
            if key not in st.session_state:
                st.session_state[key] = None

        # Red button styling for Full RF Model
        st.markdown("""
            <style>
            div.stButton > button[data-testid="baseButton"][key="load_full_rf"] {
                background-color: red !important;
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Model + data loading buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Load Light Random Forest Model", key="load_light_rf"):
                with st.spinner("Loading Light Random Forest model..."):
                    st.session_state.rf_model = load_joblib_from_url(rf_url)

        with col2:
            if st.button("Load Linear Regression", key="load_lr"):
                with st.spinner("Loading Linear Regression model..."):
                    st.session_state.lr_model = load_joblib_from_url(lr_url)

        with col3:
            if st.button("Load Test Data", key="load_test_data"):
                with st.spinner("Loading test data..."):
                    X_test, y_test = load_joblib_from_url(test_data_url)
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test

        # Run predictions and show results if all components are loaded
        if all([
            st.session_state.rf_model,
            st.session_state.lr_model,
            st.session_state.X_test is not None,
            st.session_state.y_test is not None
        ]):
            y_pred_lr = st.session_state.lr_model.predict(st.session_state.X_test)
            y_pred_rf = st.session_state.rf_model.predict(st.session_state.X_test)

            metrics = {
                "R²": [
                    r2_score(st.session_state.y_test, y_pred_rf),
                    r2_score(st.session_state.y_test, y_pred_lr)
                ],
                "MAE": [
                    mean_absolute_error(st.session_state.y_test, y_pred_rf),
                    mean_absolute_error(st.session_state.y_test, y_pred_lr)
                ],
                "RMSE": [
                    mean_squared_error(st.session_state.y_test, y_pred_rf, squared=False),
                    mean_squared_error(st.session_state.y_test, y_pred_lr, squared=False)
                ]
            }

            index = ["RFR", "LR"]
            df_metrics = pd.DataFrame(metrics, index=index)

            # Highlighted comparison
            st.markdown(f"""
                Random Forest achieved an R² of **{df_metrics.loc['RFR', 'R²']:.2f}**, outperforming Linear Regression (**{df_metrics.loc['LR', 'R²']:.2f}**).  
                MAE: **{df_metrics.loc['RFR', 'MAE']:.2f}** (RFR) vs **{df_metrics.loc['LR', 'MAE']:.2f}** (LR)  
                RMSE: **{df_metrics.loc['RFR', 'RMSE']:.2f}** (RFR) vs **{df_metrics.loc['LR', 'RMSE']:.2f}** (LR)
            """)

            # Styled metrics table
            styled_df = df_metrics.style.set_table_styles([
                {"selector": "th", "props": [("background-color", "#42A5F5"), ("color", "white"), ("font-weight", "bold")]},
                {"selector": "td", "props": [("text-align", "center")]}
            ]).format("{:.2f}")
            st.dataframe(styled_df, use_container_width=True)

            # Plot: Actual vs Predicted
            df_plot = pd.DataFrame({
                "Actual": st.session_state.y_test,
                "Random Forest": y_pred_rf,
                "Linear Regression": y_pred_lr
            })
            df_melted = df_plot.melt(id_vars="Actual", var_name="Model", value_name="Predicted")

            fig = px.scatter(
                df_melted,
                x="Actual",
                y="Predicted",
                color="Model",
                opacity=0.6,
                labels={"Actual": "Actual Values", "Predicted": "Predicted Values"},
                title="Actual vs. Predicted Values"
            )
            min_val = df_melted["Actual"].min()
            max_val = df_melted["Actual"].max()
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(dash="dash", color="black"),
                    name="Perfect Fit"
                )
            )
            st.plotly_chart(fig, use_container_width=True)  
            st.markdown("""
            **Feature Importance:**  
            - `hourly_count`, hour of day, and coordinates were most influential.
            """)
            # Feature Importance Plot for Random Forest
            if st.session_state.rf_model is not None and hasattr(st.session_state.rf_model, "feature_importances_"):
                feature_importance = st.session_state.rf_model.feature_importances_

                importance_df = pd.DataFrame({
                    'Feature': st.session_state.X_test.columns,
                    'Importance': feature_importance
                }).sort_values(by='Importance', ascending=False)  # Sort by descending importance

                # Keep top 5 features
                top_features_df = importance_df.head(5).sort_values(by='Importance', ascending=True)  # Re-sort for horizontal bar

                # Plot using Plotly
                fig_importance = px.bar(
                    top_features_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='🔍 Top 5 Feature Importances in Random Forest',
                    labels={'Importance': 'Feature Importance', 'Feature': 'Feature'},
                    height=400
                )

                st.plotly_chart(fig_importance, use_container_width=True)

if page == pages[4] :

    st.markdown("""
        ### 🧐 **Conclusion & Key Recommendations**  
                
        The analysis of bicycle usage in Paris reveals clear temporal and seasonal patterns. Cycling activity is notably higher during weekdays, particularly during commuting hours, indicating that many residents rely on bicycles as a mode of transport to and from work. Additionally, warmer months show significantly increased activity, underscoring the influence of weather conditions on cycling behavior.
        The installation dates of bicycle counters (meters) provide valuable context for interpreting usage trends over time. They also help city planners schedule timely maintenance and evaluate the effectiveness of past infrastructure investments. This information is crucial for optimizing both current and future developments in the city's cycling network.
        
        **Key Recommendations:**        

        - ✅ **Optimize Infrastructure Deployment Seasonally:**
        
            Prioritize the installation and maintenance of bike infrastructure in early spring to ensure readiness for peak usage in warmer months.
        - ✅ **Focus on Commuting Corridors:**
        
            Invest in expanding and enhancing bike lanes along major commuting routes that show high weekday traffic, particularly during morning and evening rush hours.
        - ✅ **Dynamic Resource Allocation:**
        
            Use real-time and historical meter data to dynamically allocate resources—such as bike-sharing stations and maintenance crews—where and when demand is highest.
        - ✅ **Integrate Weather Forecasting into Planning:**
        
            Consider integrating weather predictions into traffic management and promotional campaigns to encourage safe and increased cycling on favorable days.
        - ✅ **Public Awareness & Incentives:**
        
            Launch campaigns to raise awareness of cycling benefits and offer seasonal incentives (e.g., discounts or competitions) to maintain ridership during shoulder seasons.
        
        By leveraging these insights, Paris can continue to foster a more sustainable and cyclist-friendly urban environment, enhancing both mobility and quality of life for its residents.
    """)

    