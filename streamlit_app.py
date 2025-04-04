import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import joblib
import sys
import sklearn
import requests
from io import BytesIO
import io
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tempfile
import requests
from plotly.subplots import make_subplots


@st.cache_data
def load_main_data():
    url = "https://huggingface.co/datasets/BFerratto/bikesParis/resolve/main/df_processed_1023_0924.csv"
    return pd.read_csv(url)

@st.cache_data
def load_directional_data():
    url = "https://huggingface.co/datasets/BFerratto/bikesParis/resolve/main/df_processed_one_direction.csv"
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
        Paris has made significant strides toward becoming a more bike-friendly city. As urban planners and policymakers aim to optimize infrastructure and promote sustainable mobility, understanding bicycle traffic patterns is essential. This project analyzes bicycle traffic data collected from automated counters across Paris with a dual focus: evaluating general hourly traffic volumes and identifying imbalances in directional flow across routes.
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

        This project aims to uncover insights from bicycle traffic data in Paris to support data-driven urban planning and transportation policy. Specifically, the objectives are to:

        - ✅ Analyze overall cycling activity by hour, day, and season. 
        - ✅ Use machine learning to predict hourly bicycle counts based on time and location.  
        - ✅ Detect and model directional imbalances in cycling traffic on bidirectional routes.    
        - ✅ Evaluate the effectiveness of predictive models for both total volume and directional flow.    
        - ✅ Provide actionable recommendations for infrastructure planning and bike-sharing strategies.
        
    """)

if page == pages[1]:
    st.write('### 🔍 Exploration & Description')
    st.markdown("""
                The dataset, sourced from opendata.paris.fr, consists of over 940,000 entries covering October 2023 to September 2024. It includes information on:

                - Bicycle count per hour (`Comptage horaire`)

                - Meter location and identifiers

                - Timestamp (`Date et heure de comptage`)
                - Geographic coordinates
                - Metadata (e.g., photo links, installation dates)
                
                - A link to the original Data:
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
                
                After translation from French and cleaning, 16 columns were retained, providing temporal, spatial, and categorical features relevant to understanding traffic patterns. Columns like `metering_site_installation_date` and redundant photo identifiers were removed to enhance processing efficiency.

                **Pre-processing**

                The period from October 2023 to September 2024 was isolated to obtain an exact period of one year. Afterward, the index of the columns was reset. 

                Two extreme values of `hourly_count` (8190 and 2047) have been filtered out. These high counts were both generated on October 22, 2023. On this date, bicycle traffic in Paris was particularly heavy, as the "Fête du Vélo", an annual festival in honor of the bicycle, took place on this day. These two values are not representative of the normal bicycle traffic.

                Columns maintained:
    """)
    st.dataframe(df_main.columns[:6])

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
                - Bike traffic is most active during the day, with over half of all rides occurring in the Afternoon (30.7%) and Morning (27.2%) together, while nighttime periods (Late Night + Midnight) account for only ~16% of total traffic.
                """)
    
    #Integer Hour
    time_bins = [0, 6, 12, 18, 21, 24]  
    time_labels = ['Midnight', 'Morning', 'Afternoon', 'Evening', 'Late Night']

    #Timeperiod 
    df_directional['time_period'] = pd.cut(
        df_directional['hour'], 
        bins=time_bins, 
        labels=time_labels, 
        right=False  
    )
    df_pie_time = df_directional.groupby('time_period', observed=True).agg({'hourly_count': 'sum'}).reset_index().sort_values(by='hourly_count')
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
    col1, col2 = st.columns(2)  
    if col1.button("For Hourly Count Analysis"):
        st.write("#### 1. Hourly Count Prediction")
        st.markdown("""
                    - Goal: Predict hourly bicycle traffic volume.

                    - Models: RandomForestRegressor and XGBRegressor
                    """)
        st.write("#### Preprocessing:")
        st.markdown("""
                    - Extracted year, month, day, hour, and season from timestamps.

                    - One-hot encoded categorical variables (e.g., location).

                    - Scaled numerical features.
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

                    - Random Forest outperformed XGBoost with an R² of 0.87 on the test set.

                    - MAE: ~18 (Random Forest), ~27 (XGBoost)

                    - MSE and cross-validation supported the robustness of Random Forest.

                    **Feature Importance**

                    - Time-based variables, especially `hour`, strongly influenced prediction accuracy.



                    """)
        st.write("#### 2. Hourly Count Prediction per time period")
        st.markdown("""
                    - Goal: WRITE SOMETHING HERE

                    - Models: WRITE SOMETHING HERE
                    """)
        st.write("#### Preprocessing:")
        st.markdown("""
                    - WRITE SOMETHING HERE
                """)
        st.write("#### Results")
        st.markdown("""
                    - WRITE SOMETHING HERE
                """)
    # Initialize session state
    if "show_directional_ml" not in st.session_state:
        st.session_state.show_directional_ml = False
    
    # Button to toggle section visibility
    if col2.button("For Directional Flow and Route-Level Imbalance Analysis", key="toggle_directional_section"):
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
        full_rf_url = "https://huggingface.co/BFerratto/bicycle-models/resolve/main/rf_model.joblib"
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
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Load Light Random Forest Model", key="load_light_rf"):
                with st.spinner("Loading Light Random Forest model..."):
                    st.session_state.rf_model = load_joblib_from_url(rf_url)

        with col2:
            if st.button("Full Random Forest Model ( ⚠️local only)", key="load_full_rf"):
                with st.spinner("Loading Full Random Forest model..."):
                    st.session_state.rf_model = load_joblib_from_url(full_rf_url)
                st.success("Full Random Forest model loaded!")

        with col3:
            if st.button("Load Linear Regression", key="load_lr"):
                with st.spinner("Loading Linear Regression model..."):
                    st.session_state.lr_model = load_joblib_from_url(lr_url)

        with col4:
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
                }).sort_values(by='Importance', ascending=True)  # Ascending for horizontal plot

                # Plot using Plotly
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='🔍 Feature Importance in Random Forest',
                    labels={'Importance': 'Feature Importance', 'Feature': 'Feature'},
                    height=500
                )

                st.plotly_chart(fig_importance, use_container_width=True)

if page == pages[4] :

    st.markdown("""
        ### 🧐 **Conclusion & Key Recommendations**  
                
        Bicycle usage in Paris is highly time-sensitive and season-dependent. Traffic peaks during weekdays and warm months, while installation data helps schedule meter maintenance and assess city investment in cycling infrastructure.

        ### **Evaluations**        
    """)

    col1, col2 = st.columns(2)

    if col1.button("For Hourly Count Analysis"):
        st.markdown("""   
                    The Random Forest model’s superior performance highlights the relevance of cyclical time variables in predicting traffic volume. It shows that time-of-day, weekday, and location significantly shape traffic patterns. These insights support policies such as timed bike-lane allocation and resource distribution during peak hours.
        """)      

    if col2.button("For Directional Flow and Route-Level Imbalance Analysis"):
        st.markdown("""   
                    The directional model uncovers imbalances in traffic flow, particularly on specific streets. This finding supports initiatives for:

                    - ✅ Targeted infrastructure on routes with high one-way bias

                    - ✅ Redistribution of bike-sharing systems

                    - ✅ Prioritized urban design interventions in asymmetrically used corridors
            """)      
    st.markdown("""
                Together, these analyses provide a holistic view of urban cycling dynamics, empowering planners to make data-informed decisions for a smarter, more cyclist-friendly Paris.
                """)