import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image

@st.cache_data
def load_data():
    return pd.read_csv("data/bikes-paris.csv", sep=';')
    return df

# Page Title & Sidebar
st.title("🚲 Analysis of Bicycle Traffic in Paris")
st.sidebar.title("Contents")
pages = ["Introduction", "Data Exploration", "Data Visualization","Machine Learning", "Conclusions"]
page = st.sidebar.radio("Navigate to", pages)

#Load df 
df = load_data()
#Remove non used columns
columns_to_drop = [
    'ID Photos',
    'test_lien_vers_photos_du_site_de_comptage_',
    'id_photo_1',
    'url_sites',
    'type_dimage',
    'mois_annee_comptage',
    "Date d'installation du site de comptage",
    'Lien vers photo du site de comptage',
    'Identifiant technique compteur',
    'Identifiant du site de comptage',
    'Identifiant du compteur',
    'Nom du site de comptage'
]

# Ensure the columns exist in df_subset before attempting to drop them
df_dropped = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
#Translate column names

column_translation = {
    "Nom du compteur": "Meter Name",
    "Comptage horaire": "Hourly Count",
    "Date et heure de comptage": "Metering date and time",
    "Coordonnées géographiques": "Geographical Coordinates",
}

df_dropped.rename(columns=column_translation, inplace=True)
if page == pages[0] : 
  st.markdown("""
        ### ⭐ Introduction 
        Paris has made significant strides toward becoming a more bike-friendly city. As urban planners and policymakers aim to optimize infrastructure and promote sustainable mobility, understanding bicycle traffic patterns is essential. This project analyzes bicycle traffic data collected from automated counters across Paris with a dual focus: evaluating general hourly traffic volumes and identifying imbalances in directional flow across routes.
        """)
  st.subheader("Bicycle traffic map")


      

  df.columns = ['Meter identifier', 'Meter name', 'Metering site identifier', 'Name of metering site', 'Hourly count', 
                  'Metering date and time', 'Metering site installation date', 'Link to photo of metering site', 
                  'Geographical coordinates', 'Technical meter identifier', 'Photo ID', 'test link to photos of counting site', 
                  'ID photo 1', 'URL website', 'Image type', 'Month year count']

  # OKTOBER 2023 BIS SEPTEMBER 2024
  MAP_select_2324 = ['2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
  df_1023_0924 = df[df['Month year count'].str.contains('|'.join(MAP_select_2324))]
  MAP_one_year = df_1023_0924

  # Removing of the two outliers
  MAP_one_year = MAP_one_year[MAP_one_year['Hourly count'] != 8190]
  MAP_one_year = MAP_one_year[MAP_one_year['Hourly count'] != 2047]

  filtered_meterIDs = MAP_one_year.drop_duplicates(subset='Technical meter identifier')
  tecID_geo = filtered_meterIDs[['Technical meter identifier', 'Geographical coordinates']]
  Latitudes = tecID_geo['Geographical coordinates'].apply(lambda b: b.split(',')[0])
  Longitudes = tecID_geo['Geographical coordinates'].apply(lambda l: l.split(',')[1])
  lat_lon = pd.DataFrame({'Latitude': Latitudes, 'Longitude': Longitudes})
  df_tecID_geo = pd.concat([tecID_geo, lat_lon], axis=1)
  df_tecID_geo = df_tecID_geo.drop(columns=['Geographical coordinates'])
  df_tecID_geo = df_tecID_geo.rename(columns={'Technical meter identifier': 'Meter ID (technical)'})
  df_tecID_geo = df_tecID_geo.reset_index(drop=True)
  df_tecID_geo['Latitude'] = df_tecID_geo['Latitude'].astype(float)
  df_tecID_geo['Longitude'] = df_tecID_geo['Longitude'].astype(float)
  summed_counts = MAP_one_year.groupby('Technical meter identifier')['Hourly count'].sum().reset_index()
  summed_counts = summed_counts.rename(columns={'Technical meter identifier': 'Meter ID (technical)'})
  summed_counts.reset_index(drop=True)
  merged_df = df_tecID_geo.merge(summed_counts[['Meter ID (technical)', 'Hourly count']], on='Meter ID (technical)', how='left')
  merged_df['Hourly count'] = merged_df['Hourly count'].fillna(0)
  df_MAP = merged_df.rename(columns={'Hourly count': 'Total counts'})

  # Plotly
  fig = px.scatter_mapbox(
      df_MAP,
      lat='Latitude',
      lon='Longitude',
      hover_name='Meter ID (technical)', 
      color='Total counts',
      size='Total counts',
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

                After translation from French and cleaning, 16 columns were retained, providing temporal, spatial, and categorical features relevant to understanding traffic patterns. Columns like `Metering site installation date` and redundant photo identifiers were removed to enhance processing efficiency.



      
                  **Pre-processing**

                  The period from October 2023 to September 2024 was isolated to obtain an exact period of one year. Afterward, the index of the columns was reset. 

                  Two extreme values of `Hourly count` (8190 and 2047) have been filtered out. These high counts were both generated on October 22, 2023. On this date, bicycle traffic in Paris was particularly heavy, as the "Fête du Vélo", an annual festival in honor of the bicycle, took place on this day. These two values are not representative of the normal bicycle traffic.

                  Columns maintained:
      """)
    st.dataframe(df_dropped.columns)

if page == pages[2] :
  st.write("### 📈 Data Visualization")
  st.markdown("""
              To gain an initial understanding of bicycle behavior:

              - Daily traffic trends revealed a significant drop in weekend cycling, consistent with reduced commuting.

              - Seasonal traffic analysis showed summer months experiencing over twice the volume of winter, reinforcing weather's influence on biking behavior.

              - Monthly trends highlighted a dip in August due to holiday periods.

              - Meter installations over time pointed to spikes in 2018 and 2019, indicating strategic expansion phases.

              Additional visualizations focused on directional flow differences:

              - High-imbalance routes, such as Rue Turbigo and Quai de la Tournelle, exhibited strong directional biases.

              - Balanced routes, like Pont National, suggested symmetric flow, potentially due to bridge-based commutes.

              """)
if page == pages[3] :
  st.write("### ⚙️ Machine Learning")
  st.markdown("""
              Two separate ML pipelines were developed to address distinct goals.
            """)
  col1, col2 = st.columns(2)

  if col1.button("For Hourly Count Analysis"):
    st.markdown("""
                **1. Hourly Count Prediction**

                - Goal: Predict hourly bicycle traffic volume.

                - Models: RandomForestRegressor and XGBRegressor

                **Preprocessing:**

                - Extracted year, month, day, hour, and season from timestamps.

                - One-hot encoded categorical variables (e.g., location).

                - Scaled numerical features.

                **Results**

                - Random Forest outperformed XGBoost with an R² of 0.87 on the test set.

                - MAE: ~18 (Random Forest), ~27 (XGBoost)

                - MSE and cross-validation supported the robustness of Random Forest.

                **Feature Importance**

                - Time-based variables, especially `hour`, strongly influenced prediction accuracy.


      
                """)
  if col2.button("For Directional Flow and Route-Level Imbalance Analysis"):
      st.markdown("""
                  **2. Directional Flow Difference Prediction**

                  - Goal: Predict the difference in traffic volume between opposite directions on the same route.

                  - Models: RandomForestRegressor and Linear Regression

                  **Preprocessing:**

                  - Split Counter Name into Base Route and Direction

                  - Retained only routes with valid bidirectional data

                  - Created a new feature Difference

                  **Results:**

                  - Random Forest achieved an R² of 0.82, significantly outperforming Linear Regression (R² = 0.40).

                  - MAE and RMSE were less than half of those for Linear Regression.

                  **Feature Importance:** 

                  - `Hourly count` (`Comptage horaire`), hour of day, and coordinates were most influential.
                  """)
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