import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Page Title & Sidebar
st.title("🚲 Analysis of Bicycle Traffic in Paris")
st.sidebar.title("Contents")
pages = ["Introduction", "Data Exploration", "Data Visualization","Machine Learning", "Conclusions"]
page = st.sidebar.radio("Navigate to", pages)

#Load df and standardize
df= pd.read_csv('data/bikes-paris.csv', sep=';')

if page == pages[0] : 
  st.markdown("""
        ### Introduction 
        Welcome to our **analysis of bicycle traffic in Paris**. This interactive project guides you through the key steps of exploring urban cycling data—from data collection to visualization and modeling—revealing meaningful insights about biking trends across the city.
        This data set contains detailed records of bicycle traffic in Paris, collected from various automated counters installed across the city. The data is provided as numerical, categorical, and cyclical time-based information about bicycle traffic in Paris as a table as a CSV file. The original language is French.
""")
st.dataframe(df.head())
st.markdown("""
## 🎯 Objectives

The main goal of this analysis is to **understand bicycle traffic patterns in Paris** and extract insights that can support:

- Urban mobility planning  
- Policy-making  
- Infrastructure development  

The period from **October 2023 to September 2024** was extracted from the dataset `"comptage_velo_donnees_compteurs.csv"` and analyzed for bicycle traffic behavior.

The analysis aims to:

- Map high and low bicycle traffic areas across Paris
- Explore traffic volume over different time periods (hours, days, months)
- Support safety improvements and bike station placement strategies
- Evaluate infrastructure needs for traffic redistribution and maintenance

The focus is on the **‘Hourly count’** variable, which is linked to specific meters and locations. Time and date are strong influencing factors, making temporal analysis essential.

### Specific Objectives:

- ✅ Check the applicability of Machine Learning algorithms to the dataset  
- ✅ Analyze daily cycling trends to identify overall usage patterns  
- ✅ Examine fluctuations across different timeframes (daily, weekly, monthly, seasonal)  
- ✅ Compare traffic flow on similar paths in opposite directions to detect imbalances  

By studying these factors, this analysis aims to contribute to more efficient **urban planning**, **sustainable transportation**, and **cycling infrastructure improvements** in Paris.          
     
    """)

if page == pages[1]:
    st.write('### 🔍 Exploration & Description')
    st.markdown("""
    ## 🔍 Exploration & Description

This dataset contains detailed records of bicycle traffic in Paris, collected from various automated counters installed across the city. The data is provided in numerical, categorical, and cyclical time-based formats and is stored as a CSV file. The original language of the data is French.

The dataset's primary purpose is to track the number of bicycles passing a specific counting station (meters) at a given time. The raw dataset contains **943,512 entries** divided into **16 columns**. Key attributes include:

- **Counter Identification**:  
  Unique identifiers for each bicycle counter (`Identifiant_du_compteur`, `Nom_du_compteur`).

- **Location Information**:  
  Names and identifiers of the counting sites (`Identifiant_du_site_de_comptage`, `Nom_du_site_de_comptage`, `Coordonnees_géographiques`).

- **Time-based Data**:  
  Exact timestamps of each measurement (`Date_et_heure_de_comptage`, `mois_annee_comptage`).

- **Traffic Count**:  
  Number of bicycles recorded per time interval (`Comptage_horaire`).

- **Metadata**:  
  Additional attributes such as installation dates, photo links, and image types.

**Renamed English Columns**:

1. Meter identifier  
2. Meter name  
3. Metering site identifier  
4. Name of metering site  
5. Hourly count  
6. Metering date and time  
7. Metering site installation date  
8. Link to photo of metering site  
9. Geographical coordinates  
10. Technical meter identifier  
11. Photo ID  
12. Test link to photos of counting site  
13. ID photo 1  
14. URL website  
15. Image type  
16. Month year count  

""")
   
if page == pages[2] :
  st.write("### Data Visualization")
  
if page == pages[3] :
  st.write("### Machine Learning")
  
if page == pages[4] :

    st.markdown("""
        ### **Conclusion & Key Recommendations**  

        The merged insights suggest that combining temporal modeling of traffic volumes with spatial-directional analysis offers a fuller picture of Parisian cycling behavior, enabling data-driven improvements for sustainable urban mobility.  
        ### **Evaluations**        
    """)
      
    col1, col2 = st.columns(2)

    if col1.button("For Hourly Count Analysis"):
        st.markdown("""   
                    To predict the variable "Hourly Count," the two machine learning models, RandomForestRegressor and XGBRegressor, were selected to be trained on the preprocessed data. The selection of the models was justified by their ability to establish a relationship between the independent variables and the continuous target variable ("Hourly Count") and their general resistance to overfitting. The main quantitative performance metric was the R², the coefficient of determination that measured the proportion of variance in the dependent variable explained by the independent variables. R² served as a measure of the goodness of fit for the regression model. Additionally, the MAE (mean absolute error) and MSE (mean squared error) for both machine learning algorithms were compared. The RFR model explained about 98% of the variance of the target variable in the training data set, indicating it effectively captured the underlying patterns. However, its R² test score of approximately 0.86 suggested lower performance on unseen data, which implied some degree of overfitting, while the Mean Absolute Error revealed that the predictions deviated on average by about 18 units from the actual values. The XGBR model had a training R² value of 0.7848, which was lower than that of the RFR model, and its test score of 0.7817 indicated a good level of generalization without significant overfitting. In contrast, the RFR regressor demonstrated significantly lower MAEs in both training and testing phases, suggesting it performed better overall in all metrics and was more suitable for bike count prediction tasks than the XGBR. Five-fold cross-validations were performed on the test set for both models to obtain MSEs, resulting in an average MSE of 1722.2 for the RandomForestRegressor and 2575.2 for the XGBRegressor, which were similar to the MSE values calculated from the actual and predicted values, indicating that the RandomForestRegressor model demonstrated better performance in predicting the "Hourly count" variable due to its lower prediction error compared to the XGBRegressor. The feature importance of the three most significant features for both models was displayed, revealing that the RFR model was most influenced by the cyclical variable "hour," while the XGBR model was significantly affected by the locations represented by the "Name of metering site" variable, as the busiest areas corresponded to major roads in Paris with high bicycle density.
To improve the predictions of Machine Learning algorithms Hyperparameter can be adjusted.

The decrease in the average number of bicycles per weekday towards the weekend in the Figure 3 can be explained by the fact that the traveling to workplaces is reduced. It clearly shows that the bike traffic at the weekend is less than on workdays during the week. 
The seasonal and monthly bicycle traffic in Figure 4 indicates a strong weather dependence on bicycle traffic. More people use bikes to travel when the weather is good. The relatively small amount of bike traffic in August can be explained by to summer holidays. The population in general that stays in Paris during this month is smaller.
The meter installations per year in Figure 5 display how many meters were built in which year. This is important to know for example for maintenance services provided by the city. Assuming a general overhaul takes place every 2 years, most of the “counters” should have been serviced in 2020 and 2021, as most of them were installed in 2018 and 2019. 
Given that hourly bicycle counts, time-related variables, and locations are the most significant predictors, traffic management strategies should focus on peak-hour interventions. This could include adjusting bike lane availability during rush hours and implementing bike-sharing redistribution plans to address demand imbalances. The geographical influence suggests that particular routes need customized solutions, such as improving infrastructure in high-traffic areas and adding more bike-sharing stations near key locations. Some streets consistently hold significant importance, indicating that city planners should prioritize enhancing cycling infrastructure and managing traffic flow on these routes. To be able to continue to guarantee reliable bicycle traffic recording, the maintenance of the meters must be ensured. The installation periods of the individual meters can be used to estimate repair intervals.

         """)      

    if col2.button("For Directional Flow and Route-Level Imbalance Analysis"):
        st.markdown("""   
                    Complementing the hourly traffic analysis, the directional flow sub-study focused on differences in bicycle counts between opposing directions on the same routes. Using a refined dataset filtered to include only valid directional pairs and routes with complete data, the RandomForestRegressor and Linear Regression models were applied to predict these
imbalances. 
The Random Forest model outperformed Linear Regression, achieving an R² of 0.82 and much lower error rates (MAE: 7.96; RMSE: 19.86), as seen in Table 2. This confirmed the non-linear nature of directional imbalances, making Random Forest better suited for these
tasks. 
Key variables influencing directional traffic differences included total bicycle counts ("Hourly Count"), the hour of the day (“hour”),‘’weekday” (cyclical), and geographic location (“latitude”,“longitude”). The analysis showed that Rue Turbigo and Quai de la Tournelle had the most pronounced directional traffic imbalances, likely due to surrounding land use (e.g., business vs. residential zones). 
            
            #### **From an urban planning perspective, this highlights the need for:**  
            ✅ **Redistribution strategies to better balance bike availability on directionally imbalanced routes.  
            ✅ **Targeted infrastructure improvements in areas with high directional discrepancies.
            ✅ **Deployment of more bike-sharing stations in underserved directions or high-demand exit points.
            
         """)      