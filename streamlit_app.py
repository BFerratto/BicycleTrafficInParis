import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('bikes-paris.csv', sep=';')

st.title('Bicycle Traffic in Paris')
st.sidebar.title('Table of contents')
pages=['Introduction','Exploration', 'Data Visualization', 'Modelling']
page=st.sidebar.radio('Go to', pages)

if page == pages[0] : 
  st.write("### Presentation of data")
  st.dataframe(df.head())
  st.write(df.shape)
  st.dataframe(df.describe())

  if st.checkbox('Show NA'):
        st.dataframe(df.isna().sum())

if page == pages[1]:
    st.write('### DataVizualization')

   
if page == pages[2] :
  st.write("### Modelling")

 