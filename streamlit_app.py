import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This is a Machine Learning App for predicting weaning success in patients with sepsis')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/KariBB/MachineLearning/refs/heads/master/final_cleaned_dataset.csv')
  df
  
  st.write('**X**')
  X = df.drop(columns=['subject_id', 'stay_id','weaning_success','age_group'], axis=1)
  X
  
  st.write('**y**')
  y = df.weaning_success
  y

with st.expander('Data Visualization'):
    st.scatter_chart(data=df, x='age', y='bmi', color='weaning_success')

# Data preparation
with st.sidebar: 
  st.header('Input features')
