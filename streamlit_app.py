import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.write('This is a Machine Learning App for predicting weaning success in patients with sepsis')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/KariBB/MachineLearning/refs/heads/master/final_cleaned_dataset.csv')
  df
