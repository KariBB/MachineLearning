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
  # bmi,charlson_comorbidity_index,chronic_pulmonary_disease,congestive_heart_failure,dementia,severe_liver_disease,renal_disease,rheumatic_disease,diabetes,gcs_total,max_wbc,max_hemoglobin,max_platelets,max_creatinine,max_anion_gap,min_wbc,min_hemoglobin,min_platelets,min_creatinine,min_anion_gap,max_hr,max_map,max_resp_rate,max_spo2,max_temp,min_hr,min_map,min_resp_rate,min_spo2,min_temp,duration_imv_hours,duration_niv_hours,duration_other_niv_hours,weaning_success,age_group
  age = st.slider('age (years)',18,99)
  gender = st.selectbox('gender',('M','F'))
  bmi = st.number_input('BMI', min_value=12, max_value=80, value=25, step=0.1)
  
