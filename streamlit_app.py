import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

#---------------------------------------------------------------------------------------------------------

st.title('🤖 Machine Learning App')

st.info('This is a Machine Learning App for predicting weaning success in patients with sepsis')
st.write('Our main goal is the Target: **Weaning Success (WS)** as:')
st.write('- The patient didn’t need to go back on a breathing tube (intubation) or another form of mechanical ventilation (like a ventilator) within 48 hours after being taken off.')
st.write('- The patient didn’t die within 48 hours after being weaned off the ventilator.')
st.write('If the patient needed noninvasive ventilation (like a CPAP machine), it had to be for less than 48 hours after weaning.')

st.info('Click on the left side to input your data **>**')

#-------------------------------------------------------------------------------------------------------

# ✅ Load the data at the top, so it's always available
df = pd.read_csv('https://raw.githubusercontent.com/KariBB/MachineLearning/refs/heads/master/final_cleaned_dataset.csv')
X = df.drop(columns=['subject_id', 'stay_id', 'weaning_success', 'age_group'], axis=1)
y = df.weaning_success

# Data display inside expander
with st.expander('Data'):
    st.write('**Raw data**')
    st.write(df)

    st.write('**Predictors (X)**')
    st.write(X)

    st.write('**Target Variable (y)**')
    st.write(y)

#------------------------------------------------------------------------------------------------------

# Histogram by age group
st.write("### Age Group vs Weaning Success")

with st.expander("Age Group vs Weaning Success"):
    if 'age_group' in df.columns and 'weaning_success' in df.columns:
        fig = px.histogram(df, 
                           x='age_group', 
                           color='weaning_success', 
                           barmode='group', 
                           title='Weaning Success by Age Group',
                           labels={'weaning_success': 'Weaning Success (0 = Failure, 1 = Success)', 
                                   'age_group': 'Age Group'},
                           category_orders={'weaning_success': [0, 1]})

        fig.update_layout(
            xaxis_title="Age Group",
            yaxis_title="Count of Patients",
            barmode='group',
            xaxis={'categoryorder': 'category ascending'}
        )

        st.plotly_chart(fig)
    else:
        st.warning("Columns 'age_group' or 'weaning_success' not found in the dataset.")
      
#------------------------------------------------------------------------------------------------------

# Columns to exclude from the histograms
columns_to_exclude = [
    'subject_id', 'stay_id', 'chronic_pulmonary_disease', 'congestive_heart_failure', 
    'dementia', 'severe_liver_disease', 'renal_disease', 'rheumatic_disease', 'diabetes', 
    'weaning_success', 'age_group'  # Added age_group to the exclusion list
]

# Filter only the numerical columns and exclude the ones in columns_to_exclude
numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
columns_to_plot = [col for col in numerical_columns if col not in columns_to_exclude]

# Plot histograms for each of the remaining numerical columns
with st.expander('Histograms for Numerical Variables'):
    for col in columns_to_plot:
        st.subheader(f'Histogram for {col}')
        
        # Creating the histogram plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(X[col], bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        
        # Display the plot in Streamlit
        st.pyplot(fig)

        # Close the figure after displaying to avoid memory issues
        plt.close(fig)

#------------------------------------------------------------------------------------------------------

# Data preparation
with st.sidebar: 
  st.header('Input features')
  #max_wbc,max_hemoglobin,max_platelets,max_creatinine,max_anion_gap,min_wbc,min_hemoglobin,min_platelets,min_creatinine,min_anion_gap,max_hr,max_map,max_resp_rate,max_spo2,max_temp,min_hr,min_map,min_resp_rate,min_spo2,min_temp,duration_imv_hours,duration_niv_hours,duration_other_niv_houress,age_group
  gender = st.selectbox('gender',('M','F'))
  age = st.slider('age (years)',18,99)
  gcs_total = st.slider('GCS Total',0,20)
  bmi = st.number_input('BMI', min_value=10.0, max_value=80.0, value=25.0, step=0.1)
  charlson_comorbidity_index = st.number_input('Charlson Combordity Index', min_value=0, max_value=20, value=10, step=1)
  # Diseases
  chronic_pulmonary_disease = st.selectbox('Chronic Pulmonary Disease',(0,1))
  congestive_heart_failure = st.selectbox('Congestive Heart Failure',(0,1))
  dementia = st.selectbox('Dementia',(0,1))
  severe_liver_disease = st.selectbox('Severe Liver Disease',(0,1))
  renal_disease = st.selectbox('Renal Disease',(0,1))
  rheumatic_disease = st.selectbox('Rheumatic Disease',(0,1))
  diabetes = st.selectbox('Diabetes',(0,1))
  # max'ss
  max_wbc = st.number_input('Max White Blood Cells (mEq/L)', min_value=0.0, max_value=30.0, value=20.0, step=0.1)
  max_hemoglobin = st.number_input('Max Hemoglobin (g/dL)', min_value=0.0, max_value=20.0, value=15.0, step=0.1)
  max_platelets = st.number_input('Max Platelets (10^3/mL)', min_value=0.0, max_value=1000.0, value=30.0, step=0.1)
  max_creatinine = st.number_input('Max Creatinine (mmol/L)', min_value=0.0, max_value=400.0, value=200.0, step=0.1)
  max_anion_gap = st.number_input('Max Anion Gap (mEq/L)', min_value=0.0, max_value=30.0, value=10.0, step=0.1)
  max_hr = st.number_input('Max Heart Rate (bpm)', min_value=40.0, max_value=200.0, value=70.0, step=0.1)
  max_map = st.number_input('Max Heart Rate (mmHg)', min_value=40.0, max_value=200.0, value=70.0, step=0.1)
  max_resp_rate = st.number_input('Max Resp Rate (breaths/minute)', min_value=10.0, max_value=50.0, value=30.0, step=0.1)
  max_spo2 = st.number_input('Max Spo2 (%)', min_value=0.0, max_value=100.0, value=70.0, step=0.1)
  max_temp = st.number_input('Max Temp (F)', min_value=60.0, max_value=100.0, value=70.0, step=0.1)

  
