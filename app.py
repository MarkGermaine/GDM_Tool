import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved preprocessor and model
preprocessor = joblib.load('preprocessor_gdm_2.pkl')
model = joblib.load('best_logistic_regression_model_2.pkl')

# Add Coombe Logo at the Top
st.image('coombe2.jpeg', use_container_width=True)

st.markdown("### Gestational Diabetes Prediction Tool")

# Collect Height and Weight, then calculate BMI
height_cm = st.number_input('Height (in cm)', min_value=100.0, max_value=250.0, step=0.1, value=None)
weight_kg = st.number_input('Weight (in kg)', min_value=30.0, max_value=200.0, step=0.1, value=None)

# Check if height and weight are entered before calculating BMI
if height_cm and weight_kg:
    bmi = weight_kg / ((height_cm / 100) ** 2)
    st.write(f'Calculated BMI: {bmi:.2f}')
else:
    bmi = None

# Collecting additional patient data
age_at_booking = st.number_input('Age at Booking', min_value=18, max_value=50, step=1, value=None)
systolic_bp = st.number_input('Systolic Blood Pressure at Booking', min_value=20, max_value=200, step=1, value=None)
diastolic_bp = st.number_input('Diastolic Blood Pressure at Booking', min_value=20, max_value=200, step=1, value=None)
parity = st.number_input('Parity', min_value=0, max_value=20, step=1, value=None)

hx_gdm = st.selectbox('History of Gestational Diabetes', ['Select', 'YES', 'NO'])
hx_gdm_numeric = 1 if hx_gdm == 'YES' else 0 if hx_gdm == 'NO' else None
fh_diabetes = st.selectbox('Family History of Diabetes', ['Select', 'YES', 'NO'])
ethnic_origin = st.selectbox('Ethnic Origin of Patient', ['Select','CAUCASIAN', 'SOUTH EAST ASIAN', 'OTHER', 'BLACK', 'ASIAN', 'MIDDLE EASTERN'])

    with st.expander("What does each ethnicity represent?"):
    st.markdown("""
    - **Caucasian**: All white Europeans and Northern Americans
    - **Black**: All Africans and Afro-Caribbean
    - **South East Asian**: Includes Pakistan, India, Malaysia, Singapore, etc.
    - **Asian**: Includes China, Japan, North and South Korea, Mongolia, etc.
    - **Middle Eastern**: Middle East and Northern African regions
    - **Other**: All other ethnicities (e.g., Latin American, Mixed, etc.)
    """)

other_endocrine_probs = st.selectbox('Other Endocrine Problems', ['Select', 'YES', 'NO'])
other_endocrine_probs_numeric = 1 if other_endocrine_probs == 'YES' else 0 if other_endocrine_probs == 'NO' else None

with st.expander("What are considered 'Other Endocrine Problems'?"):
    st.markdown("""
    Other endocrine problems include conditions such as:
    - **PCOS** (Polycystic Ovary Syndrome)
    - **Thyroid problems** (e.g., hypothyroidism, hyperthyroidism)
    - Any other endocrine-related disorders
    """)

# Ensure all fields are filled
if (height_cm and weight_kg and bmi and age_at_booking and systolic_bp and diastolic_bp and 
    parity is not None and hx_gdm_numeric is not None and fh_diabetes != 'Select' and ethnic_origin != 'Select' and 
    other_endocrine_probs_numeric is not None):
    
    if st.button('Predict Gestational Diabetes'):
        # Create DataFrame for input
        input_data = pd.DataFrame({
            'Ethnic Origin of Patient': [ethnic_origin],
            'Age at booking': [age_at_booking],
            'Hx_GDM': [hx_gdm_numeric],
            'BMI': [bmi],
            'FH Diabetes': [fh_diabetes],
            'Other Endocrine probs': [other_endocrine_probs_numeric],
            'Systolic BP at booking': [systolic_bp],
            'Diastolic BP at booking': [diastolic_bp],
            'Parity (not inc.multiple)': [parity]
        })
        
        # Apply preprocessing
        try:
            input_data_processed = preprocessor.transform(input_data)
            prediction = model.predict(input_data_processed)

            # Display the result
            if prediction == 1:
                st.error("Prediction: **HIGH Risk of Gestational Diabetes**")
                st.write("This result indicates a high risk of developing gestational diabetes. **Flag for OGTT at week 16**")
            else:
                st.success("Prediction: **LOW Risk of Gestational Diabetes**")
                st.write("This result indicates a low risk of developing gestational diabetes. Please follow regular prenatal care.")

        except Exception as e:
            st.write(f"Error during preprocessing or prediction: {str(e)}")
else:
    st.warning("Please fill out all the required fields before making a prediction.")

# Add CRT Machine Learning Banner at the Bottom
st.image('CRT Machine Learning Lock Up Banner 10-07-20.jpeg', use_container_width=True)
