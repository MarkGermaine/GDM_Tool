import streamlit as st
import joblib
import numpy as np
import pandas as pd
from io import StringIO
import boto3

# Load the saved preprocessor and model
preprocessor = joblib.load('preprocessor_gdm.pkl')
model = joblib.load('best_logistic_regression_model.pkl')


# Access AWS credentials securely from Streamlit secrets
ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
BUCKET_NAME = 'gdmtool'

# Initialize Boto3 S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

def upload_to_s3(data, file_name):
    csv_buffer = StringIO()
    data.to_csv(csv_buffer)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=file_name, Body=csv_buffer.getvalue())
    st.success(f'File {file_name} uploaded to S3 successfully!')

# Add Coombe Logo at the Top
st.image('coombe2.jpeg', use_column_width=True)  # Adjust the logo size based on the image dimensions

st.markdown("### Gestational Diabetes Prediction Tool")

# Capture unique identifier (optional, used for tracking purposes)
study_id = st.text_input('Study Participant ID', '')

# Collect Height and Weight, then calculate BMI
height_cm = st.number_input('Height (in cm)', min_value=100.0, max_value=250.0, step=0.1)
weight_kg = st.number_input('Weight (in kg)', min_value=30.0, max_value=200.0, step=0.1)

# Calculate BMI: BMI = weight (kg) / (height (m)^2)
bmi = weight_kg / ((height_cm / 100) ** 2)
st.write(f'Calculated BMI: {bmi:.2f}')

# Age at booking (stored as int64)
age_at_booking = st.number_input('Age at Booking', min_value=18, max_value=50, step=1)

# Systolic BP at booking (stored as int64)
systolic_bp = st.number_input('Systolic Blood Pressure at Booking', min_value=20, max_value=200, step=1)

# Diastolic BP at booking (stored as float64)
diastolic_bp = st.number_input('Diastolic Blood Pressure at Booking', min_value=20, max_value=200, step=1)

# Parity (stored as int64)
parity = st.number_input('Parity (excluding multiple)', min_value=0, max_value=20, step=1)

# Hx_GDM (stored as int64: 1 for YES, 0 for NO)
hx_gdm = st.selectbox('History of Gestational Diabetes (Hx_GDM)', ['YES', 'NO'])
hx_gdm_numeric = 1 if hx_gdm == 'YES' else 0


# FH Diabetes (stored as object)
fh_diabetes = st.selectbox('Family History of Diabetes (FH Diabetes)', ['YES', 'NO'])

# Ethnic origin selection (stored as object in model)
ethnic_origin = st.selectbox('Ethnic Origin of Patient', 
                             ['CAUCASIAN', 'SOUTH EAST ASIAN', 'OTHER', 'BLACK', 'ASIAN', 'MIDDLE EASTERN'])
with st.expander("What does each ethnicity represent?"):
    st.markdown("""
    - **Caucasian**: All white Europeans and Northern Americans
    - **Black**: All Africans and Afro-Caribbean
    - **South East Asian**: Includes Pakistan, India, Malaysia, Singapore, etc.
    - **Asian**: Includes China, Japan, North and South Korea, Mongolia, etc.
    - **Middle Eastern**: Middle East and Northern African regions
    - **Other**: All other ethnicities (e.g., Latin American, Mixed, etc.)
    """)


# Skill level (stored as int64)
skill_level_display = st.selectbox('Skill Level', 
                                   ['0 - Unemployed', 
                                    '1 - Elementary occupations', 
                                    '2 - Clerical support workers/Skilled workers/Assemblers',
                                    '3 - Technicians', 
                                    '4 - Managers and Professionals'])
with st.expander("What do the skill levels represent?"):
    st.markdown("""
    - **Skill Level 4**: Professionals and Managers (ISCO classification)
    - **Skill Level 3**: Technicians
    - **Skill Level 2**: Clerical Workers, Service/Sales workers, Skilled agriculture/forestry/fishery, Crafts and tradesmen, and Plant machine operators
    - **Skill Level 1**: Elementary occupations
    - **Skill Level 0**: Unemployed
    """)

skill_level = int(skill_level_display.split(" ")[0])

# Other Endocrine problems (stored as int64: 1 for YES, 0 for NO)
other_endocrine_probs = st.selectbox('Other Endocrine Problems', ['YES', 'NO'])
other_endocrine_probs_numeric = 1 if other_endocrine_probs == 'YES' else 0
with st.expander("What are considered 'Other Endocrine Problems'?"):
    st.markdown("""
    Other endocrine problems include conditions such as:
    - **PCOS** (Polycystic Ovary Syndrome)
    - **Thyroid problems** (e.g., hypothyroidism, hyperthyroidism)
    - Any other endocrine-related disorders
    """)

# New Section for Clinician's Prediction with expander
clinician_prediction = st.selectbox('Clinician Prediction of Gestational Diabetes', ['High Risk', 'Low Risk'])
clinician_prediction_value = 1 if clinician_prediction == 'High Risk' else 0  # 1 for High Risk, 0 for Low Risk

with st.expander("What should the clinician's prediction be based on?"):
    st.markdown("""
    The clinician should assess the patient's overall risk factors and make a prediction based on clinical judgment.
    This prediction should reflect whether the clinician believes the patient is at **High Risk** or **Low Risk** for developing GDM based on risk factors and clinical assessment.
    """)


# Button to make prediction
if st.button('Predict Gestational Diabetes'):
    # Create a DataFrame for the input data with the correct format and column names
    input_data = pd.DataFrame({
        'Ethnic Origin of Patient': [ethnic_origin],        # object
        'Age at booking': [age_at_booking],                # int64
        'Skill Level': [skill_level],                      # int64
        'Hx_GDM': [hx_gdm_numeric],                        # int64
        'BMI': [bmi],                                     # float64
        'FH Diabetes': [fh_diabetes],                      # object
        'Other Endocrine probs': [other_endocrine_probs_numeric],  # int64
        'Systolic BP at booking': [systolic_bp],           # int64
        'Diastolic BP at booking': [diastolic_bp],         # float64
        'Parity (not inc.multiple)': [parity]              # int64
    })
    
    # Apply the saved preprocessor to the input data
    try:
        input_data_processed = preprocessor.transform(input_data)

        # Make prediction using the saved model
        prediction = model.predict(input_data_processed)

        # Display the result
        if prediction == 1:
            st.error(f'Prediction for Study ID {study_id}: HIGH Risk of Gestational Diabetes')
            st.write("This result indicates a high risk of developing gestational diabetes. Flag for OGTT at week 16")
        else:
            st.success(f'Prediction for Study ID {study_id}: LOW Risk of Gestational Diabetes')
            st.write("This result indicates a low risk of developing gestational diabetes. Please follow regular prenatal care.")
    
    except Exception as e:
        st.write(f"Error during preprocessing or prediction: {str(e)}")
    
    
    # Store only study_id, prediction (binary 0/1), and clinician's prediction in a new DataFrame for S3
    output_data = pd.DataFrame({
        'Study Participant ID': [study_id],
        'Prediction': [prediction],  # Storing 0 or 1 for the ML prediction
        'Clinician Prediction': [clinician_prediction_value]  # Storing 0 or 1 for clinician's prediction
    })

    # Upload to S3
    csv_file_name = f'GDM_prediction_{study_id}.csv'
    upload_to_s3(output_data, csv_file_name)  # Uploading only the required columns to S3

    # Convert full input DataFrame to CSV for download
    csv = input_data.to_csv(index=False)
    csv_file = StringIO(csv)

    # Download button for CSV file (includes full input data)
    st.download_button(
        label="Download Results as CSV",
        data=csv_file.getvalue(),
        file_name=f'GDM_prediction_{study_id}.csv',
        mime='text/csv'
    )



# Add CRT Machine Learning Banner at the Bottom
st.image('CRT Machine Learning Lock Up Banner 10-07-20.jpeg', use_column_width=True)
