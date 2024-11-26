import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model, scaler, and encoders
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
fertilizer_encoder = joblib.load('fertilizer_encoder.pkl')

# Read the fixed CSV file
file_path = 'f2.csv'  # Path to your fixed CSV file
df = pd.read_csv(file_path)

# Streamlit app
st.title('Fertilizer Recommendation System')

# Display the CSV file data
# st.write("Data from CSV file:")
# st.write(df.head())  # Display the first few rows of the data

# Get user input
temperature = st.number_input("Enter the temperature:", format="%.2f")
humidity = st.number_input("Enter the humidity:", format="%.2f")
moisture = st.number_input("Enter the moisture:", format="%.2f")
soil_type = st.selectbox("Select the soil type:", df['Soil_Type'].unique())
crop_type = st.selectbox("Select the crop type:", df['Crop_Type'].unique())
nitrogen = st.number_input("Enter the amount of nitrogen:", format="%.2f")
potassium = st.number_input("Enter the amount of potassium:", format="%.2f")
phosphorous = st.number_input("Enter the amount of phosphorous:", format="%.2f")

# Encode the user input
soil_type_encoded = label_encoders['Soil_Type'].transform([soil_type])[0]
crop_type_encoded = label_encoders['Crop_Type'].transform([crop_type])[0]

# Create a DataFrame for the input data
input_data = pd.DataFrame([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorous]], 
                          columns=['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous'])

# Standardize the input data
input_data = scaler.transform(input_data)

# Predict the fertilizer
prediction = svm_model.predict(input_data)

# Decode the prediction
fertilizer = fertilizer_encoder.inverse_transform(prediction)

# Display the prediction
st.write(f"The recommended fertilizer is: **{fertilizer[0]}**")
