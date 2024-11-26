import pandas as pd
import streamlit as st
import joblib

# Load the pre-trained model, scaler, and encoders
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
fertilizer_encoder = joblib.load('fertilizer_encoder.pkl')

# Load the Excel file with fertilizer nutrient percentages
file_path = 'C:\\Users\\vamoh\\Downloads\\fertilizer123.xlsx'
df_fertilizer = pd.read_excel(file_path)

def calculate_fertilizer_amount(fertilizer_type, nutrient_type, nutrient_kg_per_ha):
    # Find the row corresponding to the given fertilizer type
    fertilizer_row = df_fertilizer[df_fertilizer['Fertilizer'].str.lower() == fertilizer_type.lower()]
    
    if fertilizer_row.empty:
        return f"Fertilizer type '{fertilizer_type}' not found."
    
    # Get the percentage of the specific nutrient in the fertilizer
    nutrient_percentage = fertilizer_row[nutrient_type].values[0]
    
    if nutrient_percentage == 0:
        return f"The fertilizer '{fertilizer_type}' does not contain the nutrient '{nutrient_type}'."
    
    # Calculate the amount of fertilizer needed
    fertilizer_amount = (nutrient_kg_per_ha / nutrient_percentage) * 100
    return fertilizer_amount

# Streamlit app
st.title('Fertilizer Recommendation System')

# Get user input
temperature = st.number_input("Enter the temperature:", format="%.2f")
humidity = st.number_input("Enter the humidity:", format="%.2f")
moisture = st.number_input("Enter the moisture:", format="%.2f")
soil_type = st.selectbox("Select the soil type:", label_encoders['Soil_Type'].classes_)
crop_type = st.selectbox("Select the crop type:", label_encoders['Crop_Type'].classes_)
nitrogen = st.number_input("Enter the amount of nitrogen required (kg/ha):", format="%.2f")
potassium = st.number_input("Enter the amount of potassium required (kg/ha):", format="%.2f")
phosphorous = st.number_input("Enter the amount of phosphorous required (kg/ha):", format="%.2f")

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
fertilizer = fertilizer_encoder.inverse_transform(prediction)[0]

# Display the prediction
st.write(f"The recommended fertilizer is: **{fertilizer}**")

# Calculate the amount of fertilizer required for each nutrient
if fertilizer:
    nitrogen_amount = calculate_fertilizer_amount(fertilizer, 'N', nitrogen)
    phosphorus_amount = calculate_fertilizer_amount(fertilizer, 'P', phosphorous)
    potassium_amount = calculate_fertilizer_amount(fertilizer, 'K', potassium)
    
    # Display the calculated amounts if they are numeric, otherwise show the error message
    if isinstance(nitrogen_amount, (int, float)):
        st.write(f"Amount of {fertilizer} needed for Nitrogen: **{nitrogen_amount:.2f} kg/ha**")
    else:
        st.write(nitrogen_amount)

    if isinstance(phosphorus_amount, (int, float)):
        st.write(f"Amount of {fertilizer} needed for Phosphorus: **{phosphorus_amount:.2f} kg/ha**")
    else:
        st.write(phosphorus_amount)

    if isinstance(potassium_amount, (int, float)):
        st.write(f"Amount of {fertilizer} needed for Potassium: **{potassium_amount:.2f} kg/ha**")
    else:
        st.write(potassium_amount)
