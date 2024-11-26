from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model, scaler, and encoders
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
fertilizer_encoder = joblib.load('fertilizer_encoder.pkl')

# Load the fertilizer data from the Excel file
file_path = 'fertilizer123.xlsx'
fertilizer_data  = pd.read_excel(file_path)

# Load the fixed CSV file for user options
file_path = 'f2.csv'
df = pd.read_csv(file_path)

@app.route('/')
def index():
    soil_types = df['Soil_Type'].unique()
    crop_types = df['Crop_Type'].unique()
    return render_template('index.html', soil_types=soil_types, crop_types=crop_types)

@app.route('/predict', methods=['POST'])
def predict():
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    moisture = float(request.form['moisture'])
    soil_type = request.form['soil_type']
    crop_type = request.form['crop_type']
    nitrogen = float(request.form['nitrogen'])
    potassium = float(request.form['potassium'])
    phosphorous = float(request.form['phosphorous'])

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

    # Find the fertilizer in the fertilizer data
    fertilizer_row = fertilizer_data[fertilizer_data['Fertilizer'].str.lower() == fertilizer.lower()]

    if not fertilizer_row.empty:
        # Calculate the amount of fertilizer needed based on the required N, P, and K
        n_ratio = fertilizer_row['N'].values[0]
        p_ratio = fertilizer_row['P'].values[0]
        k_ratio = fertilizer_row['K'].values[0]

        n_fertilizer = nitrogen / n_ratio if n_ratio != 0 else float('inf')
        p_fertilizer = phosphorous / p_ratio if p_ratio != 0 else float('inf')
        k_fertilizer = potassium / k_ratio if k_ratio != 0 else float('inf')

        recommended_quantity = max(n_fertilizer, p_fertilizer, k_fertilizer)

        if recommended_quantity == float('inf'):
            result = f"The fertilizer '{fertilizer}' cannot meet all nutrient requirements due to missing N, P, or K content."
        else:
            result = f"The recommended fertilizer is: {fertilizer}. To meet the nutrient requirements, use {recommended_quantity:.2f} units."
    else:
        result = f"Fertilizer '{fertilizer}' not found in the dataset."

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
