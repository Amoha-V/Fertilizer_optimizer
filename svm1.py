import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Step 1: Load the dataset
file_path = 'f2.csv'  # Replace with your file path if necessary
df = pd.read_csv(file_path)

# Step 2: Handle missing values (if any)
df = df.dropna()

# Step 3: Encode categorical variables
label_encoders = {}
for column in ['Soil_Type', 'Crop_Type']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Encode target variable
fertilizer_encoder = LabelEncoder()
df['Fertilizer'] = fertilizer_encoder.fit_transform(df['Fertilizer'])

# Step 4: Define features and target variable
X = df.drop('Fertilizer', axis=1)
y = df['Fertilizer']

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Standardize the features (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train the Support Vector Machine model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Step 8: Define a function to get user input and predict fertilizer
def predict_fertilizer():
    # Collect user input
    temperature = float(input("Enter the temperature: "))
    humidity = float(input("Enter the humidity: "))
    moisture = float(input("Enter the moisture: "))
    soil_type = input("Enter the soil type (e.g., Clayey, Loamy, Red, Black, Sandy): ")
    crop_type = input("Enter the crop type (e.g., rice, wheat, tobacco, sugarcane, pulses, pomegranate, etc.): ")
    nitrogen = float(input("Enter the amount of nitrogen: "))
    potassium = float(input("Enter the amount of potassium: "))
    phosphorous = float(input("Enter the amount of phosphorous: "))
    
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
    
    print(f"The recommended fertilizer is: {fertilizer[0]}")

# Call the function to predict fertilizer based on user input
predict_fertilizer()

# Optional: Evaluate the model on the test set
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Display evaluation results
print(f"Test Accuracy: {accuracy}")
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print("Confusion Matrix:")
print(conf_matrix)
