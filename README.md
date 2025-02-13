# Diabetes-Prediction-App
This is a machine learning-based web application that predicts whether a person has diabetes based on their medical details. It uses Random Forest Classifier and is deployed with Streamlit.
🚀 Features
✅ User-Friendly UI – Enter patient details via an interactive form.
✅ Machine Learning Model – Uses Random Forest with hyperparameter tuning.
✅ Real-Time Predictions – Click "Predict" to see results instantly.
✅ Standardization Applied – Ensures model gets correctly scaled input.
📦 Diabetes-Prediction-App
│── 📜 app.py               # Main Streamlit app
│── 📜 diabetes.csv         # Dataset (Pima Indians Diabetes dataset)
│── 📜 rf_model.pkl         # Trained Random Forest model
│── 📜 scaler.pkl           # StandardScaler for input preprocessing
│── 📜 README.md            # Project documentation
│── 📜 requirements.txt     # Required dependencies

pip install -r requirements.txt
streamlit run app.py
🖥️ Usage
Enter patient details such as glucose level, BMI, insulin level, etc.
Click "Predict" to see if the person is diabetic or not.
The app will display a red warning (🔴) if the patient is diabetic and a green success message (🟢) if not.
📊 Dataset
This project uses the Pima Indians Diabetes Dataset, which contains 768 medical records with the following features:

Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI
Diabetes Pedigree Function
Age
Outcome (0 = Not Diabetic, 1 = Diabetic)
