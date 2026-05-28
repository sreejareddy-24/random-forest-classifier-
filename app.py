import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction")

st.write("Random Forest Classification Model")

# Inputs
age = st.slider("Age", 20, 100, 30)

sex = st.selectbox("Sex", [0, 1])

cp = st.slider("Chest Pain Type", 0, 3, 1)

trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)

chol = st.slider("Cholesterol", 100, 600, 200)

fbs = st.selectbox("Fasting Blood Sugar", [0, 1])

restecg = st.selectbox("Rest ECG", [0, 1, 2])

thalach = st.slider("Max Heart Rate", 60, 220, 150)

exang = st.selectbox("Exercise Induced Angina", [0, 1])

oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

slope = st.selectbox("Slope", [0, 1, 2])

ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])

thal = st.selectbox("Thal", [0, 1, 2, 3])

# Prediction
if st.button("Predict"):

    data = np.array([[
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal
    ]])

    # Scale input
    data_scaled = scaler.transform(data)

    # Prediction
    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease")