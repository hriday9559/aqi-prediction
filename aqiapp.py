import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("aqi_model.pkl")

st.title("🌫️ Air Quality Index (AQI) Prediction App")

# ONLY the features model was trained on
pm1 = st.number_input("Enter PM 1.0", min_value=0.0, step=1.0)
pm25 = st.number_input("Enter PM 2.5", min_value=0.0, step=1.0)
pm10 = st.number_input("Enter PM 10", min_value=0.0, step=1.0)

if st.button("Predict AQI"):
    data = np.array([[pm1, pm25, pm10]])
    result = model.predict(data)[0]
    st.success(f"Predicted AQI: {result}")
