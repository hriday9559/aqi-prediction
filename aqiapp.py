import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("aqi_model.pkl")

st.title("🌫️ Air Quality Index (AQI) Prediction App")

st.write("Enter pollutant levels to predict AQI")

pm1 = st.number_input("Enter PM1.0", min_value=0.0, step=1.0)
pm25 = st.number_input("Enter PM2.5", min_value=0.0, step=1.0)
pm10 = st.number_input("Enter PM10", min_value=0.0, step=1.0)

if st.button("Predict AQI"):
    # MUST match the exact feature order used in training
    data = np.array([[pm1, pm25, pm10]])
    
    result = model.predict(data)[0]
    
    st.success(f"Predicted AQI: {round(result, 2)}")

