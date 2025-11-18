import streamlit as st
import joblib
import numpy as np

st.title("🌫️ PM2.5 Prediction App (Using 5 Inputs)")

# Load the trained model
model = joblib.load("aqi_model.pkl")  # make sure this is your 5-feature model

# User inputs
pm1 = st.number_input("Enter PM1.0", min_value=0.0, step=1.0)
pm10 = st.number_input("Enter PM10", min_value=0.0, step=1.0)
temp = st.number_input("Enter Temperature (°C)", min_value=-50.0, max_value=60.0, step=0.1)
humidity = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
pressure = st.number_input("Enter Pressure (hPa)", min_value=800.0, max_value=1200.0, step=0.1)

if st.button("Predict PM2.5"):
    # Create input array with correct shape and dtype
    data = np.array([[pm1, pm10, temp, humidity, pressure]], dtype=float)
    st.write("Input data shape:", data.shape)
    result = model.predict(data)[0]
    st.success(f"Predicted PM2.5: {result:.2f}")
