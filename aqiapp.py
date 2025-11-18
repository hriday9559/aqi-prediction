# aqiapp.py — dynamic, robust, deployment-ready Streamlit app
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="AQI Predictor", layout="centered")

st.title("🌫️ AQI / PM2.5 Prediction (Robust App)")

# Load model
try:
    model = joblib.load("aqi_model.pkl")
except Exception as e:
    st.error("Failed to load model: " + str(e))
    st.stop()

# Helper: try to get feature names and expected features count
feature_names = None
n_features = None

# scikit-learn models often have these attributes
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
    n_features = len(feature_names)
elif hasattr(model, "n_features_in_"):
    n_features = int(model.n_features_in_)
    # We will guess names if feature names are not present
else:
    st.warning("Model does not expose feature information. App cannot auto-generate inputs.")
    st.write("You must re-train/save the model with scikit-learn estimator (so it has `n_features_in_` or `feature_names_in_`).")
    st.stop()

st.write(f"Model expects **{n_features}** features.")

# If feature names are known, use them. Else, guess common pollutant/weather names.
if feature_names:
    st.write("Detected feature names:", feature_names)
    input_names = feature_names
else:
    # Guess common feature groups by number
    if n_features == 3:
        input_names = ["pm1_0", "pm2_5", "pm10"]
    elif n_features == 5 or n_features == 6:
        # prefer [pm1_0, pm10, temperature, humidity, pressure, (maybe pm2_5 or extra)]
        input_names = ["pm1_0", "pm10", "temperature", "humidity", "pressure"]
        # If n_features == 6, append pm2_5 as last guess
        if n_features == 6:
            input_names.append("pm2_5")
    else:
        # Generic numbered features as a fallback
        input_names = [f"feature_{i+1}" for i in range(n_features)]

st.write("Enter values for the model inputs below:")

# Create numeric inputs dynamically
input_values = []
col_layout = st.columns(2)
for i, fname in enumerate(input_names):
    col = col_layout[i % 2]
    # Provide sensible defaults and labels
    default = 10.0
    step = 1.0
    if "temp" in fname.lower() or "temperature" in fname.lower():
        default = 25.0
        step = 0.1
    elif "humid" in fname.lower():
        default = 50.0
        step = 0.1
    elif "press" in fname.lower():
        default = 1000.0
        step = 0.1
    elif "pm" in fname.lower():
        default = 20.0
        step = 1.0

    val = col.number_input(f"{fname}", value=float(default), step=float(step), format="%.2f")
    input_values.append(val)

# Prediction button
if st.button("Predict"):
    data = np.array([input_values], dtype=float)

    # Double-check shape
    if data.shape[1] != n_features:
        st.error(f"Input shape mismatch: model expects {n_features} features but you provided {data.shape[1]}.")
        st.info("If this is unexpected, re-train your model with the exact features you want to use, and save it again.")
    else:
        try:
            pred = model.predict(data)
            # Single value -> display nicely
            if pred.ndim == 1:
                st.success(f"Prediction: {round(float(pred[0]), 3)}")
            else:
                st.success(f"Prediction array: {pred}")
        except Exception as e:
            st.error("Prediction failed: " + str(e))
            st.stop()

st.markdown("---")
st.info("If your prediction looks wrong, it's likely the model was trained on different features or a different order. "
        "Best practice: retrain the model and save using the exact feature array you want to collect in this app.")


