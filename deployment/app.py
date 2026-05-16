
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ------------------------------------------------------------
# App Title
# ------------------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance App",
    layout="centered"
)

st.title("Predictive Maintenance for Engine Failure Detection")

st.write(
    """
    This application predicts whether an engine is operating normally or may require maintenance
    based on sensor readings such as RPM, pressure, and temperature.
    """
)

# ------------------------------------------------------------
# Load Model from Hugging Face Model Hub
# ------------------------------------------------------------

MODEL_REPO_ID = "chandrachurhghosh/predictive-maintenance-model"
MODEL_FILENAME = "best_xgboost_predictive_maintenance_model.pkl"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )
    model = joblib.load(model_path)
    return model

model = load_model()

# ------------------------------------------------------------
# User Inputs
# ------------------------------------------------------------

st.subheader("Enter Engine Sensor Values")

engine_rpm = st.number_input("Engine RPM", min_value=0.0, value=700.0)
lub_oil_pressure = st.number_input("Lubricating Oil Pressure", min_value=0.0, value=3.0)
fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0, value=5.0)
coolant_pressure = st.number_input("Coolant Pressure", min_value=0.0, value=2.0)
lub_oil_temp = st.number_input("Lubricating Oil Temperature", min_value=0.0, value=80.0)
coolant_temp = st.number_input("Coolant Temperature", min_value=0.0, value=75.0)

# IMPORTANT: Column names must match training data exactly
input_data = pd.DataFrame({
    "Engine_rpm": [engine_rpm],
    "Lub_oil_pressure": [lub_oil_pressure],
    "Fuel_pressure": [fuel_pressure],
    "Coolant_pressure": [coolant_pressure],
    "lub_oil_temp": [lub_oil_temp],
    "Coolant_temp": [coolant_temp]
})

st.subheader("Input Data")
st.dataframe(input_data)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

if st.button("Predict Engine Condition"):

    if hasattr(model, "predict_proba"):
        failure_probability = model.predict_proba(input_data)[0][1]

        threshold = 0.70

        if failure_probability > threshold:
            prediction = 1
        else:
            prediction = 0

        if prediction == 1:
            st.error("Prediction: Faulty Engine / Maintenance Required")
            st.write(f"Failure Probability: {failure_probability:.2%}")
            st.warning("Recommended Action: Schedule preventive maintenance immediately.")
        else:
            st.success("Prediction: Normal Engine Condition")
            st.write(f"Failure Probability: {failure_probability:.2%}")
            st.info("Recommended Action: Continue monitoring engine health.")

    else:
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("Prediction: Faulty Engine / Maintenance Required")
        else:
            st.success("Prediction: Normal Engine Condition")

st.info(
    "This prediction is intended to support proactive maintenance decisions and should be used along with engineering judgment."
)
