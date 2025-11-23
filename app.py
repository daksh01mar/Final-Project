import streamlit as st
import pandas as pd
import numpy as np
import os
from estimate_fuel_quality import load_models_and_scaler, estimate_quality, FEATURE_ORDER, TARGETS

st.set_page_config(page_title="Fuel Quality Predictor", layout="centered")

st.title("Fuel Quality Prediction App")

st.markdown("""
This app predicts missing fuel properties using trained ML models  
and evaluates the fuel quality against specification limits.
""")

# -------------------------------
# Load models + scaler
# -------------------------------
st.subheader("Model Loading Status")

scaler, pls, rf = load_models_and_scaler()

if scaler is None and pls is None and rf is None:
    st.error("❌ No models could be loaded. Ensure scaler.joblib, pls_model.joblib, rf_model.zip/joblib are in repo root.")
else:
    st.success("✅ Models loaded successfully.")


# -------------------------------
# Input Section
# -------------------------------
st.header("Enter Fuel Properties")

inputs = {}

# Inputs for normal features
for feature in FEATURE_ORDER:
    val = st.text_input(f"{feature} (leave blank if unknown)")
    if val.strip() != "":
        try:
            inputs[feature] = float(val)
        except:
            st.warning(f"Value for {feature} is invalid, ignoring.")

# Optional target inputs (CN, BP50, FREEZE)
st.subheader("Optional: Enter Target Properties (Will be predicted if left blank)")
for t in TARGETS:
    val = st.text_input(f"{t} (optional)")
    if val.strip() != "":
        try:
            inputs[t] = float(val)
        except:
            st.warning(f"Invalid value for {t}, ignoring.")


# -------------------------------
# Run Prediction
# -------------------------------
if st.button("Estimate Fuel Quality"):
    try:
        result = estimate_quality(inputs, scaler, pls, rf)

        st.success("Prediction Completed.")

        st.subheader("Filled Input (after imputing missing values)")
        st.json(result["filled_input"])

        st.subheader("Evaluation Against Specs")
        st.json(result["evaluation"])

        st.header(f"Overall Score: {result['evaluation']['score_percent']:.2f}%")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.markdown("---")
st.caption("Make sure all required .joblib/.zip/.xlsx files are in the GitHub repo root.")
