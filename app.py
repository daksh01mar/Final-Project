import streamlit as st
import os
import pandas as pd
from estimate_fuel_quality import load_models_and_scaler, estimate_quality, TARGETS, get_feature_order_from_properties

st.set_page_config(page_title="Fuel Quality Predictor", layout="centered")
st.title("Fuel Quality Predictor")

st.markdown("This app imputes missing CN/BP50/FREEZE values (if left blank) and evaluates fuel against specs.")

# Load models (shows status)
scaler, pls, rf = load_models_and_scaler()
if scaler is None and pls is None and rf is None:
    st.error("No models loaded. Make sure scaler.joblib, pls_model.joblib, and rf_model.zip/joblib are in the repo root.")
else:
    st.success("Models loaded (if any).")

# Determine features from diesel_properties_clean.xlsx
prop_candidates = ["./diesel_properties_clean.xlsx", "./diesel_properties_clean.xls", "/mnt/data/diesel_properties_clean.xlsx"]
prop_path = next((p for p in prop_candidates if os.path.exists(p)), None)
if prop_path is None:
    st.error("diesel_properties_clean.xlsx not found in repo root. The app needs it to determine input fields.")
    st.stop()

# Use the helper to get FEATURE_ORDER dynamically (drops TARGETS)
FEATURE_ORDER = get_feature_order_from_properties(path_candidates=[prop_path], drop_targets=TARGETS)

st.subheader("Enter measured properties (leave blank to impute)")
st.write("Detected feature columns from your dataset (these are the model inputs):")
st.write(FEATURE_ORDER)

# Collect inputs for features
inputs = {}
cols = st.columns(2)
for i, feat in enumerate(FEATURE_ORDER):
    with cols[i % 2]:
        v = st.text_input(f"{feat} (leave blank if unknown)", value="")
        if v is not None and str(v).strip() != "":
            try:
                inputs[feat] = float(v)
            except:
                st.warning(f"Could not parse {feat}; ignoring the provided value.")

st.subheader("Optional: provide targets (these will be used if you enter them; otherwise they'll be imputed)")
for t in TARGETS:
    v = st.text_input(f"{t} (optional)", value="")
    if v is not None and str(v).strip() != "":
        try:
            inputs[t] = float(v)
        except:
            st.warning(f"Could not parse {t}; ignoring the provided value.")

# Button to estimate
if st.button("Estimate Fuel Quality"):
    try:
        res = estimate_quality(inputs, scaler, pls, rf)
        st.success("Done.")
        st.subheader("Filled Input (after imputation)")
        st.json(res["filled_input"])
        st.subheader("Evaluation against specs")
        st.json(res["evaluation"])
        score = res["evaluation"].get("score_percent")
        if score is not None:
            st.metric("Overall score (%)", f"{score:.2f}")
    except Exception as e:
        st.error(f"Estimation failed: {e}")
