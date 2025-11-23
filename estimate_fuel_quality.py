#!/usr/bin/env python3
"""
estimate_fuel_quality.py

Updated to support model/scaler files provided either as:
 - plain joblib files (e.g. scaler.joblib)
 - or zip archives that contain joblib/pickle artifacts (e.g. rf_model.zip)

Place this file in the repo root alongside (or point the constants to) your artifacts:
  - scaler.joblib  OR scaler.zip (containing scaler.joblib)
  - pls_model.joblib OR pls_model.zip
  - rf_model.joblib OR rf_model.zip
  - diesel_properties_clean.xlsx
  - diesel_spec.xlsx
"""

import json
import sys
import os
import tempfile
import shutil
from typing import Dict, Any, Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
import zipfile

# ---- CONFIG ----
ROOT = "."  # repository root where everything lives
SCALER_PATH = os.path.join(ROOT, "scaler.joblib")      # or scaler.zip
PLS_MODEL_PATH = os.path.join(ROOT, "pls_model.joblib")# or pls_model.zip
RF_MODEL_PATH = os.path.join(ROOT, "rf_model.joblib")  # or rf_model.zip
PROPERTIES_XLSX = os.path.join(ROOT, "diesel_properties_clean.xlsx")
SPECS_XLSX = os.path.join(ROOT, "diesel_spec.xlsx")

# If you uploaded a zip to a different path (e.g. /mnt/data/rf_model.zip),
# set the constant accordingly or pass a full path in your environment.
# Example (environment): RF_MODEL_PATH = "/mnt/data/rf_model.zip"

FEATURE_ORDER = [
    "Density_15", "Viscosity", "Sulfur", "Ash", "FlashPoint", "Distillation_BPoint",
    "Aromatics", "PAH", "Cetane_Index", "API"
]

TARGETS = ["CN", "BP50", "FREEZE"]

# ---- UTIL: load joblib or extract from zip ----

def _find_candidate_in_zip(z: zipfile.ZipFile, candidates_ext=(".joblib", ".pkl", ".sav")) -> Optional[str]:
    """
    Return the first file in the zip that ends with one of candidates_ext (case-insensitive).
    """
    for name in z.namelist():
        lname = name.lower()
        for ext in candidates_ext:
            if lname.endswith(ext):
                return name
    return None

def safe_load(path: str, inner_filename: Optional[str] = None):
    """
    Load a joblib/pickle object from either:
      - a plain file path (joblib/pkl), or
      - a zip file containing the serialized artifact.
    If path is a zip and inner_filename is None, the first matching candidate (.joblib/.pkl/.sav)
    inside the zip will be loaded.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

    # If it's a zip file: open, extract candidate file to tempdir, load and cleanup.
    if path.lower().endswith(".zip"):
        tmpdir = tempfile.mkdtemp(prefix="model_zip_")
        try:
            with zipfile.ZipFile(path, "r") as z:
                target = inner_filename
                if target is None:
                    target = _find_candidate_in_zip(z)
                    if target is None:
                        raise FileNotFoundError(f"No .joblib/.pkl/.sav artifact found inside zip: {path}")
                # Extract the selected file
                z.extract(member=target, path=tmpdir)
                extracted_path = os.path.join(tmpdir, target)
                # If the member is stored with directories inside the zip, ensure path exists
                # joblib.load accepts filenames; use extracted_path
                return joblib.load(extracted_path)
        finally:
            # remove the temporary directory and its contents
            shutil.rmtree(tmpdir, ignore_errors=True)

    # Not a zip — load directly
    return joblib.load(path)

def load_models_and_scaler() -> Tuple[Any, Any, Any]:
    """Load scaler, pls, rf models from repo root. Supports zipped artifacts too."""
    scaler = None
    pls = None
    rf = None
    try:
        scaler = safe_load(SCALER_PATH)
    except Exception as e:
        print(f"[warning] Could not load scaler at {SCALER_PATH}: {e}")

    try:
        pls = safe_load(PLS_MODEL_PATH)
    except Exception as e:
        print(f"[warning] Could not load PLS model at {PLS_MODEL_PATH}: {e}")

    try:
        rf = safe_load(RF_MODEL_PATH)
    except Exception as e:
        print(f"[warning] Could not load RF model at {RF_MODEL_PATH}: {e}")

    return scaler, pls, rf

# ---- rest of the script (unchanged logic, with minor robustness tweaks) ----

def read_specs(specs_path: str = SPECS_XLSX) -> pd.DataFrame:
    """Read specs Excel and return a DataFrame."""
    if not os.path.exists(specs_path):
        raise FileNotFoundError(f"Spec file not found: {specs_path}")
    df = pd.read_excel(specs_path)
    df.columns = [c.strip() for c in df.columns]
    return df

def prepare_feature_array(input_dict: Dict[str, Any], feature_order: List[str]) -> np.ndarray:
    arr = []
    for f in feature_order:
        v = input_dict.get(f, np.nan)
        arr.append(np.nan if v is None else v)
    return np.array(arr, dtype=float).reshape(1, -1)

def impute_with_model(X_row: np.ndarray, scaler, model) -> float:
    X = X_row.copy()
    if np.isnan(X).any():
        # try to use dataset column means
        try:
            df = pd.read_excel(PROPERTIES_XLSX)
            col_means = {}
            for i, col in enumerate(FEATURE_ORDER):
                if col in df.columns:
                    col_means[i] = float(df[col].dropna().mean())
                else:
                    col_means[i] = 0.0
            for i in range(X.shape[1]):
                if np.isnan(X[0, i]):
                    X[0, i] = col_means.get(i, 0.0)
        except Exception:
            X = np.nan_to_num(X, nan=0.0)

    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            # if scaler fails, proceed without scaling
            pass

    y = model.predict(X)
    if np.asarray(y).ndim > 1:
        y = np.asarray(y).ravel()
        return float(y[0])
    return float(y)

def predict_missing_properties(input_dict: Dict[str, Any], scaler, pls, rf) -> Dict[str, Any]:
    filled = input_dict.copy()
    predicted = {}
    missing = [t for t in TARGETS if t not in filled or filled.get(t) is None or pd.isna(filled.get(t))]
    if not missing:
        return filled

    X = prepare_feature_array(filled, FEATURE_ORDER)
    model_used = rf if rf is not None else pls if pls is not None else None
    if model_used is None:
        print("[warning] No model available for imputation (RF/PLS missing).")
        return filled

    try:
        pred = impute_with_model(X, scaler, model_used)
        if isinstance(pred, (list, np.ndarray)) and len(pred) == len(TARGETS):
            for i, t in enumerate(TARGETS):
                if t in missing:
                    filled[t] = float(pred[i])
                    predicted[t] = float(pred[i])
        else:
            first = missing[0]
            filled[first] = float(pred)
            predicted[first] = float(pred)
    except Exception as e:
        print(f"[warning] Imputation attempt failed: {e}")

    filled["_imputed"] = predicted
    filled["_imputation_model"] = ("rf" if rf is not None else "pls") if model_used is not None else None
    return filled

def evaluate_against_specs(fuel_params: Dict[str, Any], specs_df: pd.DataFrame) -> Dict[str, Any]:
    result = {}
    passed = 0
    total = 0
    param_col = None
    for cand in ["Parameter", "parameter", "Param", "param", "Name", "name"]:
        if cand in specs_df.columns:
            param_col = cand
            break
    if param_col is None:
        param_col = specs_df.columns[0]

    min_col = next((c for c in specs_df.columns if "min" in c.lower()), None)
    max_col = next((c for c in specs_df.columns if "max" in c.lower()), None)

    for _, row in specs_df.iterrows():
        param = str(row[param_col]).strip()
        if param == "" or param.lower() == "nan":
            continue
        total += 1
        val = fuel_params.get(param)
        entry = {"value": val, "pass": None, "reason": None}
        if pd.isna(val) or val is None:
            entry["pass"] = False
            entry["reason"] = "missing"
        else:
            ok = True
            reasons = []
            if min_col and not pd.isna(row[min_col]):
                try:
                    if float(val) < float(row[min_col]):
                        ok = False
                        reasons.append(f"below min {row[min_col]}")
                except Exception:
                    pass
            if max_col and not pd.isna(row[max_col]):
                try:
                    if float(val) > float(row[max_col]):
                        ok = False
                        reasons.append(f"above max {row[max_col]}")
                except Exception:
                    pass
            entry["pass"] = ok
            entry["reason"] = "; ".join(reasons) if reasons else "within spec"
            if ok:
                passed += 1
        result[param] = entry

    score = (passed / total * 100.0) if total > 0 else None
    return {"per_parameter": result, "passed": passed, "total": total, "score_percent": score}

def estimate_quality(input_dict: Dict[str, Any], scaler, pls, rf, specs_path: str = SPECS_XLSX) -> Dict[str, Any]:
    fuel = input_dict.copy()
    fuel_filled = predict_missing_properties(fuel, scaler, pls, rf)
    try:
        specs_df = read_specs(specs_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read spec file: {e}")
    evaluation = evaluate_against_specs(fuel_filled, specs_df)
    res = {
        "input_original": input_dict,
        "filled_input": fuel_filled,
        "evaluation": evaluation
    }
    return res

def example_usage():
    print("Example usage (python script):")
    print("  python estimate_fuel_quality.py demo")
    print("  or import and call estimate_quality() from another script.")

def demo_run():
    print("Running demo with example inputs...")
    scaler, pls, rf = load_models_and_scaler()
    sample_input = {
        "Density_15": 835.0,
        "Viscosity": 3.8,
        "Sulfur": 12.0,
    }
    res = estimate_quality(sample_input, scaler, pls, rf)
    print(json.dumps(res, indent=2, default=str))

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1].lower() == "demo":
        demo_run()
    elif len(sys.argv) >= 2 and sys.argv[1].lower() == "help":
        example_usage()
    else:
        print("estimate_fuel_quality.py - updated to support zipped model artifacts.")
        print("Expected files (either .joblib/.pkl or .zip containing them):")
        print(f"  SCALER: {SCALER_PATH}")
        print(f"  PLS:    {PLS_MODEL_PATH}")
        print(f"  RF:     {RF_MODEL_PATH}")
        print("")
        print("To run demo: python estimate_fuel_quality.py demo")
