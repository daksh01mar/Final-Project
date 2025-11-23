#!/usr/bin/env python3
"""
estimate_fuel_quality.py

Loads scaler/PLS/RF models (supports .joblib or .zip containing joblib),
reads dataset header to determine feature order, imputes missing targets,
and evaluates against specs.
Place this file in repo root alongside:
  - scaler.joblib  OR scaler.zip
  - pls_model.joblib OR pls_model.zip
  - rf_model.joblib OR rf_model.zip
  - diesel_properties_clean.xlsx
  - diesel_spec.xlsx
"""

import os
import joblib
import zipfile
import tempfile
import shutil
from typing import Any, Tuple, Dict, List, Optional
import numpy as np
import pandas as pd

# ------- Paths (repo root) -------
ROOT = "."
SCALER_PATH = os.path.join(ROOT, "scaler.joblib")
PLS_MODEL_PATH = os.path.join(ROOT, "pls_model.joblib")
RF_MODEL_PATH = os.path.join(ROOT, "rf_model.joblib")  # or rf_model.zip
PROPERTIES_XLSX = os.path.join(ROOT, "diesel_properties_clean.xlsx")
SPECS_XLSX = os.path.join(ROOT, "diesel_spec.xlsx")

# Default target names in your dataset (from your header)
TARGETS = ["CN", "BP50", "FREEZE"]

# ----------------- Helpers to load joblib or from zip -----------------

def _find_candidate_in_zip(z: zipfile.ZipFile, ext_choices=(".joblib", ".pkl", ".sav")) -> Optional[str]:
    for name in z.namelist():
        if name.lower().endswith(ext_choices):
            return name
    return None

def safe_load(path: str, inner_filename: Optional[str] = None):
    """Load a joblib/pickle from a file or from inside a zip archive."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.lower().endswith(".zip"):
        tmpdir = tempfile.mkdtemp(prefix="model_zip_")
        try:
            with zipfile.ZipFile(path, "r") as z:
                target = inner_filename or _find_candidate_in_zip(z)
                if target is None:
                    raise FileNotFoundError(f"No joblib/pkl found inside zip: {path}")
                z.extract(member=target, path=tmpdir)
                extracted_path = os.path.join(tmpdir, target)
                return joblib.load(extracted_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return joblib.load(path)

def load_models_and_scaler() -> Tuple[Any, Any, Any]:
    """Attempt to load scaler, pls, rf from repo root (supports zip)."""
    scaler = pls = rf = None
    # attempt scaler
    for p in [SCALER_PATH, SCALER_PATH.replace(".joblib", ".zip")]:
        if os.path.exists(p):
            try:
                scaler = safe_load(p)
                break
            except Exception as e:
                print(f"[warning] cannot load scaler from {p}: {e}")
    # attempt pls
    for p in [PLS_MODEL_PATH, PLS_MODEL_PATH.replace(".joblib", ".zip")]:
        if os.path.exists(p):
            try:
                pls = safe_load(p)
                break
            except Exception as e:
                print(f"[warning] cannot load pls from {p}: {e}")
    # attempt rf
    for p in [RF_MODEL_PATH, RF_MODEL_PATH.replace(".joblib", ".zip")]:
        if os.path.exists(p):
            try:
                rf = safe_load(p)
                break
            except Exception as e:
                print(f"[warning] cannot load rf from {p}: {e}")

    return scaler, pls, rf

# ----------------- Feature order detection -----------------

def get_feature_order_from_properties(path_candidates: List[str] = None, drop_targets: List[str] = None) -> List[str]:
    """
    Read the header row of diesel_properties_clean.xlsx and return columns
    excluding the TARGETS (drop_targets).
    """
    if path_candidates is None:
        path_candidates = [PROPERTIES_XLSX, os.path.join("/mnt/data", "diesel_properties_clean.xlsx")]
    if drop_targets is None:
        drop_targets = TARGETS

    p = next((pp for pp in path_candidates if os.path.exists(pp)), None)
    if p is None:
        # fallback minimal guess
        print("[warning] properties file not found; using fallback feature order.")
        return [c for c in ["D4052", "FLASH", "TOTAL", "VISC"] if c not in drop_targets]

    df = pd.read_excel(p, nrows=0)
    cols = list(df.columns)
    # default behavior: features are all columns except the targets
    features = [c for c in cols if c not in drop_targets]
    return features

# Determine FEATURE_ORDER at import time
FEATURE_ORDER = get_feature_order_from_properties()

# ----------------- Prediction / imputation -----------------

def prepare_feature_array(input_dict: Dict[str, Any], feature_order: List[str]) -> np.ndarray:
    arr = []
    for f in feature_order:
        v = input_dict.get(f, np.nan)
        arr.append(np.nan if v is None else v)
    return np.array(arr, dtype=float).reshape(1, -1)

def _fill_nan_with_means(X: np.ndarray, properties_path: str = PROPERTIES_XLSX) -> np.ndarray:
    """If X has NaNs, try to replace with column means from properties file; else zeros."""
    if not np.isnan(X).any():
        return X
    try:
        df = pd.read_excel(properties_path)
        means = []
        for i, col in enumerate(FEATURE_ORDER):
            if col in df.columns:
                means.append(float(df[col].dropna().mean()))
            else:
                means.append(0.0)
        Xf = X.copy()
        for i in range(X.shape[1]):
            if np.isnan(Xf[0, i]):
                Xf[0, i] = means[i]
        return Xf
    except Exception:
        return np.nan_to_num(X, nan=0.0)

def impute_with_model(X_row: np.ndarray, scaler, model) -> float:
    X = _fill_nan_with_means(X_row)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass
    y = model.predict(X)
    if np.asarray(y).ndim > 1:
        y = np.asarray(y).ravel()
        return float(y[0])
    return float(y)

def predict_missing_properties(input_dict: Dict[str, Any], scaler, pls, rf) -> Dict[str, Any]:
    """
    Use RF if available else PLS to predict missing targets (CN/BP50/FREEZE).
    Uses FEATURE_ORDER discovered from the properties file.
    """
    filled = dict(input_dict)
    predicted = {}
    missing = [t for t in TARGETS if t not in filled or filled.get(t) is None or pd.isna(filled.get(t))]
    if not missing:
        filled["_imputed"] = {}
        filled["_imputation_model"] = None
        return filled

    X = prepare_feature_array(filled, FEATURE_ORDER)
    model = rf if rf is not None else pls if pls is not None else None
    if model is None:
        print("[warning] No imputation model available (rf/pls missing).")
        filled["_imputed"] = {}
        filled["_imputation_model"] = None
        return filled

    try:
        pred = impute_with_model(X, scaler, model)
        # If model returns multiple outputs, map to TARGETS if lengths match
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
        print(f"[warning] imputation failed: {e}")

    filled["_imputed"] = predicted
    filled["_imputation_model"] = ("rf" if rf is not None else "pls") if model is not None else None
    return filled

# ----------------- Spec evaluation -----------------

def read_specs(specs_path: str = SPECS_XLSX) -> pd.DataFrame:
    if not os.path.exists(specs_path):
        # try /mnt/data fallback
        alt = os.path.join("/mnt/data", os.path.basename(specs_path))
        if os.path.exists(alt):
            specs_path = alt
        else:
            raise FileNotFoundError(f"Spec file not found at {specs_path}")
    df = pd.read_excel(specs_path)
    df.columns = [c.strip() for c in df.columns]
    return df

def evaluate_against_specs(fuel_params: Dict[str, Any], specs_df: pd.DataFrame) -> Dict[str, Any]:
    result = {}
    passed = 0
    total = 0
    # find parameter column
    param_col = next((c for c in specs_df.columns if c.lower() in ("parameter","param","name")), specs_df.columns[0])
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

# ----------------- High-level API -----------------

def estimate_quality(input_dict: Dict[str, Any], scaler, pls, rf, specs_path: str = SPECS_XLSX) -> Dict[str, Any]:
    fuel_filled = predict_missing_properties(input_dict, scaler, pls, rf)
    specs_df = read_specs(specs_path)
    evaluation = evaluate_against_specs(fuel_filled, specs_df)
    return {"input_original": input_dict, "filled_input": fuel_filled, "evaluation": evaluation}

# For quick tests
if __name__ == "__main__":
    sc, pl, rf = load_models_and_scaler()
    print("FEATURE_ORDER:", FEATURE_ORDER)
    sample = {"D4052": 830.0, "FLASH": 60.0, "TOTAL": 12.0, "VISC": 3.5}
    res = estimate_quality(sample, sc, pl, rf)
    import json
    print(json.dumps(res, indent=2, default=str))
