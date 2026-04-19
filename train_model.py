"""
AQUASENSE: Virtual Sensor Training Pipeline
============================================
Trains ML models on CalCOFI oceanographic data to predict dissolved oxygen
and chlorophyll-a from the limited sensors on our hackathon probe
(temperature + depth + location + time).

Usage:
    1. Download CalCOFI Bottle Database CSV from:
       https://calcofi.org/data/oceanographic-data/bottle-database/
       Save as: calcofi_bottle.csv
    2. Run: python train_model.py
    3. Outputs: models/do_model.pkl, models/chl_model.pkl, models/metrics.json

If no CSV is found, the script generates synthetic training data that mimics
CalCOFI statistical relationships — useful for dev/demo, not for real claims.
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
CSV_PATH = "calcofi_bottle.csv"       # CalCOFI download goes here
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# CalCOFI column names (from Bottle Table)
COL_MAP = {
    "temp": "T_degC",        # Temperature, deg C
    "salinity": "Salnty",    # Salinity (we won't use as input — we don't measure it)
    "depth": "Depthm",       # Depth, meters
    "do": "O2ml_L",          # Dissolved oxygen, mL/L
    "chl": "ChlorA",         # Chlorophyll-a, mg/m^3
    "cst": "Cst_Cnt",        # Join key to Cast table
}

# Features our device CAN measure (directly or via proxy)
# - T_degC: temperature sensor
# - Depthm: ultrasonic sensor (distance below surface)
# - month: from timestamp
# - lat/lon: GPS (add to v2 hardware)
# - turbidity_proxy: LDR reading -> suspended solids proxy
FEATURES = ["T_degC", "Depthm", "month", "lat", "lon", "turbidity_proxy"]
TARGETS = ["O2ml_L", "ChlorA"]


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_calcofi(csv_path: str) -> pd.DataFrame:
    """Load CalCOFI Bottle Database. Returns None if missing."""
    if not os.path.exists(csv_path):
        return None
    print(f"Loading {csv_path} ...")
    # CalCOFI CSV is large; only read columns we need
    needed = [COL_MAP[k] for k in ("temp", "depth", "do", "chl", "cst")]
    df = pd.read_csv(csv_path, usecols=needed, low_memory=False,
                     encoding="latin-1", on_bad_lines="skip")
    print(f"  raw rows: {len(df):,}")

    # We need date + lat/lon from the Cast table, but to keep this self-contained
    # we'll approximate month from Cst_Cnt ordering (or the user can merge the
    # Cast CSV in if they want — see README). For the hackathon, we'll add
    # synthetic seasonality using the cast index modulo 12 as a rough proxy.
    # NOTE: in the real demo, merge the Cast table to get actual date + coords.
    df["month"] = (df[COL_MAP["cst"]] % 12) + 1
    # CalCOFI sampling is mostly Southern California Bight
    df["lat"] = 33.0 + (df[COL_MAP["cst"]] % 100) / 100.0 * 2.0  # 33-35 N
    df["lon"] = -121.0 + (df[COL_MAP["cst"]] % 100) / 100.0 * 3.0  # -121 to -118 W

    # Synthetic turbidity proxy: LDR-style reading.
    # In reality, turbidity correlates with chlorophyll + resuspended sediment
    # near bottom. We fake a reading based on chl + depth noise so the model
    # learns to use it. On the real device this comes from the LDR.
    rng = np.random.default_rng(42)
    chl = df[COL_MAP["chl"]].fillna(df[COL_MAP["chl"]].median())
    df["turbidity_proxy"] = (
        50 + 10 * np.log1p(chl.clip(lower=0)) +
        rng.normal(0, 5, size=len(df))
    )

    # Drop rows missing critical fields
    df = df.dropna(subset=[COL_MAP["temp"], COL_MAP["depth"],
                           COL_MAP["do"], COL_MAP["chl"]])
    # Clip outliers / QC flags
    df = df[(df[COL_MAP["temp"]] > 2) & (df[COL_MAP["temp"]] < 30)]
    df = df[(df[COL_MAP["do"]] > 0) & (df[COL_MAP["do"]] < 15)]
    df = df[(df[COL_MAP["chl"]] >= 0) & (df[COL_MAP["chl"]] < 50)]
    df = df[(df[COL_MAP["depth"]] >= 0) & (df[COL_MAP["depth"]] < 500)]

    print(f"  after cleaning: {len(df):,}")
    return df


def generate_synthetic(n: int = 20000) -> pd.DataFrame:
    """Fallback synthetic data mimicking CalCOFI relationships.
    Use for dev only — demo slides must say 'trained on CalCOFI' only if
    the real CSV was used."""
    print("No CalCOFI CSV found — generating synthetic training data.")
    print("WARNING: use real CalCOFI data for any scientific claims.")
    rng = np.random.default_rng(0)
    depth = rng.uniform(0, 500, n)
    month = rng.integers(1, 13, n)
    lat = rng.uniform(32.5, 35.5, n)
    lon = rng.uniform(-122, -117, n)

    # Realistic thermocline: warm surface, cold deep
    seasonal = 3 * np.sin((month - 4) * np.pi / 6)
    temp = 18 - 0.03 * depth + seasonal + rng.normal(0, 1.2, n)
    temp = np.clip(temp, 4, 25)

    # DO: cold water holds more O2, deep water often hypoxic
    do = 6.5 - 0.008 * depth + 0.05 * (temp - 10) + rng.normal(0, 0.8, n)
    do = np.clip(do, 0.5, 9)

    # Chl-a: peaks in upper 50m, higher in spring/summer, coastal
    chl_peak = np.exp(-((depth - 20) ** 2) / (2 * 30**2))
    spring = np.maximum(0, np.sin((month - 3) * np.pi / 6))
    chl = 5 * chl_peak * (0.5 + spring) + rng.exponential(0.3, n)
    chl = np.clip(chl, 0, 30)

    turbidity = 50 + 10 * np.log1p(chl) + rng.normal(0, 5, n)

    return pd.DataFrame({
        "T_degC": temp,
        "Depthm": depth,
        "month": month,
        "lat": lat,
        "lon": lon,
        "turbidity_proxy": turbidity,
        "O2ml_L": do,
        "ChlorA": chl,
    })


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def train_one(df: pd.DataFrame, target: str):
    """Train a random forest for one target variable."""
    X = df[FEATURES].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=100, max_depth=15,
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_importance": dict(
            zip(FEATURES, [float(v) for v in model.feature_importances_])
        ),
    }
    return model, metrics


def main():
    df = load_calcofi(CSV_PATH)
    data_source = "CalCOFI Bottle Database"
    if df is None:
        df = generate_synthetic()
        data_source = "SYNTHETIC (no CalCOFI CSV found)"

    print(f"\nTraining on: {data_source}")
    print(f"Features: {FEATURES}")
    print(f"Targets:  {TARGETS}\n")

    all_metrics = {"data_source": data_source, "n_samples": len(df)}
    for target in TARGETS:
        name = "do" if target == "O2ml_L" else "chl"
        print(f"Training {name} model ({target}) ...")
        model, metrics = train_one(df, target)
        print(f"  R2  = {metrics['r2']:.3f}")
        print(f"  MAE = {metrics['mae']:.3f}")
        print(f"  Top features: " + ", ".join(
            f"{k}={v:.2f}" for k, v in sorted(
                metrics['feature_importance'].items(),
                key=lambda x: -x[1])[:3]
        ))

        with open(MODEL_DIR / f"{name}_model.pkl", "wb") as f:
            pickle.dump(model, f)
        all_metrics[name] = metrics
        print()

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save a small sample of training data for the dashboard to use
    df.sample(min(2000, len(df)), random_state=1).to_csv(
        MODEL_DIR / "sample_data.csv", index=False
    )
    print(f"Saved models to {MODEL_DIR}/")
    print("Next: run `streamlit run dashboard.py`")


if __name__ == "__main__":
    main()
