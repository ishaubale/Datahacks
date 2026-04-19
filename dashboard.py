"""
AquaSense Dashboard — Streamlit demo UI
========================================
Simulates live sensor readings from our probe, predicts oceanographic
conditions using models trained on CalCOFI, and shows fishery-relevant
outputs: habitat suitability per species, depth profile, acoustic variance.

Run with:   streamlit run dashboard.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from habitat_suitability import SPECIES, suitability_all

MODEL_DIR = Path("models")

# -------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AquaSense — Fisheries Intelligence",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .big-metric { font-size: 2.2rem; font-weight: 600; }
    .subtle { color: #666; font-size: 0.85rem; }
    .banner {
        background: linear-gradient(90deg, #0a3d62, #1e6091);
        color: white; padding: 1.2rem 1.5rem; border-radius: 8px;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='banner'>
      <h1 style='margin:0'>🌊 AquaSense</h1>
      <div>Low-cost probe → CalCOFI-calibrated virtual sensors → fishery intelligence</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Load models
# -------------------------------------------------------------------
@st.cache_resource
def load_models():
    with open(MODEL_DIR / "do_model.pkl", "rb") as f:
        do_model = pickle.load(f)
    with open(MODEL_DIR / "chl_model.pkl", "rb") as f:
        chl_model = pickle.load(f)
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return do_model, chl_model, metrics


@st.cache_data
def load_sample():
    return pd.read_csv(MODEL_DIR / "sample_data.csv")


try:
    do_model, chl_model, metrics = load_models()
    sample = load_sample()
except FileNotFoundError:
    st.error("Models not found. Run `python train_model.py` first.")
    st.stop()

FEATURES = ["T_degC", "Depthm", "month", "lat", "lon", "turbidity_proxy"]


def predict(temp, depth, month, lat, lon, turbidity):
    X = np.array([[temp, depth, month, lat, lon, turbidity]])
    return float(do_model.predict(X)[0]), float(chl_model.predict(X)[0])


# -------------------------------------------------------------------
# Sidebar — sensor inputs (simulates live probe)
# -------------------------------------------------------------------
st.sidebar.header("📡 Probe Sensor Readings")
st.sidebar.caption("Simulated live feed from the deployed AquaSense probe")

temp = st.sidebar.slider("Temperature (°C)", 5.0, 25.0, 16.5, 0.1,
                         help="Direct reading from DS18B20 temp sensor")
depth = st.sidebar.slider("Depth (m)", 0.0, 300.0, 30.0, 1.0,
                          help="From ultrasonic rangefinder (HC-SR04)")
turbidity = st.sidebar.slider("Turbidity proxy (LDR units)", 0.0, 150.0, 60.0, 1.0,
                              help="LDR + LED makeshift turbidity sensor")
month = st.sidebar.slider("Month", 1, 12, 6)

st.sidebar.subheader("📍 GPS")
lat = st.sidebar.number_input("Latitude", 32.0, 36.0, 33.6, 0.1)
lon = st.sidebar.number_input("Longitude", -123.0, -117.0, -119.0, 0.1)

st.sidebar.subheader("🔊 Acoustic channel")
acoustic_var = st.sidebar.slider(
    "Ultrasonic variance (cm²)", 0.0, 50.0, 8.0, 0.5,
    help="Std. dev. of distance readings over 10s window. "
         "Higher = more disturbance in water column (proxy for activity)."
)

# -------------------------------------------------------------------
# Predictions
# -------------------------------------------------------------------
do_pred, chl_pred = predict(temp, depth, month, lat, lon, turbidity)

# -------------------------------------------------------------------
# Top row — sensor readings + virtual sensor outputs
# -------------------------------------------------------------------
st.subheader("Current Conditions")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("🌡️ Temperature", f"{temp:.1f} °C", help="Direct measurement")
with c2:
    st.metric("📏 Depth", f"{depth:.0f} m", help="Direct measurement")
with c3:
    st.metric("💧 DO (predicted)", f"{do_pred:.2f} mL/L",
              help=f"Virtual sensor — CalCOFI model, MAE={metrics['do']['mae']:.2f}")
with c4:
    st.metric("🌱 Chlorophyll-a (predicted)", f"{chl_pred:.2f} mg/m³",
              help=f"Virtual sensor — CalCOFI model, MAE={metrics['chl']['mae']:.2f}")
with c5:
    disturbance = "Low" if acoustic_var < 5 else "Medium" if acoustic_var < 20 else "High"
    st.metric("🔊 Acoustic activity", disturbance,
              help="Relative disturbance index from ultrasonic variance. "
                   "NOT calibrated biomass — see methodology.")

# -------------------------------------------------------------------
# Habitat suitability
# -------------------------------------------------------------------
st.subheader("🎣 Habitat Suitability by Species")
st.caption("HSI 0-1. Based on published temperature / DO / depth preferences. "
           "Green zones = favorable fishing conditions for sustainable harvest.")

hsi = suitability_all(temp, depth, do_pred, chl_pred)
hsi_df = pd.DataFrame({
    "Species": list(hsi.keys()),
    "HSI": list(hsi.values()),
})
hsi_df = hsi_df.sort_values("HSI", ascending=True)

fig = px.bar(hsi_df, x="HSI", y="Species", orientation="h",
             color="HSI", color_continuous_scale="RdYlGn",
             range_x=[0, 1], range_color=[0, 1])
fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=10),
                  coloraxis_showscale=False)
st.plotly_chart(fig, width='stretch')

best = hsi_df.iloc[-1]
if best["HSI"] > 0.6:
    st.success(f"**Recommended target: {best['Species']}** "
               f"(HSI {best['HSI']:.2f}) — conditions favorable.")
elif best["HSI"] > 0.3:
    st.info(f"Marginal conditions. Best option: **{best['Species']}** "
            f"(HSI {best['HSI']:.2f}).")
else:
    st.warning("No favorable habitat at this depth/location. Try a different "
               "depth or relocate.")

# -------------------------------------------------------------------
# Depth profile — what the probe would see on the way down
# -------------------------------------------------------------------
st.subheader("📊 Predicted Vertical Profile")
st.caption("Virtual cast: what the probe predicts at the current location "
           "across all depths. This is the CalCOFI-style 'bottle profile' our "
           "device produces from a single descent.")

depths = np.arange(0, 301, 5)
# Assume temperature follows a simple thermocline around the measured point
surface_t = temp + 0.03 * depth  # back out surface temp
profile_temps = surface_t - 0.03 * depths + np.random.normal(0, 0.2, len(depths))
profile_temps = np.clip(profile_temps, 4, 25)

profile_turbid = turbidity * np.exp(-depths / 150) + np.random.normal(0, 2, len(depths))

profile_do = []
profile_chl = []
for d, t, tb in zip(depths, profile_temps, profile_turbid):
    do_p, chl_p = predict(t, d, month, lat, lon, tb)
    profile_do.append(do_p)
    profile_chl.append(chl_p)

prof_df = pd.DataFrame({
    "Depth (m)": depths,
    "Temperature (°C)": profile_temps,
    "DO (mL/L)": profile_do,
    "Chlorophyll-a (mg/m³)": profile_chl,
})

col_a, col_b, col_c = st.columns(3)
for col, var, color in [
    (col_a, "Temperature (°C)", "#e74c3c"),
    (col_b, "DO (mL/L)", "#3498db"),
    (col_c, "Chlorophyll-a (mg/m³)", "#27ae60"),
]:
    with col:
        f = go.Figure()
        f.add_trace(go.Scatter(x=prof_df[var], y=prof_df["Depth (m)"],
                               mode="lines", line=dict(color=color, width=3)))
        f.update_yaxes(autorange="reversed", title="Depth (m)")
        f.update_xaxes(title=var)
        f.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                        showlegend=False)
        st.plotly_chart(f, width='stretch')

# -------------------------------------------------------------------
# Model diagnostics — for the biotech pitch
# -------------------------------------------------------------------
with st.expander("🔬 Model validation & methodology (for judges)"):
    st.markdown(f"""
    **Training data:** {metrics['data_source']} — {metrics['n_samples']:,} samples

    **Model type:** Random Forest Regressor (100 trees, max depth 15)

    **Features used (what our probe measures):**
    - Temperature (°C) — direct, DS18B20 waterproof sensor
    - Depth (m) — ultrasonic rangefinder
    - Turbidity proxy — LDR + LED optical
    - Month, lat, lon — GPS + RTC

    **Targets (what we predict):**
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"**Dissolved Oxygen (O₂, mL/L)**\n\n"
            f"- R² = `{metrics['do']['r2']:.3f}`\n"
            f"- MAE = `{metrics['do']['mae']:.3f}` mL/L\n"
            f"- Train/test: {metrics['do']['n_train']:,} / {metrics['do']['n_test']:,}"
        )
        imp = metrics['do']['feature_importance']
        imp_df = pd.DataFrame({"feature": list(imp.keys()),
                               "importance": list(imp.values())}
                             ).sort_values("importance", ascending=True)
        fig = px.bar(imp_df, x="importance", y="feature", orientation="h")
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown(
            f"**Chlorophyll-a (mg/m³)**\n\n"
            f"- R² = `{metrics['chl']['r2']:.3f}`\n"
            f"- MAE = `{metrics['chl']['mae']:.3f}` mg/m³\n"
            f"- Train/test: {metrics['chl']['n_train']:,} / {metrics['chl']['n_test']:,}"
        )
        imp = metrics['chl']['feature_importance']
        imp_df = pd.DataFrame({"feature": list(imp.keys()),
                               "importance": list(imp.values())}
                             ).sort_values("importance", ascending=True)
        fig = px.bar(imp_df, x="importance", y="feature", orientation="h")
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width='stretch')

    st.markdown("""
    **Why this works:** Dissolved oxygen in seawater is governed by temperature
    (solubility), depth (mixing + biological consumption), and productivity
    (which our turbidity proxy captures). Chlorophyll follows seasonal light
    and nutrient cycles well-captured by month + location + depth.

    **Honest caveats:**
    - This replaces a ~$2,000 DO sensor and ~$5,000 fluorometer with statistical
      inference. It will never beat a direct measurement — but it's good enough
      for trend detection and zone-level decisions.
    - Models are valid for the Southern California Bight (CalCOFI coverage).
      Deployment elsewhere would require local retraining.
    - Acoustic activity channel is a **relative** index, not calibrated biomass.
      Production version would use a proper echosounder.
    """)

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.caption("Prototype built for hackathon demo. Data: CalCOFI (Scripps). "
           "Hardware BOM: ~$50 (Arduino + DS18B20 + HC-SR04 + LDR + LED + claw servo).")
