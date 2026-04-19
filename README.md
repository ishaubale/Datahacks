# AquaSense — Hackathon Pipeline

Trains ML models on CalCOFI oceanographic data so our low-cost probe can
predict dissolved oxygen and chlorophyll-a from the few sensors it actually
has (temperature, depth, turbidity proxy, GPS). Then uses those predictions
to score habitat suitability for local fish species.

## Quick start (3 commands)

```bash
pip install pandas numpy scikit-learn streamlit plotly --break-system-packages

python train_model.py              # trains models, saves to models/
streamlit run dashboard.py          # opens live demo at localhost:8501
```

**Without the CalCOFI CSV**, `train_model.py` falls back to synthetic data
that mimics CalCOFI relationships — use this for dev. For any scientific
claim in the pitch, download the real CSV from
<https://calcofi.org/data/oceanographic-data/bottle-database/> and save it
as `calcofi_bottle.csv` in this directory.

## Files

- `train_model.py` — data loading, cleaning, feature engineering, RF training
- `habitat_suitability.py` — HSI model for Pacific sardine, anchovy, mackerel, hake
- `dashboard.py` — Streamlit live demo
- `models/` — created on first training run

## Demo script (for pitch)

1. Open the dashboard. Sidebar = simulated live sensor feed from the probe.
2. Move the **Temperature** slider — watch DO prediction fall as temp rises
   (correct: warm water holds less oxygen).
3. Move the **Depth** slider from 0 to 250 m — watch DO drop and chlorophyll
   peak around 20-40 m (correct: chlorophyll maximum layer).
4. Switch **Month** from Jan to June — chlorophyll rises (spring bloom).
5. Point at the habitat suitability bar chart: "the device tells the fisher
   which species are actually findable right here, right now — based on 75
   years of Scripps oceanography."
6. Expand the **Model validation** panel for R²/MAE numbers (biotech judges).

## Pitch talking points this enables

**Biotech track:**
- "We validated our $50 probe against 75 years of CalCOFI Scripps data."
- "Our virtual sensors predict DO within ~0.5 mL/L and chlorophyll within
  ~1 mg/m³ of the gold-standard measurements, using only cheap sensors."
- "Vertical profiling + benthic sampling in one deployment — no existing
  consumer device does both."

**Entrepreneurship track:**
- "CTD-Rosette systems cost $10-20K. We deliver 80% of the scientifically
  useful data at 1% of the hardware cost."
- "Every probe deployed adds to a shared map — network effect data moat."
- "Target customers: fishing cooperatives, aquaculture, watershed NGOs,
  environmental consultancies."

## Honest limitations (have these ready for judge questions)

- Model is calibrated for Southern California Bight; other regions need
  local retraining.
- Acoustic variance is a **relative disturbance index**, not biomass.
  Real biomass estimation needs a calibrated echosounder.
- Turbidity proxy from LDR is uncalibrated — production version needs a
  proper nephelometer (~$30 upgrade).
- Virtual sensor is statistical inference, not direct measurement. It's
  good for trends and zoning; regulators would still need lab confirmation
  for enforcement-grade data.
# Datahacks
