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

