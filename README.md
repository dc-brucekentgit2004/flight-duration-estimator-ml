# Advanced Flight Prediction (Production-ready)

## Features
- Advanced feature engineering (datetime features, haversine distance, route complexity, rolling stats)
- Target encoding + OneHot + scaling pipeline
- Stacked ensemble (LightGBM, XGBoost, RandomForest) with XGBoost as meta-learner
- Streamlit demo UI and FastAPI endpoint
- Synthetic data generator included

## Quickstart
1. Create virtualenv and install: `pip install -r requirements.txt`
2. Put your dataset at `data/flights.csv` (or generate synthetic data: `python -m src.synthetic_generator`)
3. Train model: run a small script to call training (example in `src/model_training.py`)
4. Run Streamlit app: `streamlit run app/streamlit_app.py`
5. Run API: `uvicorn app.api:app --reload --port 8000`

## Files
- `src/` code
- `app/` Streamlit & FastAPI
- `data/` dataset
