# src/model_predict.py
import pandas as pd
import numpy as np
from joblib import load
from src.config import STACKED_MODEL_PATH, PREPROCESSOR_PATH, RANDOM_STATE
from src.preprocess import load_data
from src.persistence import load_joblib
from typing import Dict

def load_model_and_preprocessor():
    model = load(STACKED_MODEL_PATH)
    preprocessor = load(PREPROCESSOR_PATH)
    return model, preprocessor

def prepare_single_input(df_row, preprocessor):
    # df_row is a single-row DataFrame with original raw columns (FL_DATE, ORIGIN, DEST, CRS_DEP_TIME, CRS_ARR_TIME, etc.)
    # We must run the same feature engineering
    from src.feature_engineering import full_feature_engineering
    df_fe = full_feature_engineering(df_row.copy())
    X_t = preprocessor.transform(df_fe)
    return X_t

def predict_price(df_row):
    model, preprocessor = load_model_and_preprocessor()
    X_t = prepare_single_input(df_row, preprocessor)
    pred = model.predict(X_t)
    # For uncertainty, we can return +/- based on validation RMSE meta if available (approx)
    return float(pred[0])

# helper for batch
def predict_batch(df):
    model, preprocessor = load_model_and_preprocessor()
    X_t = preprocessor.transform(df)
    preds = model.predict(X_t)
    return preds
