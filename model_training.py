# src/model_training.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
from src.preprocess import build_preprocessor, load_data
from src.config import STACKED_MODEL_PATH, PREPROCESSOR_PATH, FEATURE_LIST_PATH, RANDOM_STATE
from src.persistence import save_json
import warnings
warnings.filterwarnings("ignore")

def train_and_save(df_path, target_col="ARR_DELAY", test_size=0.2):
    print("Loading data...")
    df = load_data(df_path)
    print("Building preprocessor...")
    preprocessor, feature_names = build_preprocessor(df, save_path=PREPROCESSOR_PATH)
    X = df.copy()
    y = X[target_col].values
    # drop rows with NaNs in target
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    # keep same features as preprocessor expects (transform will select columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
    print("Fitting base models...")
    # base learners
    lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=RANDOM_STATE)
    xgbr = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, random_state=RANDOM_STATE)
    cat = CatBoostRegressor(iterations=800, learning_rate=0.05, verbose=0, random_state=RANDOM_STATE)
    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
    # stacking regressor
    estimators = [
        ('lgbm', lgbm),
        ('xgbr', xgbr),
        ('rf', rf)
    ]
    final_est = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, random_state=RANDOM_STATE)
    stack = StackingRegressor(estimators=estimators, final_estimator=final_est, n_jobs=-1, passthrough=False)
    # build pipeline manually: preprocess -> fit stack on transformed arrays
    print("Transforming features for training (this may take time)...")
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    print("Training stacked ensemble (this will take several minutes)...")
    stack.fit(X_train_t, y_train)
    preds = stack.predict(X_test_t)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Validation RMSE: {rmse:.3f}")
    # persist model and metadata
    dump(stack, STACKED_MODEL_PATH)
    save_json({"rmse": float(rmse)}, STACKED_MODEL_PATH + ".meta.json")
    print("Model and artifacts saved to model/ directory")
    return stack, preprocessor
