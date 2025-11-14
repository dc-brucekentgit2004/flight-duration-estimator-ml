# src/config.py
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# model filenames
STACKED_MODEL_PATH = os.path.join(MODEL_DIR, "stacked_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.joblib")

# target and thresholds
DELAY_THRESHOLD_MIN = 15  # minutes for classification (if used)
RANDOM_STATE = 42
