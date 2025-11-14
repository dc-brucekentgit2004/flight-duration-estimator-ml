# src/persistence.py
import json
from joblib import dump, load

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_joblib(obj, path):
    dump(obj, path)

def load_joblib(path):
    return load(path)
