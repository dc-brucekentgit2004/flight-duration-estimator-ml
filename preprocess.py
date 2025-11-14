import pandas as pd
from category_encoders import TargetEncoder
from src.feature_engineering import full_feature_engineering


def load_data(path):
    """
    Loads raw flight CSV dataset.
    """
    return pd.read_csv(path)


def preprocess_data(df):
    """
    Cleans dataset and applies encoding (basic preprocessing).
    """
    df = df.copy()

    # Remove rows with missing values in critical fields
    critical_cols = ["airline", "source", "destination", "duration"]
    df = df.dropna(subset=critical_cols)

    # Encode categorical fields
    encoder = TargetEncoder()
    df["airline_enc"] = encoder.fit_transform(df["airline"], df["duration"])

    return df


def load_and_preprocess(path):
    """
    Full pipeline: load → preprocess → advanced feature engineering.
    """
    df = load_data(path)
    df = preprocess_data(df)
    df = full_feature_engineering(df)
    return df
