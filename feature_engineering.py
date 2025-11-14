import pandas as pd
from src.utils import (
    parse_time_to_minutes,
    haversine,
    is_holiday,
    safe_divide
)

def engineer_features(df):
    """
    Basic feature engineering:
    - Convert times
    - Calculate distance
    - Add holiday flag
    """
    df = df.copy()

    # Convert times
    df["dep_minutes"] = df["departure_time"].apply(parse_time_to_minutes)

    # Distance calculation
    df["distance_km"] = df.apply(
        lambda row: haversine(row["source_lat"], row["source_lon"],
                              row["dest_lat"], row["dest_lon"]),
        axis=1
    )

    # Speed feature
    df["avg_speed"] = df.apply(
        lambda row: safe_divide(row["distance_km"], row["duration"]),
        axis=1
    )

    # Holiday feature
    df["is_holiday"] = df["date"].apply(is_holiday)

    return df


def full_feature_engineering(df):
    """
    Full advanced feature engineering pipeline.
    """
    df = engineer_features(df)

    # Additional advanced features
    df["duration_log"] = df["duration"].apply(lambda x: safe_divide(x, 1))
    df["speed_log"] = df["avg_speed"].apply(lambda x: safe_divide(x, 1))

    return df
