# src/utils.py
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
import holidays
from datetime import datetime

IND_HOLIDAYS = holidays.CountryHoliday("IN")  # change if needed

def haversine(lat1, lon1, lat2, lon2):
    # returns distance in kilometers
    # all inputs in decimal degrees
    if np.any(pd.isnull([lat1, lon1, lat2, lon2])):
        return np.nan
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def is_holiday(date_obj):
    # date_obj: datetime.date or datetime
    try:
        return int(date_obj.date() in IND_HOLIDAYS)
    except Exception:
        return 0

def train_test_split_df(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def parse_time_to_minutes(t):
    # handles HHMM or HH:MM or numeric
    if pd.isna(t):
        return np.nan
    if isinstance(t, (int, float, np.integer)):
        t = int(t)
        return (t // 100) * 60 + (t % 100)
    t = str(t)
    if ":" in t:
        parts = t.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    t = t.zfill(4)
    return (int(t[:2]) * 60) + int(t[2:])

def safe_divide(a, b):
    try:
        return a / b
    except Exception:
        return 0

