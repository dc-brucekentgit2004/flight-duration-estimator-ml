# src/synthetic_generator.py
# Use this if you don't have a dataset. Generates a synthetic but realistic flights CSV.
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from src.utils import parse_time_to_minutes

AIRLINES = ["IndiGo","AirIndia","SpiceJet","Vistara","GoAir","AirAsia"]
AIRPORTS = ["DEL","BOM","BLR","HYD","MAA","CCU","GOI","COK","TRV","PNQ"]
DIST_MAP = {
    ("DEL","BOM"):1150, ("DEL","BLR"):1740, ("BOM","BLR"):980, ("BLR","HYD"):500
}
def gen_row(date):
    origin = random.choice(AIRPORTS)
    dest = random.choice([a for a in AIRPORTS if a!=origin])
    carrier = random.choice(AIRLINES)
    dep_hour = random.randint(0,23)
    dep_min = random.choice([0,15,30,45])
    dep_time = dep_hour*100 + dep_min
    # duration
    base_dist = DIST_MAP.get((origin,dest), DIST_MAP.get((dest,origin), random.randint(200,2000)))
    duration = max(30, int(base_dist/8))  # rough conversion
    arr_time_mins = parse_time_to_minutes(dep_time) + duration
    arr_hour = (arr_time_mins//60)%24
    arr_min = arr_time_mins%60
    arr_time = arr_hour*100 + arr_min
    dep_delay = int(np.random.normal(10,20))
    dep_delay = max(-15, dep_delay)
    arr_delay = dep_delay + int(np.random.normal(0,10))
    distance = base_dist
    return {
        "FL_DATE": date.strftime("%Y-%m-%d"),
        "OP_CARRIER": carrier,
        "ORIGIN": origin,
        "DEST": dest,
        "CRS_DEP_TIME": dep_time,
        "CRS_ARR_TIME": arr_time,
        "DEP_DELAY": dep_delay,
        "ARR_DELAY": arr_delay,
        "DISTANCE": distance,
        "CANCELLED": 0,
    }

def generate(n_days=180, rows_per_day=80, out_path="data/flights_synthetic.csv"):
    rows=[]
    start = datetime.today() - timedelta(days=n_days)
    for i in range(n_days):
        date = start + timedelta(days=i)
        for _ in range(rows_per_day):
            rows.append(gen_row(date))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print("Synthetic dataset saved to", out_path)

if __name__=="__main__":
    generate()
