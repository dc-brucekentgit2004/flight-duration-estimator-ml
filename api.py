# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List
from src.model_predict import predict_price

app = FastAPI(title="Flight Prediction API")

class FlightRequest(BaseModel):
    FL_DATE: str
    OP_CARRIER: str
    ORIGIN: str
    DEST: str
    CRS_DEP_TIME: str
    CRS_ARR_TIME: str
    DISTANCE: float
    DEP_DELAY: float = 0.0

class BatchedRequest(BaseModel):
    flights: List[FlightRequest]

@app.post("/predict")
def predict(request: FlightRequest):
    df = pd.DataFrame([request.dict()])
    price = predict_price(df)
    return {"prediction": price}

@app.post("/predict_batch")
def predict_batch(req: BatchedRequest):
    df = pd.DataFrame([f.dict() for f in req.flights])
    from src.model_predict import predict_batch
    preds = predict_batch(df)
    return {"predictions": preds.tolist()}
