from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import os

app = FastAPI()


with open("models/xgb_rul_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/preprocessing_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

class SensorData(BaseModel):
    operational_setting_1: float
    operational_setting_2: float
    operational_setting_3: float
    sensor_measurement_1: float
    sensor_measurement_2: float
    sensor_measurement_3: float
    sensor_measurement_4: float
    sensor_measurement_5: float
    sensor_measurement_6: float
    sensor_measurement_7: float
    sensor_measurement_8: float
    sensor_measurement_9: float
    sensor_measurement_10: float
    sensor_measurement_11: float
    sensor_measurement_12: float
    sensor_measurement_13: float
    sensor_measurement_14: float
    sensor_measurement_15: float
    sensor_measurement_16: float
    sensor_measurement_17: float
    sensor_measurement_18: float
    sensor_measurement_19: float
    sensor_measurement_20: float
    sensor_measurement_21: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_rul(data: SensorData):
    try:
        df = pd.DataFrame([data.dict()])
        X_scaled = pipeline.transform(df)
        prediction = model.predict(X_scaled)
        return {"predicted_RUL": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
def predict_batch(data: list[SensorData]):
    try:
        df = pd.DataFrame([item.dict() for item in data])
        X_scaled = pipeline.transform(df)
        prediction = model.predict(X_scaled)
        return {"predicted_RULs": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
