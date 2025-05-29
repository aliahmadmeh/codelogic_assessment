from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Predictive Maintenance API")

model_path = "models/xgb_rul_model.pkl"
model = joblib.load(model_path)

class SensorInput(BaseModel):
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_2: float
    sensor_3: float
    sensor_7: float
    sensor_8: float
    sensor_11: float
    sensor_15: float
    sensor_2_mean: float
    sensor_2_std: float
    sensor_3_mean: float
    sensor_3_std: float
    sensor_7_mean: float
    sensor_7_std: float
    sensor_8_mean: float
    sensor_8_std: float
    sensor_11_mean: float
    sensor_11_std: float
    sensor_15_mean: float
    sensor_15_std: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_rul(input_data: SensorInput):
    try:
        df = pd.DataFrame([input_data.dict()])
        pred = model.predict(df)[0]
        return {"predicted_RUL": round(pred, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_rul_batch(inputs: list[SensorInput]):
    try:
        df = pd.DataFrame([inp.dict() for inp in inputs])
        preds = model.predict(df).tolist()
        return {"predictions": [round(p, 2) for p in preds]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
