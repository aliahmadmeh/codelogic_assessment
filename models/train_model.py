import pandas as pd
import numpy as np
import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


train_path = "data/train_FD001.txt"
test_path = "data/test_FD001.txt"
rul_path = "data/RUL_FD001.txt"


assert os.path.exists(train_path), f"{train_path} not found!"
assert os.path.exists(test_path), f"{test_path} not found!"
assert os.path.exists(rul_path), f"{rul_path} not found!"


column_names = [
    "unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2",
    "operational_setting_3", "sensor_measurement_1", "sensor_measurement_2", "sensor_measurement_3",
    "sensor_measurement_4", "sensor_measurement_5", "sensor_measurement_6", "sensor_measurement_7",
    "sensor_measurement_8", "sensor_measurement_9", "sensor_measurement_10", "sensor_measurement_11",
    "sensor_measurement_12", "sensor_measurement_13", "sensor_measurement_14", "sensor_measurement_15",
    "sensor_measurement_16", "sensor_measurement_17", "sensor_measurement_18", "sensor_measurement_19",
    "sensor_measurement_20", "sensor_measurement_21"
]


train_df = pd.read_csv(train_path, sep=" ", header=None)
train_df.dropna(axis=1, how="all", inplace=True)
train_df.columns = column_names


rul = train_df.groupby("unit_number")["time_in_cycles"].max().reset_index()
rul.columns = ["unit_number", "max_cycle"]
train_df = train_df.merge(rul, on="unit_number")
train_df["RUL"] = train_df["max_cycle"] - train_df["time_in_cycles"]


features = column_names[2:] 
X = train_df[features]
y = train_df["RUL"]

pipeline = Pipeline([
    ("scaler", StandardScaler())
])

X_scaled = pipeline.fit_transform(X)


model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_scaled, y)


os.makedirs("models", exist_ok=True)
with open("models/xgb_rul_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/preprocessing_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model and preprocessing pipeline saved.")
