import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os


df = pd.read_csv("data/FD001.txt", sep="\s+", header=None)
df.columns = ['unit_number', 'cycle'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]

drop_cols = ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
df.drop(columns=drop_cols, inplace=True)


rul_df = df.groupby("unit_number")["cycle"].max().reset_index()
rul_df.columns = ["unit_number", "max_cycle"]
df = df.merge(rul_df, on="unit_number", how="left")
df["RUL"] = df["max_cycle"] - df["cycle"]
df.drop(columns=["max_cycle"], inplace=True)


rolling_window = 5
rolling_cols = ['sensor_2', 'sensor_3', 'sensor_7', 'sensor_8', 'sensor_11', 'sensor_15']

for col in rolling_cols:
    df[f"{col}_mean"] = df.groupby("unit_number")[col].rolling(window=rolling_window).mean().reset_index(0, drop=True)
    df[f"{col}_std"] = df.groupby("unit_number")[col].rolling(window=rolling_window).std().reset_index(0, drop=True)

df.fillna(method="bfill", inplace=True)


feature_cols = [col for col in df.columns if col not in ['unit_number', 'cycle', 'RUL']]
X = df[feature_cols]
y = df["RUL"]


unique_units = df["unit_number"].unique()
train_units, test_units = train_test_split(unique_units, test_size=0.2, random_state=42)

X_train = X[df["unit_number"].isin(train_units)]
y_train = y[df["unit_number"].isin(train_units)]
X_test = X[df["unit_number"].isin(test_units)]
y_test = y[df["unit_number"].isin(test_units)]


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse:.2f} cycles")
print(f"MAE: {mae:.2f} cycles")

# Save model and pipeline
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/xgb_rul_model.pkl")
print("Model saved to models/xgb_rul_model.pkl")
