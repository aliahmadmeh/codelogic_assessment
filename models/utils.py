import joblib
import pandas as pd

def predict_rul(input_df: pd.DataFrame) -> float:
    """
    Loads trained model and preprocessing pipeline, returns RUL prediction
    :param input_df: pd.DataFrame with required features
    :return: predicted RUL as float
    """
 
    model = joblib.load("models/xgb_rul_model.pkl")

   
    expected_features = model.named_steps['scaler'].get_feature_names_out()
    if not all(feat in input_df.columns for feat in expected_features):
        raise ValueError("Input features are incomplete or improperly formatted.")

    
    prediction = model.predict(input_df)[0]
    return round(prediction, 2)
