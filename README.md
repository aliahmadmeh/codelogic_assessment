# codelogic_assessment

---

## 🛠️ Setup Instructions

### 1. Clone the Repository


git clone https://github.com/your-username/predictive-maintenance-mlops.git
cd predictive-maintenance-mlops

### 2. Install Dependencies

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 3. Prepare Data
Place the following files in the data/ directory:

train_FD001.txt
test_FD001.txt
RUL_FD001.txt

These are from the NASA CMAPSS dataset (FD001 subset).

### 4. Train the Model

python models/train_model.py
This saves the trained model and preprocessing pipeline in models/.

🚀 Run the API
uvicorn api.main:app --reload
Visit http://localhost:8000/docs to access Swagger UI.

📈 API Endpoints
/health
Health check.

/predict
Make a single prediction.

Request Body:

{
  "operational_setting_1": 0.5,
  "operational_setting_2": 0.0,
  "operational_setting_3": 100.0,
  "sensor_measurement_1": 518.67,
  ...
}
/predict_batch
Submit a list of sensor readings for batch predictions.

📊 Model Performance

Model: GradientBoostingRegressor
Metrics:

RMSE: TBD

MAE: TBD

You can evaluate these in the training script and plot metrics as needed.

📌 Notes
Built with time series feature engineering and robust model packaging.
Fully compatible with Docker and CI/CD.
Add Prometheus/Grafana for monitoring and PostgreSQL for logging predictions (in Docker Compose).

📬 Contact

aliahmadmehmod@gmail.com
