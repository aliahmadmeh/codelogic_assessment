# Joblogic_assessment

---

## 🛠️ Setup Instructions

### 1. 📁 Clone the Repository


git clone https://github.com/aliahmadmeh/codelogic_assessment

cd codelogic_assessment

### 2. 🐍 Install Dependencies

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

### 3. 📁 Prepare Data
Place the following files in the data/ directory:

  train_FD001.txt

  test_FD001.txt

  RUL_FD001.txt

These are from the NASA CMAPSS dataset (FD001 subset).

### 🧠 Step 4: Train the Model

python models/train_model.py

This saves the trained model and preprocessing pipeline in models/.

🚀 Run the API

uvicorn api.main:app --reload

Visit http://localhost:8000/docs to access Swagger UI.



📈 API Endpoints

| Endpoint         | Method | Description                            |
| ---------------- | ------ | -------------------------------------- |
| `/health`        | GET    | Health check                           |
| `/predict`       | POST   | Single engine RUL prediction           |
| `/predict_batch` | POST   | Batch predictions for multiple engines |


📊 Model Overview

✅ Model Type: Gradient Boosting Regressor

🛠️ Feature Engineering: Rolling mean, time-window stats, normalization

🧪 Validation Strategy: Time-aware split

📉 Evaluation Metrics:

* RMSE: TBD

* MAE: TBD

📌 Notes

Built with time series feature engineering and robust model packaging.

Fully compatible with Docker and CI/CD.

Add Prometheus/Grafana for monitoring and PostgreSQL for logging predictions (in Docker Compose).

🐳 Docker & CI/CD (optional extensions)

This project is designed to support:

✅ Docker + Docker Compose

✅ PostgreSQL or SQLite for logging predictions

✅ Prometheus / Grafana for monitoring

✅ GitHub Actions for CI/CD

📬 Contact

aliahmadmehmod@gmail.com
