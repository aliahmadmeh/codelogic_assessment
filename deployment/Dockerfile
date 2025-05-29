# deployment/Dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY ./api ./api
COPY ./models ./models
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
