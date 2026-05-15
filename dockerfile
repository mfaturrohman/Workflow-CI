FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install mlflow pandas numpy scikit-learn

WORKDIR /app/MLProject

EXPOSE 8080

CMD mlflow models serve -m mlruns/0/7215d8933b634a63a90e0d82af7333eb/artifacts/model -h 0.0.0.0 -p 8080 --no-conda