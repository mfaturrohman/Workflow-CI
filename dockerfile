FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install mlflow pandas numpy scikit-learn

WORKDIR /app/MLProject

CMD ["python", "modelling.py"]