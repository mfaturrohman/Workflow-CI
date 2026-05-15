FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install mlflow pandas numpy scikit-learn

WORKDIR /app/MLProject

EXPOSE 8080

CMD mlflow models serve -m mlruns/1/models/m-4ef04d1a735347deab4db861859a223a/artifacts -h 0.0.0.0 -p 8080 --no-conda