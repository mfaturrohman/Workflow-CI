import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# SET TRACKING URI (ABSOLUT / RELATIF AMAN)
mlflow.set_tracking_uri("file:./mlruns")

# SET EXPERIMENT (PASTI BUAT ID BARU, BUKAN 0)
mlflow.set_experiment("Telco_Churn_Prediction")

# AKTIFKAN AUTOLOG SEBELUM RUN
mlflow.sklearn.autolog()

# Load data
df = pd.read_csv("telco_customer_churn_preprocessing.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# START RUN
with mlflow.start_run():
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy_manual", accuracy)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
