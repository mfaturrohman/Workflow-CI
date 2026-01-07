import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("telco_churn_experiment")

# ===== LOAD DATA (WAJIB: DATASET HASIL PREPROCESSING) =====
df = pd.read_csv("telco_customer_churn_preprocessing.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== MLFLOW =====
mlflow.autolog()

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Manual metric tambahan (aman walau autolog aktif)
    mlflow.log_metric("accuracy_manual", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_manual", f1_score(y_test, y_pred))
    mlflow.log_metric("precision_manual", precision_score(y_test, y_pred))
    mlflow.log_metric("recall_manual", recall_score(y_test, y_pred))
