import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("telco_churn_experiment")

mlflow.autolog()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mlflow.log_metric("accuracy_manual", accuracy_score(y_test, y_pred))
mlflow.log_metric("f1_manual", f1_score(y_test, y_pred))
mlflow.log_metric("precision_manual", precision_score(y_test, y_pred))
mlflow.log_metric("recall_manual", recall_score(y_test, y_pred))
