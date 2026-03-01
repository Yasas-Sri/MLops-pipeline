import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocessing import load_split_data

mlflow.set_experiment("MLops-Demo")

X_train, X_test, y_train, y_test = load_split_data()

model = LogisticRegression(max_iter=5000)

with mlflow.start_run():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")

    print("Training complete. Accuracy:", acc)