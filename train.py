import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import load_split_data
from mlflow_config import setup_mlflow

def train():
    setup_mlflow()
    X_train, X_test, y_train, y_test = load_split_data()


    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        model = LogisticRegression(max_iter=5000)
        mlflow.log_param("model_type", "LogisticRegression")
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds)
        }
        mlflow.log_metrics(metrics)


        

        model_name = "BreastCancerClassifier"

        
        mlflow.sklearn.log_model(model, name="model",    registered_model_name=model_name)

        print(f"Model registered as: {model_name}")
        
        
        print(f"Training complete. Run ID: {run_id}")
        return run_id

if __name__ == "__main__":
    train()