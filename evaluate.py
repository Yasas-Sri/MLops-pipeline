import sys
import argparse
import mlflow
from mlflow_config import setup_mlflow

def evaluate(run_id):
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    try:
        run = client.get_run(run_id)
    except Exception as e:
        print(f"Error: Could not find Run ID {run_id}. {e}")
        sys.exit(1)

    accuracy = run.data.metrics.get("accuracy")
    
    if accuracy is None:
        print(f"Accuracy metric missing for run {run_id}")
        sys.exit(1)

    print(f"Evaluating Run: {run_id}")
    print(f"Model accuracy: {accuracy:.4f}")

    threshold = 0.90
    if accuracy < threshold:
        print(f"Model failed quality gate (Target: {threshold})")
        sys.exit(1)

    print("Model passed quality gate")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, help="MLflow Run ID to evaluate")
    args = parser.parse_args()
    
    evaluate(args.run_id)