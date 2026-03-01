import sys
import mlflow

client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name("MLops-Demo")
runs = client.search_runs(experiment.experiment_id)

latest_run = runs[0]
accuracy = latest_run.data.metrics["accuracy"]

print("Latest model accuracy:", accuracy)

if accuracy < 0.90:
    print("Model failed quality gate")
    sys.exit(1)

print("Model passed quality gate ")