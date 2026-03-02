import argparse
import mlflow
from mlflow_config import setup_mlflow

MODEL_NAME = "BreastCancerClassifier"
ALIAS_NAME = "production"


def promote_model(run_id):
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    
    new_run = client.get_run(run_id)
    new_accuracy = new_run.data.metrics.get("accuracy")

    print(f"New model accuracy: {new_accuracy:.4f}")

    
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

   
    new_version = None
    for v in versions:
        if v.run_id == run_id:
            new_version = v.version
            break

    if new_version is None:
        print("Model version not found for this run.")
        return

  
    try:
        prod_version = client.get_model_version_by_alias(
            MODEL_NAME, ALIAS_NAME
        )
        prod_run = client.get_run(prod_version.run_id)
        prod_accuracy = prod_run.data.metrics.get("accuracy")

        print(f"Current production accuracy: {prod_accuracy:.4f}")

        if new_accuracy <= prod_accuracy:
            print("New model is NOT better than production")
            return

    except Exception:
        print("No production model found. Promoting first version.")

    
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS_NAME,
        version=new_version
    )

    print(f"Version {new_version} promoted to alias '{ALIAS_NAME}' ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="Run ID to promote")
    args = parser.parse_args()

    promote_model(args.run_id)