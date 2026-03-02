import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from mlflow_config import setup_mlflow

MODEL_NAME = "BreastCancerClassifier"
ALIAS_NAME = "production"


setup_mlflow()

app = FastAPI()


model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}@{ALIAS_NAME}"
)

@app.get("/")
def health():
    return {"status": "Model API running"}

@app.post("/predict")
def predict(features: dict):
    """
    Send JSON with feature names matching training data.
    Example:
    {
        "mean radius": 14.0,
        ...
    }
    """

    df = pd.DataFrame([features])
    prediction = model.predict(df)

    return {"prediction": int(prediction[0])}