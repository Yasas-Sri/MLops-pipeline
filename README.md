# MLOps Pipeline 

A complete MLOps pipeline for training, evaluating, and deploying a breast cancer classification model using MLflow and FastAPI.

## Project Overview

This project implements an end-to-end machine learning pipeline that includes:
- Data preprocessing using the  Breast Cancer dataset
- Model training with Logistic Regression
- MLflow experiment tracking and model registry
- Automated model evaluation with quality gates
- Model promotion workflow



## Project Structure

```
.
├── api.py                  # FastAPI application for model serving
├── train.py                # Model training script
├── evaluate.py             # Model evaluation with quality gates
├── promoteModel.py         # Model promotion logic
├── preprocessing.py        # Data loading and preprocessing
├── mlflow_config.py        # MLflow configuration
├── requirements.txt        # Python dependencies
├── mlflow.db              # SQLite database for MLflow tracking
└── mlruns/                # MLflow artifacts and models
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MLpipeline-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train a Model

Train a new model and log it to MLflow:

```bash
python train.py
```

This will:
- Load and split the breast cancer dataset
- Train a Logistic Regression model
- Log parameters, metrics, and the model to MLflow
- Register the model as "BreastCancerClassifier"
- Output a Run ID for the next steps

### 2. Evaluate the Model

Evaluate a trained model against quality gates:

```bash
python evaluate.py --run_id <RUN_ID>
```

The model must achieve >90% accuracy to pass the quality gate.

### 3. Promote to Production

Promote a model to production if it's better than the current production model:

```bash
python promoteModel.py <RUN_ID>
```

This compares the new model's accuracy with the current production model and promotes it if better.

### 4. Start the MLflow UI

View experiments and models:

```bash
mlflow ui --port 5001
```

Access at: http://localhost:5001

### 5. Start the Prediction API

Launch the FastAPI service:

```bash
uvicorn api:app --reload
```

Access at: http://localhost:8000

API documentation: http://localhost:8000/docs

## API Endpoints

### Health Check
```bash
GET /
```

Response:
```json
{
  "status": "Model API running"
}
```

### Make Predictions
```bash
POST /predict
```

Request body example:
```json
{
  "mean radius": 14.0,
  "mean texture": 20.0,
  "mean perimeter": 90.0,
  "mean area": 600.0,
  "mean smoothness": 0.1,
  ...
}
```

Response:
```json
{
  "prediction": 1
}
```

## MLflow Configuration

- **Tracking URI**: SQLite database (`sqlite:///mlflow.db`)
- **Experiment Name**: MLops-Demo
- **Model Name**: BreastCancerClassifier
- **Production Alias**: production

## Model Metrics

The pipeline tracks the following metrics:
- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1 Score**: Harmonic mean of precision and recall

## Workflow

1. **Train**: Run `train.py` to create a new model version
2. **Evaluate**: Run `evaluate.py` to check if the model meets quality standards
3. **Promote**: Run `promoteModel.py` to update production if the new model is better
4. **Serve**: The API automatically serves the model tagged with the "production" alias


