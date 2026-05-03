
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import xgboost as xgb

from huggingface_hub import HfApi, create_repo
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# Configuration
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing.")

DATASET_REPO_ID = "chandrachurhghosh/predictive-maintenance-project"
MODEL_REPO_ID = "chandrachurhghosh/predictive-maintenance-model"
REPO_TYPE = "dataset"

api = HfApi(token=HF_TOKEN)

mlflow.set_experiment("MLOps_CICD_PredictiveMaintenance")

# ============================================================
# Load Train and Test Data from Hugging Face Data Space
# ============================================================

BASE_PATH = f"hf://datasets/{DATASET_REPO_ID}/data/processed"

print("Loading train and test data from Hugging Face...")

X_train = pd.read_csv(f"{BASE_PATH}/X_train.csv")
X_test = pd.read_csv(f"{BASE_PATH}/X_test.csv")
y_train = pd.read_csv(f"{BASE_PATH}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{BASE_PATH}/y_test.csv").squeeze()

print("Data loaded successfully.")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# ============================================================
# Define Model and Parameters
# ============================================================

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.05],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "reg_lambda": [0.1, 1]
}

# ============================================================
# Tune Model with Defined Parameters
# ============================================================

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="recall",
    n_jobs=-1,
    verbose=1
)

# ============================================================
# Experiment Tracking with MLflow
# ============================================================

with mlflow.start_run(run_name="xgboost_predictive_maintenance_gridsearch"):

    grid_search.fit(X_train, y_train)

    # ========================================================
    # Log All Tuned Parameters
    # ========================================================

    cv_results = grid_search.cv_results_

    for i, params in enumerate(cv_results["params"]):
        with mlflow.start_run(run_name=f"xgboost_trial_{i}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_recall", cv_results["mean_test_score"][i])
            mlflow.log_metric("std_cv_recall", cv_results["std_test_score"][i])

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_recall", grid_search.best_score_)

    # ========================================================
    # Evaluate Model Performance
    # ========================================================

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    y_proba_test = best_model.predict_proba(X_test)[:, 1]

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_proba_test)

    metrics = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1_score": test_f1,
        "test_roc_auc": test_roc_auc
    }

    mlflow.log_metrics(metrics)

    print("\nBest Parameters:")
    print(best_params)

    print("\nBest CV Recall:")
    print(round(grid_search.best_score_, 4))

    print("\nTest Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {round(metric_value, 4)}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))

    # ========================================================
    # Save Best Model Locally
    # ========================================================

    artifact_dir = Path.cwd() / "modelbuilding" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "best_xgboost_predictive_maintenance_model.pkl"
    results_path = artifact_dir / "xgboost_gridsearch_results.csv"

    joblib.dump(best_model, model_path)

    results_df = pd.DataFrame(cv_results)
    results_df.to_csv(results_path, index=False)

    mlflow.log_artifact(str(model_path))
    mlflow.log_artifact(str(results_path))

    print(f"\nBest model saved locally at: {model_path}")

# ============================================================
# Register Best Model in Hugging Face Model Hub
# ============================================================

create_repo(
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    private=False,
    token=HF_TOKEN,
    exist_ok=True
)

api.upload_file(
    path_or_fileobj=str(model_path),
    path_in_repo="best_xgboost_predictive_maintenance_model.pkl",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN
)

api.upload_file(
    path_or_fileobj=str(results_path),
    path_in_repo="xgboost_gridsearch_results.csv",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN
)

print(f"\nBest model registered successfully in Hugging Face Model Hub: {MODEL_REPO_ID}")
