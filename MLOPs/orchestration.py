import os
import mlflow
from sklearn.metrics import log_loss, f1_score, accuracy_score, recall_score
import xgboost as xgb
from prefect import task, Flow
import pickle
import numpy as np
import pandas as pd
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule
from datetime import timedelta

@task
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@task
def train_model(X_train, y_train, X_test, y_test):
    # Create and train XGBoost model
    best_params = {
    "learning_rate": 0.1,
    "max_depth": 6,  
    "min_child_weight": 2,  
    "objective": "binary:logistic",
    "seed": 42,
}

    mlflow.log_params(best_params)

    dtrain = xgb.DMatrix(X_train, y_train)
    deval = xgb.DMatrix(X_test, y_test)

    booster = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=10,
        evals=[(dtrain, "training"), (deval, "evaluation")], 
        early_stopping_rounds=20,
    )

    return booster

@task
def evaluate_model(model, X_test, y_test):
    # Convert X_test to DMatrix
    dtest = xgb.DMatrix(X_test)

    # Make predictions
    y_pred = model.predict(dtest)
    y_pred_proba = 1.0 / (1.0 + np.exp(-y_pred))
    y_pred_binary = y_pred.round()

    # Calculate log loss
    log_loss_test = log_loss(y_test, y_pred_proba)
    mlflow.log_metric("Log Loss", log_loss_test)

    # Calculate other classification metrics
    f1score_test = f1_score(y_test, y_pred_binary)
    mlflow.log_metric("F1 Score", f1score_test)

    accuracy_test = accuracy_score(y_test, y_pred_binary)
    mlflow.log_metric("Accuracy Score", accuracy_test)

    recall_test = recall_score(y_test, y_pred_binary)
    mlflow.log_metric("Recall Score", recall_test)


@Flow
def main_flow(data_path: str):
    with mlflow.start_run(run_name="xgb_classifier", tags={"algo": "XGBoost", "dev": "NIKAvengers"}):
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

        model = train_model(X_train, y_train, X_test, y_test)

        evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main_flow(data_path='./output/')


deployment = Deployment.build_from_flow(
    flow=main_flow,
    name="model_training",
    schedule=IntervalSchedule(interval=timedelta(minutes=60)),
    work_queue_name="ml"
)

deployment.apply()
