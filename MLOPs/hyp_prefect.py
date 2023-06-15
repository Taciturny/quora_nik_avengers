import os
import mlflow
from sklearn.metrics import log_loss, accuracy_score, recall_score, f1_score, precision_score
import xgboost as xgb
from prefect import task, Flow
import pickle
from optuna.samplers import TPESampler
import optuna

@task
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@task
def objective(trial, train_data, test_data):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 30, 1),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 1, 10, 1),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 5, 1),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 2, 1),
        'random_state': 42,
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        'n_jobs': -1
    }

    with mlflow.start_run(run_name="xgb_classifier-opt", nested=True):
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(*train_data)
        y_pred_proba = model.predict_proba(test_data[0])[:, 1]
        y_pred = model.predict(test_data[0])

        # Calculate log loss
        log_loss_test = log_loss(test_data[1], y_pred_proba)

        # Calculate accuracy
        accuracy = accuracy_score(test_data[1], y_pred)

        # Calculate recall
        recall = recall_score(test_data[1], y_pred)

        # Calculate F1 score
        f1 = f1_score(test_data[1], y_pred)

        # Calculate precision
        precision = precision_score(test_data[1], y_pred)

        mlflow.log_metrics({
            "Log Loss": log_loss_test,
            "Accuracy": accuracy,
            "Recall": recall,
            "F1": f1,
            "Precision": precision
        })

        return log_loss_test, accuracy, recall, f1, precision

@Flow
def main_flow(data_path: str, n_trials: int):
    if mlflow.active_run():
        mlflow.end_run()

    mlflow.start_run(run_name="xgb_classifier-opt", tags={"algo": "XGBoost", "dev": "NIKAvengers"})
    
    train_data = load_pickle(os.path.join(data_path, "train.pkl"))
    test_data = load_pickle(os.path.join(data_path, "test.pkl"))
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    for _ in range(n_trials):
        trial = study.ask()
        objective_value = objective(trial, train_data, test_data)
        study.tell(trial, objective_value)
    
    mlflow.end_run()  

if __name__ == '__main__':
    main_flow(data_path='./output/', n_trials=10)
