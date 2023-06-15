import os
import pickle
import click
import mlflow
import optuna

from optuna.samplers import TPESampler
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import log_loss, f1_score, accuracy_score, recall_score


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("stacking-classifier-hyperopt")


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed data was saved"
)
@click.option(
    "--num_trials",
    default=5,
    help="The number of parameter evaluations for the optimizer to explore"
)


def run_optimization(data_path: str, num_trials: int):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    def objective(trial):
        with mlflow.start_run():
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 30, 1),
                'max_depth': trial.suggest_int('max_depth', 1, 10, 1),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 5, 1),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 2, 1),
                'random_state': 42,
                'n_jobs': -1
            }
            mlflow.log_params(params)

            stacking = StackingClassifier(**params)
            stacking.fit(X_train, y_train)
            y_pred = stacking.predict(X_test)
            y_pred_proba = stacking.predict_proba(X_test)

            # Calculate log loss
            log_loss_test_score = log_loss(y_test, y_pred_proba)
            mlflow.log_metric("Log Loss", log_loss_test_score)

            # Generate the classification report
            f1score_test = f1_score(y_test, y_pred)
            mlflow.log_metric("F1 Score", f1score_test)

            accuracy_test = accuracy_score(y_test, y_pred)
            mlflow.log_metric("Accuracy Score", accuracy_test)

            recall_test = recall_score(y_test, y_pred)
            mlflow.log_metric("Recall Score", recall_test)

        return log_loss_test_score, f1score_test, accuracy_test, recall_test

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


if __name__ == '__main__':
    run_optimization()
