import os
import pickle
import click
import mlflow
from sklearn.metrics import log_loss, f1_score, accuracy_score, recall_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("StackingClassifier")
mlflow.sklearn.autolog()


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed Quora question similarity data was saved"
)


def run_train(data_path: str):
    with mlflow.start_run(run_name="stacking_classifier", tags={"algo": "Stacking", "dev": " NIKAvengers"}):
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

        # Define the base models
        base_models = [
            ('lgm', LGBMClassifier(random_state=0)),
            ('xgb', xgb.XGBClassifier(random_state=0)),
            ('randomforest', RandomForestClassifier(random_state=0))
        ]

        # Specify the final estimator
        final_estimator = xgb.XGBClassifier(tree_method='gpu_hist', random_state=0)

        # Instantiate the StackingClassifier
        sc = StackingClassifier(estimators=base_models, final_estimator=final_estimator)
        
        # Fit the StackingClassifier
        sc.fit(X_train, y_train)
        
        # Make predictions
        y_pred = sc.predict(X_test)
        y_pred_proba = sc.predict_proba(X_test)

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


if __name__ == '__main__':
    run_train()
