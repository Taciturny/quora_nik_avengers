import os
import pickle
import click
from functools import partial

import wandb

from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run_train(
    data_artifact: str,
):
    wandb.init()
    config = wandb.config

    # Fetch the preprocessed dataset from artifacts
    artifact = wandb.use_artifact(data_artifact,  type="preprocessed_dataset")
    data_path = artifact.download()

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    # Define the base models
    base_models = [
        (
            "lgbm",
            LGBMClassifier(
                random_state=0,
                **config,
                min_child_samples=config.lgbm__min_child_samples,
            ),
        ),
        (
            "xgb",
            xgb.XGBClassifier(
                random_state=0,
                **config,
                min_child_weight=config.xgb__min_child_weight,
            ),
        ),
    ]

    # Specify the final estimator
    final_estimator = xgb.XGBClassifier(random_state=0)

    # Instantiate the StackingClassifier
    sc = StackingClassifier(estimators=base_models, final_estimator=final_estimator)

    # Fit the StackingClassifier
    sc.fit(X_train, y_train)
    y_pred_train = sc.predict(X_train)
    y_pred_test = sc.predict(X_test)
    y_pred_proba_train = sc.predict_proba(X_train)
    y_pred_proba_test = sc.predict_proba(X_test)

    # Log Metrics to Weights & Biases
    wandb.log({
        "Train/Accuracy": accuracy_score(y_train, y_pred_train),
        "Testing/Accuracy": accuracy_score(y_test, y_pred_test),
        "Train/Presicion": precision_score(y_train, y_pred_train),
        "Testing/Presicion": precision_score(y_test, y_pred_test),
        "Train/Recall": recall_score(y_train, y_pred_train),
        "Testing/Recall": recall_score(y_test, y_pred_test),
        "Train/F1-Score": f1_score(y_train, y_pred_train),
        "Testing/F1-Score": f1_score(y_test, y_pred_test),
        "Train/Logloss-Score": log_loss(y_train, y_pred_proba_train),
        "Testing/Logloss-Score": log_loss(y_test, y_pred_proba_test),
    })

    # Plot plots to Weights & Biases
    label_names = ["is-duplicate", "Not-duplicate"]
    wandb.sklearn.plot_class_proportions(y_train, y_test, label_names)
    wandb.sklearn.plot_summary_metrics(sc, X_train, y_train, X_test, y_test)
    wandb.sklearn.plot_roc(y_test, y_pred_proba_test, labels=label_names)
    wandb.sklearn.plot_precision_recall(y_test, y_pred_proba_test, labels=label_names)
    wandb.sklearn.plot_confusion_matrix(y_test, y_pred_test, labels=label_names)


    with open("stackedclassifier-hyp.pkl", "wb") as f:
        pickle.dump(sc, f)

    artifact = wandb.Artifact(f"{wandb.run.id}-model", type="model")
    artifact.add_file("stackedclassifier-hyp.pkl")
    wandb.log_artifact(artifact)


SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "Log Loss", "goal": "minimize"},
    "parameters": {
        "lgbm__n_estimators": {
            "distribution": "int_uniform",
            "min": 10,
            "max": 100,
        },
        "lgbm__max_depth": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 20,
        },
        "lgbm__min_child_samples": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 50,
        },
        "xgb__n_estimators": {
            "distribution": "int_uniform",
            "min": 10,
            "max": 100,
        },
        "xgb__max_depth": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 20,
        },
        "xgb__learning_rate": {
            "distribution": "uniform",
            "min": 0.01,
            "max": 0.2,
        },
        "xgb__min_child_weight": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 10.0,
        },
    },
}

@click.command()
@click.option("--wandb_project", help="Name of Weights & Biases project")
@click.option("--wandb_entity", help="Name of Weights & Biases entity")
@click.option(
    "--data_artifact",
    help="Address of the Weights & Biases artifact holding the preprocessed data",
)
@click.option("--count", default=5, help="Number of iterations in the sweep")
def run_sweep(wandb_project: str, wandb_entity: str, data_artifact: str, count: int):
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=wandb_project, entity=wandb_entity)
    wandb.agent(sweep_id, partial(run_train, data_artifact=data_artifact), count=count)

if __name__ == "__main__":
    run_sweep()


#python sweep.py --wandb_project quora-sim-question-wandb --wandb_entity uzoagulup --data_artifact "uzoagulup/quora-sim-question-wandb/Quora-sim:v1" --count 5

