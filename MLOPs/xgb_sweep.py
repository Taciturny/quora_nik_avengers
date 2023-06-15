import os
import pickle
import click
from functools import partial

import wandb

import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run_train(
    data_artifact: str,
):
    wandb.init(name="xgb_classifier-hyp",)
    config = wandb.config

    # Fetch the preprocessed dataset from artifacts
    artifact = wandb.use_artifact(data_artifact,  type="preprocessed_dataset")
    data_path = artifact.download()

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    model = xgb.XGBClassifier(**config, random_state=0)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba_train = model.predict_proba(X_train)
    y_pred_proba_test = model.predict_proba(X_test)


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
    wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)
    wandb.sklearn.plot_roc(y_test, y_pred_proba_test, labels=label_names)
    wandb.sklearn.plot_precision_recall(y_test, y_pred_proba_test, labels=label_names)
    wandb.sklearn.plot_confusion_matrix(y_test, y_pred_test, labels=label_names)


    with open("xgbclassifier-hyp.pkl", "wb") as f:
        pickle.dump(model, f)

    artifact = wandb.Artifact(f"{wandb.run.id}-model", type="model")
    artifact.add_file("xgbclassifier-hyp.pkl")
    wandb.log_artifact(artifact)


SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "Log Loss", "goal": "minimize"},
    "parameters": {
        "n_estimators": {
            "distribution": "int_uniform",
            "min": 10,
            "max": 100,        
        },
        "max_depth": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 20,
        },
        "learning_rate": {
            "distribution": "uniform",
            "min": 0.01,
            "max": 0.2,
        },
        "min_child_weight": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 10.0,
        },
    }
}


@click.command()
@click.option("--wandb_project", help="Name of Weights & Biases project")
@click.option("--wandb_entity", help="Name of Weights & Biases entity")
@click.option(
    "--data_artifact",
    help="Address of the Weights & Biases artifact holding the preprocessed data",
)
@click.option("--count", default=30, help="Number of iterations in the sweep")
def run_sweep(wandb_project: str, wandb_entity: str, data_artifact: str, count: int):
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=wandb_project, entity=wandb_entity)
    wandb.agent(sweep_id, partial(run_train, data_artifact=data_artifact), count=count)

if __name__ == "__main__":
    run_sweep()


#python v2.py --wandb_project quora-sim-question-wandb --wandb_entity uzoagulup --data_artifact "uzoagulup/quora-sim-question-wandb/Quora-sim:v0" --count 30

