import os
import pickle
import click
import wandb

import xgboost as xgb

from sklearn.metrics import log_loss, f1_score, accuracy_score, recall_score, precision_score


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option("--wandb_project", help="Name of Weights & Biases project")
@click.option("--wandb_entity", help="Name of Weights & Biases entity")
@click.option(
    "--data_artifact",
    help="Address of the Weights & Biases artifact holding the preprocessed data",
)
@click.option("--random_state", default=0, help="Random state")
def run_train(
    wandb_project: str,
    wandb_entity: str,
    data_artifact: str,
    random_state: int,
):
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name="xgb_classifier",
        job_type="train",
        config={"random_state": random_state},
    )

    # Fetch the preprocessed dataset from artifacts
    artifact = wandb.use_artifact(data_artifact, type="preprocessed_dataset")
    artifact_dir = artifact.download()

    X_train, y_train = load_pickle(os.path.join(artifact_dir, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(artifact_dir, "test.pkl"))

    model = xgb.XGBClassifier(random_state=0)
    model.fit(X_train, y_train)

    # Make predictions
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

    with open("xgb_classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    artf = wandb.Artifact(f"XGBClassifier-model", type="model")
    artf.add_file("xgb_classifier.pkl")
    wandb.log_artifact(artf)


if __name__ == "__main__":
    run_train()

# python v1.py --wandb_project quora-sim-question-wandb --wandb_entity uzoagulup --data_artifact "uzoagulup/quora-sim-question-wandb/Quora-sim:v0" 
