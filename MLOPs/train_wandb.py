import os
import pickle
import click
import wandb

from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb

from sklearn.metrics import log_loss, f1_score, accuracy_score, recall_score


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
        name="stacking_classifier",
        job_type="train",
        config={"random_state": random_state},
    )

    # Fetch the preprocessed dataset from artifacts
    artifact = wandb.use_artifact(data_artifact, type="preprocessed_dataset")
    artifact_dir = artifact.download()

    X_train, y_train = load_pickle(os.path.join(artifact_dir, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(artifact_dir, "test.pkl"))

    # Define the base models
    base_models = [
        ("lgbm", LGBMClassifier(random_state=random_state)),
        ("xgb", xgb.XGBClassifier(random_state=random_state)),
    ]

    # Specify the final estimator
    final_estimator = xgb.XGBClassifier(random_state=random_state)

    # Instantiate the StackingClassifier
    sc = StackingClassifier(estimators=base_models, final_estimator=final_estimator)

    # Fit the StackingClassifier
    sc.fit(X_train, y_train)

    # Make predictions
    y_pred = sc.predict(X_test)
    y_pred_proba = sc.predict_proba(X_test)

    # Calculate log loss
    log_loss_test_score = log_loss(y_test, y_pred_proba)
    wandb.log({"Log Loss": log_loss_test_score})

    # Generate the classification report
    f1score_test = f1_score(y_test, y_pred)
    wandb.log({"F1 Score": f1score_test})

    accuracy_test = accuracy_score(y_test, y_pred)
    wandb.log({"Accuracy Score": accuracy_test})

    recall_test = recall_score(y_test, y_pred)
    wandb.log({"Recall Score": recall_test})

    with open("classifier.pkl", "wb") as f:
        pickle.dump(sc, f)

    artf = wandb.Artifact(f"Stacking_Classifier-model", type="model")
    artf.add_file("classifier.pkl")
    wandb.log_artifact(artf)


if __name__ == "__main__":
    run_train()


#python train.py --wandb_project quora-sim-qustion-wandb --wandb_entity uzoagulup --data_artifact "uzoagulup/quora-sim-qustion-wandb/stacking:v0
 # artifact = wandb.use_artifact(data_artifact, type="preprocessed_dataset")
    # artifact_dir = artifact.download()
# python train.py --wandb_project quora-sim-question-wandb --wandb_entity uzoagulup --data_artifact "uzoagulup/quora-sim-question-wandb/Quora-sim:v0
