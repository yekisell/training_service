import click
import mlflow
import mlflow.sklearn
import yaml
import os
from xgboost import XGBRegressor
from time import time

from src.data_manip import read_data, train_valid_split_ts
from src.evaluating import mae, rmse, rmspe
from src.encoding import target_encoding

config_path = os.path.join("config/parameters.yaml")
config = yaml.safe_load(open(config_path))


def train(data_path: str, predicting_weeks: int) -> None:
    data = read_data(data_path)
    data_encoded = target_encoding(data, "Sales")

    X_train, X_valid, y_train, y_valid = train_valid_split_ts(
        data_encoded.copy(), "Sales", predicting_weeks
    )

    with mlflow.start_run():
        model = XGBRegressor(**config["model"])

        start = time()
        model.fit(X_train, y_train)
        end = time()

        y_pred = model.predict(X_valid)

        rmse_score = rmse(y_pred, y_valid)
        rsmpe_score = rmspe(y_pred, y_valid)
        mae_score = mae(y_pred, y_valid)

        mlflow.log_param("n_estimators", config["model"]["n_estimators"])
        mlflow.log_param("learning_rate", config["model"]["learning_rate"])
        mlflow.log_param("max_depth", config["model"]["max_depth"])
        mlflow.log_metric("rmse", rmse_score)
        mlflow.log_metric("rsmpe", rsmpe_score)
        mlflow.log_metric("mae", mae_score)

        click.echo(f"Rsmpe: {rsmpe_score}.")
        click.echo(f"Training time: {end - start}")

        mlflow.sklearn.log_model(model, artifact_path="model_xgb")
        mlflow.end_run()


if __name__ == "__main__":
    train(**config["train"])
