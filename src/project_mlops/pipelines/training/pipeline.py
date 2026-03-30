from kedro.pipeline import Pipeline, Node
from .nodes import auto_ml


def create_pipeline(**kwargs):
    return Pipeline(
        [
            Node(
                auto_ml,
                [
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "params:automl_max_evals",
                    "params:mlflow_enabled",
                    "params:mlflow_experiment_id",
                ],
                dict(
                    model="model",
                    mlflow_run_id="mlflow_run_id",
                    mlflow_model_uri="mlflow_model_uri",
                ),
            )
        ]
    )
