import os
import warnings
from collections.abc import Callable
from typing import Any, TypedDict

import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe
from lightgbm.sklearn import LGBMClassifier
from matplotlib import pyplot as plt
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import RepeatedKFold

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")


class ModelSpec(TypedDict, total=True):
    """Model specification type."""

    name: str
    model_class: Callable[..., Any]
    params: dict[str, Any]
    override_schemas: dict[str, type]


MODELS: list[ModelSpec] = [
    {
        "name": "LightGBM",
        "model_class": LGBMClassifier,
        "params": {
            "objective": "binary",
            "verbose": -1,
            "learning_rate": hp.uniform("learning_rate", 0.001, 1),
            "num_iterations": hp.quniform("num_iterations", 100, 1000, 20),
            "max_depth": hp.quniform("max_depth", 4, 12, 6),
            "num_leaves": hp.quniform("num_leaves", 8, 128, 10),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "min_child_samples": hp.quniform("min_child_samples", 1, 20, 10),
            "reg_alpha": hp.choice("reg_alpha", [0, 1e-1, 1, 2, 5, 10]),
            "reg_lambda": hp.choice("reg_lambda", [0, 1e-1, 1, 2, 5, 10]),
        },
        "override_schemas": {
            "num_leaves": int,
            "min_child_samples": int,
            "max_depth": int,
            "num_iterations": int,
        },
    }
]


def get_model_config(instance: BaseEstimator) -> ModelSpec:
    """Return the configuration dictionary for the given model instance."""
    for model_spec in MODELS:
        model_cls = model_spec["model_class"]
        if isinstance(model_cls, type) and isinstance(instance, model_cls):
            return model_spec
    msg = f"Unsupported model: {type(instance)}"
    raise ValueError(msg)


def train_model(
    instance: BaseEstimator,
    training_set: tuple[pd.DataFrame, pd.Series | np.ndarray],
    params: dict[str, Any] | None = None,
) -> BaseEstimator:
    """Train a model with given parameters."""
    model_conf = get_model_config(instance)
    params = params or {}

    override_schemas = model_conf.get("override_schemas", {})
    for p in list(params.keys()):
        if p in override_schemas:
            params[p] = override_schemas[p](params[p])

    model = clone(instance)
    model.set_params(**params)
    model.fit(*training_set)
    return model


def optimize_hyp(
    instance: BaseEstimator,
    dataset: tuple[pd.DataFrame, pd.Series],
    search_space: dict[str, Any],
    metric: Callable[[Any, Any], float],
    max_evals: int = 40,
) -> dict[str, Any]:
    """Optimize hyperparameters using Bayesian optimization."""
    X, y = dataset

    def objective(params: dict[str, Any]) -> float:
        rep_kfold = RepeatedKFold(n_splits=4, n_repeats=1, random_state=42)
        scores_test = []

        for train_i, test_i in rep_kfold.split(X):
            X_fold_train = X.iloc[train_i, :]
            y_fold_train = y.iloc[train_i].values.flatten()
            X_fold_test = X.iloc[test_i, :]
            y_fold_test = y.iloc[test_i].values.flatten()

            model = train_model(
                instance=instance,
                training_set=(X_fold_train, y_fold_train),
                params=params,
            )
            scores_test.append(metric(y_fold_test, model.predict(X_fold_test)))

        return float(np.mean(scores_test))

    return fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=max_evals)


def save_pr_curve(X: pd.DataFrame, y: pd.Series, model: BaseEstimator) -> None:
    """Save precision-recall curve."""
    plt.figure(figsize=(16, 11))
    prec, recall, _ = precision_recall_curve(y, model.predict_proba(X)[:, 1], pos_label=1)
    PrecisionRecallDisplay(precision=prec, recall=recall).plot(ax=plt.gca())
    # ↑ supprime "pr_display ="
    plt.title("PR Curve", fontsize=16)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.savefig(os.path.expanduser("data/08_reporting/pr_curve.png"))
    plt.close()


def auto_ml(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_evals: int = 40,
    log_to_mlflow: bool = False,
    experiment_id: int = -1,
) -> dict[str, BaseEstimator | str]:
    """Run AutoML training pipeline."""
    X = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], ignore_index=True)

    y_train_flat = y_train.squeeze() if isinstance(y_train, pd.DataFrame) else y_train
    y_test_flat = y_test.squeeze() if isinstance(y_test, pd.DataFrame) else y_test
    y = pd.concat([pd.Series(y_train_flat), pd.Series(y_test_flat)], ignore_index=True)

    opt_models: list[dict[str, Any]] = []

    run_id = ""
    mlflow_model_uri = ""

    if log_to_mlflow:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_SERVER", "http://localhost:5000"))
        if experiment_id > 0:
            run = mlflow.start_run(experiment_id=str(experiment_id))
        else:
            mlflow.set_experiment("purchase_predict")
            run = mlflow.start_run()
        run_id = run.info.run_id

    for model_specs in MODELS:
        model_instance = model_specs["model_class"]()
        optimum_params = optimize_hyp(
            model_instance,
            dataset=(X, y),
            search_space=model_specs["params"],
            metric=lambda yt, yp: -f1_score(yt, yp),
            max_evals=max_evals,
        )

        model = train_model(
            model_instance,
            training_set=(pd.DataFrame(X_train), pd.Series(y_train_flat)),
            params=optimum_params,
        )

        opt_models.append(
            {
                "model": model,
                "name": model_specs["name"],
                "params": optimum_params,
                "score": f1_score(
                    pd.Series(y_test_flat),
                    model.predict(pd.DataFrame(X_test)),
                ),
            }
        )

    best_model = max(opt_models, key=lambda x: x["score"])

    if log_to_mlflow:
        try:
            model_metrics = {"f1": float(best_model["score"])}
            signature = infer_signature(
                pd.DataFrame(X_train),
                best_model["model"].predict(pd.DataFrame(X_train)),
            )

            save_pr_curve(
                pd.DataFrame(X_test),
                pd.Series(y_test_flat),
                best_model["model"],
            )

            mlflow.log_metrics(model_metrics)
            mlflow.log_params(best_model["params"])
            mlflow.log_artifacts("data/08_reporting", artifact_path="plots")
            mlflow.log_artifact("data/04_feature/transform_pipeline.pkl")

            mlflow_info = mlflow.sklearn.log_model(
                sk_model=best_model["model"],
                artifact_path="model",
                signature=signature,
            )
            mlflow_model_uri = mlflow_info.model_uri
        except Exception as exc:
            warnings.warn(
                f"MLflow logging skipped due to connection error: {exc}",
                stacklevel=2,
            )
        finally:
            mlflow.end_run()

    return {
        "model": best_model["model"],
        "mlflow_run_id": run_id,
        "mlflow_model_uri": mlflow_model_uri,
    }
