import warnings
from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe
from lightgbm.sklearn import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedKFold

warnings.filterwarnings("ignore")


class ModelSpec(TypedDict, total=True):
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
    """Returns the configuration dictionary for the given model instance."""
    for model_spec in MODELS:
        model_cls = model_spec["model_class"]
        if isinstance(model_cls, type) and isinstance(instance, model_cls):
            return model_spec
    raise ValueError(f"Unsupported model: {type(instance)}")


def train_model(
    instance: BaseEstimator,
    training_set: tuple[pd.DataFrame, pd.Series | np.ndarray],
    params: dict[str, Any] | None = None,
) -> BaseEstimator:
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


def auto_ml(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    max_evals: int = 40,
) -> BaseEstimator:
    X = pd.concat((X_train, X_test), axis=0)
    y = pd.concat((y_train, y_test), axis=0)

    opt_models: list[dict[str, Any]] = []

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
            training_set=(X_train, y_train),
            params=optimum_params,
        )

        opt_models.append(
            {
                "model": model,
                "name": model_specs["name"],
                "params": optimum_params,
                "score": f1_score(y_test, model.predict(X_test)),
            }
        )

    best_model = max(opt_models, key=lambda x: x["score"])
    return best_model["model"]
