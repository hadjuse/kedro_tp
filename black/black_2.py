from lightgbm.sklearn import LGBMClassifier
from hyperopt import hp

MODEL_SPECS = {
    "name": "LightGBM",
    "class": LGBMClassifier,
    "max_evals": 20,
    "params": {
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
