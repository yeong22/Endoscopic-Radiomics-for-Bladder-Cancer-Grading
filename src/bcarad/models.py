"""The five classifiers of the paper and their Optuna search spaces (paper Section 2.4)."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

MODELS = {"lr": LogisticRegression, "svm": SVC, "rf": RandomForestClassifier, "xgb": XGBClassifier,
          "lgbm": LGBMClassifier}
NAMES = {"lr": "LR", "svm": "SVM", "rf": "RF", "xgb": "XGBoost", "lgbm": "LightGBM"}


def class_weights(y):
    """Inverse-frequency ("balanced") weights from the training labels, as {0: w0, 1: w1}."""
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return {0: float(w[0]), 1: float(w[1])}


def model_config(name, params, y_train, cfg, early_stopping=False):
    """Constructor arguments of a model: tuned/published params + fixed settings of the original code."""
    w = class_weights(y_train)
    seed, n_jobs = cfg["seed"], cfg["train"]["n_jobs"]
    fixed = {
        "lr": {"max_iter": 1000, "class_weight": w, "random_state": seed},
        "svm": {"kernel": "rbf", "probability": True, "class_weight": w, "random_state": seed},
        "rf": {"class_weight": w, "n_jobs": n_jobs, "random_state": seed},
        "xgb": {"scale_pos_weight": w[1] / w[0], "n_jobs": n_jobs, "random_state": seed},
        "lgbm": {"scale_pos_weight": w[1] / w[0], "n_jobs": n_jobs, "random_state": seed, "verbose": -1},
    }[name]
    conf = {**fixed, **(params or {})}
    if early_stopping and name in cfg["train"]["early_stopping_rounds"]:
        conf["early_stopping_rounds"] = cfg["train"]["early_stopping_rounds"][name]
    return conf


def make_model(name, conf):
    return MODELS[name](**conf)


def suggest(trial, space):
    """Sample one parameter set from a config search space (see configs/default.yaml)."""
    params = {}
    for key, spec in space.items():
        if isinstance(spec, dict):
            params[key] = trial.suggest_categorical(key, spec["choices"])
        elif all(isinstance(v, int) for v in spec[:2]):
            params[key] = trial.suggest_int(key, spec[0], spec[1])
        else:
            params[key] = trial.suggest_float(key, float(spec[0]), float(spec[1]), log=len(spec) > 2)
    return params
