"""Hyperparameter tuning and final fit on the training cohort (paper Section 2.4).

Each model is tuned with Optuna (TPE sampler) to maximise the mean validation AUROC of a
patient-grouped, stratified 5-fold cross-validation inside the training cohort. The final model is
refit on the whole training cohort with the best parameters. The internal and external validation
cohorts are not used here. The decision threshold for sensitivity/specificity is the Youden point of
the out-of-fold training predictions.
"""
import json

import joblib
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .config import out_path
from .models import make_model, model_config, suggest


def cross_validate(name, params, X, y, groups, cfg):
    """Mean validation AUROC over the folds and the out-of-fold predictions."""
    t = cfg["train"]
    folds = StratifiedGroupKFold(n_splits=t["cv_folds"], shuffle=True, random_state=t["cv_random_state"])
    oof, aucs = np.zeros(len(y)), []
    for tr, va in folds.split(X, y, groups):
        model = make_model(name, model_config(name, params, y[tr], cfg, early_stopping=True))
        fit_kw = {}
        if name == "xgb":
            fit_kw = {"eval_set": [(X[va], y[va])], "verbose": False}
        elif name == "lgbm":
            fit_kw = {"eval_set": [(X[va], y[va])]}
        model.fit(X[tr], y[tr], **fit_kw)
        oof[va] = model.predict_proba(X[va])[:, 1]
        if len(np.unique(y[va])) == 2:  # a tiny cohort can leave one grade out of a fold
            aucs.append(roc_auc_score(y[va], oof[va]))
    return float(np.mean(aucs)), oof


def tune(name, X, y, groups, cfg):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = cfg["search_space"][name]
    sampler = optuna.samplers.TPESampler(seed=cfg["train"]["sampler_seed"])
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: cross_validate(name, suggest(trial, space), X, y, groups, cfg)[0],
                   n_trials=cfg["train"]["n_trials"][name], show_progress_bar=True)
    return study.best_params


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def train_models(train, features, cfg, models):
    """train: training-cohort regions with feature columns, y and patient_id. Saves one file per model."""
    scaler = StandardScaler().fit(train[features])  # fitted on training regions only
    X = scaler.transform(train[features])
    y, groups = train["y"].to_numpy(), train["patient_id"].to_numpy()
    summary = {}
    for name in models:
        if cfg["train"]["tune"]:
            params = tune(name, X, y, groups, cfg)
        else:
            params = cfg["published_params"].get(name)
            if params is None:
                print(f"[train] {name}: no published parameters, skipped (set train.tune: true)")
                continue
        cv_auroc, oof = cross_validate(name, params, X, y, groups, cfg)
        conf = model_config(name, params, y, cfg)
        model = make_model(name, conf).fit(X, y)
        ckpt = {"model_name": name, "model_config": conf, "model": model, "scaler": scaler,
                "features": list(features), "cv_auroc": cv_auroc, "threshold": youden_threshold(y, oof)}
        joblib.dump(ckpt, out_path(cfg, "models", f"{name}.joblib"))
        summary[name] = {"params": params, "cv_auroc": round(cv_auroc, 4), "threshold": ckpt["threshold"]}
        print(f"[train] {name}: CV AUROC {cv_auroc:.3f}")
    with open(out_path(cfg, "models", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def load_model(cfg, name):
    """Rebuild works too: make_model(name, ckpt["model_config"])."""
    return joblib.load(out_path(cfg, "models", f"{name}.joblib"))
