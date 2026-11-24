"""Coe-Thr-Lasso feature selection on the training cohort (paper Section 2.3, Supplementary Section S1).

1. drop features with zero variance (or missing values) in the training set
2. keep features with Welch's t-test p < 0.05 between HG and LG regions
3. fit StandardScaler + LassoCV once and take |coefficient| of every remaining feature
4. split the training set once; for each threshold t, fit LassoCV on part A with the features
   whose |coefficient| > t and score mean squared error on part B
5. keep the features above the threshold with the lowest error
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _balance_age(age, a, b, max_iter=1000):
    """Swap single regions between the two parts until their mean age differs by < 1 year."""
    a, b = list(a), list(b)
    for _ in range(max_iter):
        mean_a, mean_b = age[a].mean(), age[b].mean()
        if abs(mean_a - mean_b) < 1:
            break
        to_b = a[int(np.argmin(np.abs(age[a] - mean_b)))]
        to_a = b[int(np.argmin(np.abs(age[b] - mean_a)))]
        a.remove(to_b)
        b.remove(to_a)
        a.append(to_a)
        b.append(to_b)
    return a, b


def _holdout_split(train, s):
    """Positions of parts A (fit) and B (score) inside the training set."""
    if s["holdout_by_patient"]:
        n_splits = round(1 / s["holdout_fraction"])
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=s["holdout_random_state"])
        return next(sgkf.split(train, train["y"], groups=train["patient_id"]))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=s["holdout_fraction"], random_state=s["holdout_random_state"])
    a, b = next(sss.split(train, train["y"]))
    return _balance_age(train["age"].to_numpy(dtype=float), a, b) if s["balance_age"] else (a, b)


def coe_thr_lasso(train, feature_cols, cfg):
    """train: one row per training region with feature columns, y, age and patient_id."""
    s = cfg["selection"]
    X = train[feature_cols]
    steps = {"extracted": X.shape[1]}

    keep = X.columns[(X.notna().all()) & (X.var() > 0)]
    steps["non-zero variance"] = len(keep)

    hg, lg = X.loc[train["y"] == 1, keep], X.loc[train["y"] == 0, keep]
    with warnings.catch_warnings():  # precision warnings from near-constant features
        warnings.simplefilter("ignore")
        p = ttest_ind(hg, lg, equal_var=False, nan_policy="omit").pvalue
    sig = list(keep[np.asarray(p) < s["p_value"]])
    steps[f"t-test p < {s['p_value']}"] = len(sig)

    c = s["coef_lasso"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = make_pipeline(StandardScaler(),
                             LassoCV(cv=c["cv"], random_state=c["random_state"], max_iter=c["max_iter"]))
        pipe.fit(train[sig], train["y"])
    coef_abs = np.abs(pipe[-1].coef_)

    train = train.reset_index(drop=True)
    a_pos, b_pos = _holdout_split(train, s)
    a, b = train.iloc[a_pos], train.iloc[b_pos]
    thresholds = np.arange(coef_abs.min(), coef_abs.max() + s["threshold_step"], s["threshold_step"])
    t = s["threshold_lasso"]
    curve = []
    for thr in thresholds:
        cols = [f for f, w in zip(sig, coef_abs) if w > thr]
        if not cols:
            curve.append((thr, 0, np.inf))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LassoCV(cv=t["cv"], random_state=t["random_state"])
            model.fit(a[cols], a["y"])  # features are not scaled in this step
        mse = mean_squared_error(b["y"], model.predict(b[cols]))
        curve.append((thr, len(cols), mse))
    curve = pd.DataFrame(curve, columns=["threshold", "n_features", "mse"])

    best = curve.loc[curve["mse"].idxmin()]
    selected = [f for f, w in zip(sig, coef_abs) if w > best["threshold"]]
    steps[f"Lasso threshold {best['threshold']:.3f} (min MSE {best['mse']:.3f})"] = len(selected)
    coefs = pd.DataFrame({"feature": sig, "abs_coef": coef_abs}).sort_values("abs_coef", ascending=False)
    return selected, curve, coefs, steps


def plot_curve(curve, path):
    best = curve.loc[curve["mse"].idxmin()]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(curve["threshold"], curve["mse"].replace(np.inf, np.nan))
    ax1.axvline(best["threshold"], color="r", ls="--", label=f"threshold {best['threshold']:.3f}")
    ax1.set_ylabel("Mean squared error")
    ax1.legend()
    ax2.plot(curve["threshold"], curve["n_features"])
    ax2.axvline(best["threshold"], color="r", ls="--", label=f"{int(best['n_features'])} features")
    ax2.set_xlabel("Lasso coefficient threshold")
    ax2.set_ylabel("Selected features")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
