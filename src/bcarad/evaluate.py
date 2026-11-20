"""Discrimination, calibration and clinical utility (paper Section 2.5, Table 2, Figure 3, Supplementary Figure 2).

Re-implemented from the paper's Methods: the analysis code of the final revision is not available.
- AUROC and AUPRC at the tumour-region level; 95% CIs from a patient-level cluster bootstrap
  (resample patients with replacement, keep all regions of each sampled patient; 2000 resamples).
- DeLong test for paired AUROC differences between models.
- Calibration: LOWESS-smoothed calibration curve; intercept and slope of the logistic recalibration
  model logit(P(HG)) = a + b * logit(p) with Wald 95% CIs, which reproduces the values reported in the
  paper (0.43 [0.10-0.75] and 0.93 [0.67-1.18] on the surviving external predictions). Options:
  calibration-in-the-large intercept and patient-level bootstrap CIs. Brier score with bootstrap CI.
- Decision curve analysis: net benefit = TP/n - FP/n * t / (1 - t) against treat-all and treat-none.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

from .models import NAMES

EPS = 1e-6


def predict(df, ckpt):
    X = ckpt["scaler"].transform(df[ckpt["features"]])
    return ckpt["model"].predict_proba(X)[:, 1]


def cluster_bootstrap(y, p, groups, stat, n_boot, seed):
    """stat(y, p) on n_boot patient-level resamples (samples with one class only are skipped)."""
    rng = np.random.default_rng(seed)
    patients = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in patients}
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[g] for g in rng.choice(patients, size=len(patients), replace=True)])
        if y[idx].min() != y[idx].max():
            out.append(stat(y[idx], p[idx]))
    return np.array(out)


def ci(values):
    return np.percentile(values, [2.5, 97.5], axis=0)


def calibration_fit(y, p, intercept="joint"):
    """Logistic recalibration logit(P(y=1)) = a + b * logit(p).

    Returns (estimates, Wald 95% CIs) for (intercept, slope). intercept="joint" gives a from that model
    (the values reported in the paper); "offset" gives calibration-in-the-large (b fixed at 1).
    """
    lp = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
    with warnings.catch_warnings():  # perfect separation can occur in small bootstrap samples
        warnings.simplefilter("ignore")
        joint = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        fit_a = joint if intercept == "joint" else \
            sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit()
    est = np.array([fit_a.params[0], joint.params[1]])
    wald = np.array([fit_a.conf_int()[0], joint.conf_int()[1]])
    return est, wald


def net_benefit(y, p, thresholds):
    n = len(y)
    nb = []
    for t in thresholds:
        pos = p >= t
        nb.append(np.sum(pos & (y == 1)) / n - np.sum(pos & (y == 0)) / n * t / (1 - t))
    return np.array(nb)


def delong(y, p1, p2):
    """Two-sided DeLong test for two correlated AUROCs (Sun & Xu 2014 fast algorithm). Returns z, p."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def components(p):
        psi = (p[pos][:, None] > p[neg][None, :]) + 0.5 * (p[pos][:, None] == p[neg][None, :])
        return psi.mean(axis=1), psi.mean(axis=0)  # structural components V10 (positives), V01 (negatives)

    a10, a01 = components(p1)
    b10, b01 = components(p2)
    s10, s01 = np.cov(np.vstack([a10, b10])), np.cov(np.vstack([a01, b01]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    z = (a10.mean() - b10.mean()) / np.sqrt(var) if var > 0 else 0.0
    return z, 2 * norm.sf(abs(z))


def evaluate(preds, ckpts, cfg):
    """preds: one row per region with cohort, patient_id, y and one score column per model."""
    e = cfg["evaluate"]
    models = list(ckpts)
    rows, delong_rows, calib_rows, dca = [], [], [], {}
    thresholds = np.arange(*e["dca_thresholds"])
    for cohort in e["cohorts"]:
        d = preds[preds["cohort"] == cohort]
        if d.empty:
            continue
        y, g = d["y"].to_numpy(), d["patient_id"].to_numpy()
        for m in models:
            p = d[m].to_numpy()
            boot = cluster_bootstrap(y, p, g, lambda a, b: [roc_auc_score(a, b), average_precision_score(a, b)],
                                     e["n_bootstrap"], e["bootstrap_seed"])
            lo, hi = ci(boot)
            thr = ckpts[m]["threshold"]
            rows.append({"cohort": cohort, "model": NAMES[m], "regions": len(d), "patients": len(np.unique(g)),
                         "auroc": roc_auc_score(y, p), "auroc_lo": lo[0], "auroc_hi": hi[0],
                         "auprc": average_precision_score(y, p), "auprc_lo": lo[1], "auprc_hi": hi[1],
                         "threshold": thr, "sensitivity": np.mean(p[y == 1] >= thr),
                         "specificity": np.mean(p[y == 0] < thr)})
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                z, pval = delong(y, d[m1].to_numpy(), d[m2].to_numpy())
                delong_rows.append({"cohort": cohort, "model_1": NAMES[m1], "model_2": NAMES[m2], "z": z, "p": pval})
        if cohort == e["calibration_cohort"]:
            dca = {"threshold": thresholds, "treat_all": net_benefit(y, np.ones_like(y, float), thresholds),
                   "treat_none": np.zeros_like(thresholds)}
            for m in [m for m in e["calibration_models"] if m in models]:
                p = d[m].to_numpy()
                est, wald = calibration_fit(y, p, e["calibration_intercept"])
                if e["calibration_ci"] == "bootstrap":
                    boot = cluster_bootstrap(y, p, g, lambda a, b: calibration_fit(a, b, e["calibration_intercept"])[0],
                                             e["n_bootstrap"], e["bootstrap_seed"])
                    wald = ci(boot).T
                brier = cluster_bootstrap(y, p, g, brier_score_loss, e["n_bootstrap"], e["bootstrap_seed"])
                stats = [("intercept", est[0], *wald[0]), ("slope", est[1], *wald[1]),
                         ("brier", brier_score_loss(y, p), *ci(brier))]
                for name, value, lo, hi in stats:
                    calib_rows.append({"cohort": cohort, "model": NAMES[m], "statistic": name,
                                       "estimate": value, "lo": lo, "hi": hi})
                dca[NAMES[m]] = net_benefit(y, p, thresholds)
    return pd.DataFrame(rows), pd.DataFrame(delong_rows), pd.DataFrame(calib_rows), pd.DataFrame(dca)


def plot_roc(preds, models, cohort, path):
    d = preds[preds["cohort"] == cohort]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for m in models:
        fpr, tpr, _ = roc_curve(d["y"], d[m])
        ax.plot(fpr, tpr, label=f"{NAMES[m]} ({roc_auc_score(d['y'], d[m]):.2f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
    ax.set(xlabel="1 - specificity", ylabel="Sensitivity", title=f"ROC, {cohort} validation")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def lowess_band(y, p, groups, frac, n_boot, seed, grid):
    """Pointwise 95% band of the LOWESS calibration curve over patient-level bootstrap resamples."""
    rng = np.random.default_rng(seed)
    patients = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in patients}
    curves = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[g] for g in rng.choice(patients, size=len(patients), replace=True)])
        c = lowess(y[idx], p[idx], frac=frac, return_sorted=True)
        curves.append(np.interp(grid, c[:, 0], c[:, 1], left=np.nan, right=np.nan))
    return np.nanpercentile(np.array(curves), [2.5, 97.5], axis=0)


def plot_calibration_dca(preds, dca, calib, cfg, path):
    e = cfg["evaluate"]
    d = preds[preds["cohort"] == e["calibration_cohort"]]
    models = [m for m in e["calibration_models"] if m in d.columns]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
    for k, m in enumerate(models):
        y, p = d["y"].to_numpy(), d[m].to_numpy()
        curve = lowess(y, p, frac=e["lowess_frac"], return_sorted=True)
        line, = ax1.plot(curve[:, 0], curve[:, 1], label=NAMES[m])
        ax2.plot(dca["threshold"], dca[NAMES[m]], label=NAMES[m])
        if k == 0:  # band and recalibration estimates for the main model, as in Figure 3A
            grid = np.linspace(p.min(), p.max(), 100)
            lo, hi = lowess_band(y, p, d["patient_id"].to_numpy(), e["lowess_frac"], e["n_bootstrap"],
                                 e["bootstrap_seed"], grid)
            ax1.fill_between(grid, lo, hi, color=line.get_color(), alpha=0.2, lw=0, label="Bootstrap 95% band")
            c = calib[calib["model"] == NAMES[m]].set_index("statistic")
            ax1.text(0.03, 0.97, "\n".join(f"{s}: {c.loc[s, 'estimate']:.2f} ({c.loc[s, 'lo']:.2f}-{c.loc[s, 'hi']:.2f})"
                                           for s in ("intercept", "slope")),
                     transform=ax1.transAxes, va="top", fontsize=8)
    ax1.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
    ax1.set(xlabel="Predicted probability of HG", ylabel="Observed proportion of HG", xlim=(0, 1), ylim=(0, 1),
            title="Calibration (LOWESS)")
    ax1.legend(fontsize=8, loc="lower right")
    ax2.plot(dca["threshold"], dca["treat_all"], color="grey", label="Treat all")
    ax2.plot(dca["threshold"], dca["treat_none"], color="black", lw=0.8, label="Treat none")
    ax2.set(xlabel="Threshold probability", ylabel="Net benefit", title="Decision curve",
            ylim=(-0.05, max(0.05, float(np.nanmax(dca["treat_all"])) + 0.05)))
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
