"""Bladder cancer grade prediction from white-light cystoscopy radiomics.

    python main.py --step all --config configs/default.yaml

Steps (results are written to outputs/<run_name>/):
    split     training, internal and external cohorts (patient level)
    extract   preprocessing and PyRadiomics features
    select    Coe-Thr-Lasso, or the 19 features of the paper
    train     Optuna tuning with 5-fold CV, then fit on the training cohort
    evaluate  AUROC, AUPRC, DeLong test, calibration, decision curve
    explain   SHAP and GWL10 maps
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from bcarad import data, evaluate, explain, features, selection, train  # noqa: E402
from bcarad.config import load_config, out_path  # noqa: E402
from bcarad.models import NAMES  # noqa: E402

STEPS = ["split", "extract", "select", "train", "evaluate", "explain"]
PREFIXES = ("gray_", "r_", "g_", "b_")


def load_table(cfg):
    regions = pd.read_csv(out_path(cfg, "regions_split.csv"))
    feats = pd.read_csv(out_path(cfg, "features.csv"))
    return regions.merge(feats, on="region_id", how="inner")


def step_split(cfg, args):
    df = data.read_regions(cfg)
    if "cohort" not in df.columns or args.resplit:
        df = data.split_cohorts(df, cfg)
    df.to_csv(out_path(cfg, "regions_split.csv"), index=False)
    summary = df.groupby("cohort").agg(patients=("patient_id", "nunique"), regions=("region_id", "size"),
                                       HG_regions=("y", "sum"))
    print(summary.reindex(["train", "internal", "external"]).to_string())


def step_extract(cfg, args):
    regions = pd.read_csv(out_path(cfg, "regions_split.csv"))
    feats, errors = features.extract_all(regions, cfg)
    feats.to_csv(out_path(cfg, "features.csv"), index=False)
    print(f"[extract] {len(feats)} regions x {feats.shape[1] - 1} features")
    if errors:
        out_path(cfg, "extraction_errors.txt").write_text("\n".join(errors) + "\n")
        print(f"[extract] {len(errors)} regions failed, see extraction_errors.txt")


def step_select(cfg, args):
    df = load_table(cfg)
    s = cfg["selection"]
    if s["use"] == "published":
        selected = s["published_features"]
        missing = [f for f in selected if f not in df.columns]
        if missing:
            raise SystemExit(f"[select] published features missing from features.csv: {missing}")
    else:
        cols = [c for c in df.columns if c.startswith(PREFIXES)]
        selected, curve, coefs, steps = selection.coe_thr_lasso(df[df["cohort"] == "train"], cols, cfg)
        curve.to_csv(out_path(cfg, "selection", "threshold_curve.csv"), index=False)
        coefs.to_csv(out_path(cfg, "selection", "lasso_coefficients.csv"), index=False)
        selection.plot_curve(curve, out_path(cfg, "selection", "threshold_curve.png"))
        for name, n in steps.items():
            print(f"[select] {name}: {n}")
    out_path(cfg, "selected_features.txt").write_text("\n".join(selected) + "\n")
    print(f"[select] {len(selected)} features -> selected_features.txt")


def step_train(cfg, args):
    df = load_table(cfg)
    selected = out_path(cfg, "selected_features.txt").read_text().split()
    train.train_models(df[df["cohort"] == "train"], selected, cfg, args.models or cfg["train"]["models"])


def step_evaluate(cfg, args):
    df = load_table(cfg)
    names = [m for m in (args.models or cfg["train"]["models"]) if out_path(cfg, "models", f"{m}.joblib").exists()]
    if not names:
        raise SystemExit("[evaluate] no trained model found, run --step train first")
    ckpts = {m: train.load_model(cfg, m) for m in names}
    preds = df[["region_id", "patient_id", "cohort", "y"]].copy()
    for m, ckpt in ckpts.items():
        preds[m] = evaluate.predict(df, ckpt)
    preds.to_csv(out_path(cfg, "results", "predictions.csv"), index=False)

    metrics, dl, calib, dca = evaluate.evaluate(preds, ckpts, cfg)
    metrics.to_csv(out_path(cfg, "results", "metrics.csv"), index=False)
    dl.to_csv(out_path(cfg, "results", "delong.csv"), index=False)
    calib.to_csv(out_path(cfg, "results", "calibration.csv"), index=False)
    dca.to_csv(out_path(cfg, "results", "dca.csv"), index=False)
    for cohort in metrics["cohort"].unique():
        evaluate.plot_roc(preds, names, cohort, out_path(cfg, "results", f"roc_{cohort}.png"))
    if not dca.empty:
        evaluate.plot_calibration_dca(preds, dca, calib, cfg, out_path(cfg, "results", "calibration_dca.png"))

    def with_ci(col):
        return [f"{a:.2f} ({lo:.2f}-{hi:.2f})" for a, lo, hi in metrics[[col, f"{col}_lo", f"{col}_hi"]].values]

    show = metrics.assign(AUROC=with_ci("auroc"), AUPRC=with_ci("auprc"))
    print(show[["cohort", "model", "regions", "patients", "AUROC", "AUPRC"]].to_string(index=False))
    if not calib.empty:
        print(calib.round(3).to_string(index=False))


def step_explain(cfg, args):
    x = cfg["explain"]
    df = load_table(cfg)
    d = df[df["cohort"] == x["cohort"]].reset_index(drop=True)
    if not out_path(cfg, "models", f"{x['model']}.joblib").exists():
        raise SystemExit(f"[explain] {x['model']} is not trained, run --step train first")
    ckpt = train.load_model(cfg, x["model"])
    out_dir = cfg["out"] / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)

    values, X = explain.shap_values(ckpt, d, cfg)
    values.assign(region_id=d.loc[values.index, "region_id"]).to_csv(out_dir / "shap_values.csv", index=False)
    explain.plot_shap(values, X, out_dir)
    ranking = values.abs().mean().sort_values(ascending=False)
    print(f"[explain] top SHAP features ({NAMES[x['model']]}):\n{ranking.head(5).round(4).to_string()}")

    p = evaluate.predict(d, ckpt)
    k = max(1, x["map_examples"] // 2)
    order = np.argsort(p)
    for i in list(order[::-1][:k]) + list(order[:k]):  # most and least likely high grade
        row = d.iloc[i]
        p10 = explain.gwl10_map(row, cfg, out_dir / f"gwl10_{row['region_id']}.png",
                                title=f"{row['region_id']}  true {row['grade']}  P(HG)={p[i]:.2f}")
        if x["map_feature"] in d.columns:
            print(f"[explain] {row['region_id']}: map P10 {p10:.3f} vs extracted {row[x['map_feature']]:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--step", choices=STEPS + ["all"], default="all")
    parser.add_argument("--models", nargs="+", choices=list(NAMES), help="subset of models for train/evaluate")
    parser.add_argument("--resplit", action="store_true", help="split again even if regions.csv has a cohort column")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for step in STEPS if args.step == "all" else [args.step]:
        print(f"===== {step} =====")
        globals()[f"step_{step}"](cfg, args)


if __name__ == "__main__":
    main()
