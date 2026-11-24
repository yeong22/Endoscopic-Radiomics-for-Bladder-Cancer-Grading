"""Model interpretation (paper Section 3.4, Figure 2, Supplementary Figure 1).

- SHAP values of the SVM (KernelExplainer on predict_proba, logit link), beeswarm and bar plots.
- GWL10 map: green channel, normalisation, standardisation, CLAHE and wavelet-L filtering. Pixels at or
  below the 10th percentile inside the tumour contour are drawn as red dots.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import SimpleITK as sitk
from radiomics import featureextractor, imageoperations

from .data import load_rgb, polygon_to_mask
from .preprocess import normalize_standardize_clahe


def shap_values(ckpt, df, cfg):
    x = cfg["explain"]
    X = pd.DataFrame(ckpt["scaler"].transform(df[ckpt["features"]]), columns=ckpt["features"], index=df.index)
    background = X.sample(min(x["background_size"], len(X)), random_state=cfg["seed"])
    explained = X if not x["max_explain"] else X.sample(min(x["max_explain"], len(X)), random_state=cfg["seed"])
    explainer = shap.KernelExplainer(ckpt["model"].predict_proba, background, link="logit")
    values = explainer.shap_values(explained, nsamples=x["nsamples"], silent=True)
    values = values[1] if isinstance(values, list) else values[..., 1]  # SHAP for P(HG)
    return pd.DataFrame(values, columns=ckpt["features"], index=explained.index), explained


def plot_shap(values, X, out_dir):
    for kind, name in [("dot", "shap_beeswarm.png"), ("bar", "shap_bar.png")]:
        shap.summary_plot(values.to_numpy(), X, plot_type=kind, show=False, max_display=len(X.columns))
        plt.gcf().set_size_inches(9, 0.35 * len(X.columns) + 1.5)
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=200)
        plt.close()


def wavelet_l(image, mask, settings):
    """Wavelet-L image, tumour mask and crop origin, on the same crop as the feature extraction."""
    settings = featureextractor.RadiomicsFeatureExtractor(**settings).settings  # with the PyRadiomics defaults
    img, msk = imageoperations.resampleImage(sitk.GetImageFromArray(image), sitk.GetImageFromArray(mask), **settings)
    wavelet = {name: im for im, name, _ in imageoperations.getWaveletImage(img, msk, **settings)}
    x0, y0 = (round(v) for v in img.GetOrigin())
    return sitk.GetArrayFromImage(wavelet["wavelet-L"]), sitk.GetArrayFromImage(msk).astype(bool), (x0, y0)


def gwl10_map(row, cfg, path, title=""):
    x = cfg["explain"]
    rgb = load_rgb(cfg["paths"]["image_root"] / row["image_path"])
    mask = polygon_to_mask(row["polygon"], rgb.shape[:2])
    steps = normalize_standardize_clahe(rgb[..., 1], cfg["preprocess"], return_steps=True)

    wl, roi, (x0, y0) = wavelet_l(steps["clahe"], mask, cfg["radiomics"]["settings"])
    p10 = np.percentile(wl[roi], x["map_percentile"])
    low = roi & (wl <= p10)
    h, w = wl.shape
    crop = rgb[y0:y0 + h, x0:x0 + w]

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.4))
    panels = [(rgb, None, "A. Original"), (steps["normalised"], "gray", "B. Green, normalised"),
              (steps["standardised"], "gray", "C. Standardised"), (steps["clahe"], "gray", "D. CLAHE"),
              (np.where(roi, wl, np.nan), "viridis", "E. Wavelet-L (ROI)"), (crop, None, f"F. <= P{x['map_percentile']}")]
    for ax, (im, cmap, name) in zip(axes, panels):
        ax.imshow(im, cmap=cmap)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    axes[0].contour(mask, levels=[0.5], colors="yellow", linewidths=0.8)
    axes[5].contour(roi, levels=[0.5], colors="yellow", linewidths=0.8)
    ys, xs = np.nonzero(low)
    axes[5].scatter(xs, ys, s=1.5, c="red")
    fig.suptitle(f"{title}  {x['map_feature']} = {p10:.1f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return p10
