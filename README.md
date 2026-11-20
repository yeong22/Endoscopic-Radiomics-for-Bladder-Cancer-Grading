[European Urology Open Science 2026] A Novel Radiomics-based Interpretable Model for Bladder Cancer Grade Prediction Using White-Light Cystoscopy Images
=========================

📄 [Paper](https://doi.org/10.1016/j.euros.2026.03.018) &nbsp; 📑 Supplementary material: on the article page

Grading a bladder tumour before resection still relies on how it looks under white-light cystoscopy (WLC), which varies between urologists. We outline tumour regions on WLC frames, extract radiomic features from the grey, red, green and blue channels, keep **19** features with Coe-Thr-Lasso, and compare five classifiers for low-grade vs high-grade. Trained on 2,624 regions from 311 patients at one centre, the support vector machine reached an AUROC of **0.87** on internal validation (584 regions, 75 patients) and **0.79** on external validation at a second centre (358 regions, 37 patients). SHAP ranks the green-channel wavelet-L 10th percentile (GWL10) first; lower values push predictions towards high grade.

<p align="center">
  <img src="imgs/fig1_overview.jpg" width="85%"><br>
  <em>Figure 1 of the paper (CC BY 4.0). (A) Patient selection at two centres. (B) Segmentation, preprocessing, 4-channel feature extraction, Coe-Thr-Lasso, five classifiers, evaluation and SHAP.</em>
</p>

## Quick Start

Create an environment (`llvm-openmp` provides the OpenMP runtime that XGBoost and LightGBM need on macOS):
```bash
conda create -n bca-wlc -c conda-forge python=3.9 llvm-openmp -y
conda activate bca-wlc
pip install -r requirements.txt
```

Run the whole pipeline on synthetic data (no real data needed, about a minute):
```bash
python scripts/make_toy_data.py
python main.py --step all --config configs/toy.yaml
```
Results go to `outputs/toy/`. The toy images are random textures, so their numbers mean nothing; they only show that every step runs.

## Data

The study images are not distributed with this repository. They are retrospective WLC frames from Gangnam Severance Hospital (Centre 1, IRB 3-2023-0112) and Yonsei University Severance Hospital (Centre 2, IRB 1-2022-0052).

To run the code on your own data, put one table and the image files here:
```
data/
├── regions.csv      # one row per tumour region: ids, image path, polygon, grade, age, sex
└── images/          # DICOM (.dcm) or PNG/JPG frames, paths relative to this folder
```
Column list and format: [data/README.md](data/README.md).

## Usage

`python main.py --step <step> --config configs/default.yaml` runs one step; `--step all` runs them in order. Every setting is in [configs/default.yaml](configs/default.yaml).

| Step | Command | What it does | Paper |
|---|---|---|---|
| 1 | `python main.py --step split` | Patient-level 8:2 split of Centre 1 (stratified by grade, age group, sex); Centre 2 is the external cohort | 2.1 |
| 2 | `python main.py --step extract` | Grey/R/G/B channels, min-max normalisation → standardisation → CLAHE; PyRadiomics on original, LBP-2D, wavelet and LoG images (4,473 features) | 2.3 |
| 3 | `python main.py --step select` | The 19 published features, or Coe-Thr-Lasso again with `selection.use: computed` | 2.3, S1 |
| 4 | `python main.py --step train` | LR, SVM, RF, XGBoost, LightGBM; Optuna on patient-grouped stratified 5-fold CV of the training cohort; refit on the whole training cohort | 2.4, S2 |
| 5 | `python main.py --step evaluate` | AUROC/AUPRC with patient-level cluster bootstrap CIs, DeLong tests, calibration, decision curves | 2.5 |
| 6 | `python main.py --step explain` | SHAP for the SVM and GWL10 maps | 3.4 |

Useful switches in the config:
- `train.tune: false` trains with the hyperparameters of Supplementary Section S2 instead of searching (XGBoost has no published values, see the comment in the config).
- `selection.use: computed` reruns Coe-Thr-Lasso on your training cohort.
- `--models svm lr` limits `train` and `evaluate` to some models.

Outputs (`outputs/<run_name>/`): `regions_split.csv`, `features.csv`, `selected_features.txt`, `models/*.joblib` (model, scaler, feature list and constructor arguments), `results/` (metrics, DeLong, calibration, DCA tables and plots), `explain/` (SHAP plots, GWL10 maps).

## Main Results

Numbers reported in the paper (Table 2), tumour-region level, 95% CI from a patient-level cluster bootstrap (2,000 resamples).

| Model | Internal AUROC | Internal AUPRC | External AUROC | External AUPRC |
|---|---|---|---|---|
| **SVM** | **0.87** (0.81–0.91) | **0.89** (0.81–0.95) | **0.79** (0.71–0.88) | **0.90** (0.80–0.96) |
| LR | 0.86 (0.80–0.90) | 0.89 (0.81–0.94) | 0.79 (0.71–0.88) | 0.89 (0.79–0.96) |
| XGBoost | 0.85 (0.78–0.89) | 0.87 (0.78–0.93) | 0.73 (0.64–0.85) | 0.85 (0.73–0.93) |
| LightGBM | 0.83 (0.76–0.88) | 0.85 (0.76–0.91) | 0.58 (0.58–0.71) | 0.79 (0.67–0.93) |
| RF | 0.84 (0.77–0.89) | 0.85 (0.78–0.93) | 0.68 (0.63–0.77) | 0.85 (0.73–0.93) |

Calibration and decision curve of the SVM on the external cohort (Figure 3): calibration intercept 0.43 (0.10–0.76), slope 0.93 (0.67–1.18); net benefit above treat-all and treat-none from a threshold probability of about 0.30.

<p align="center">
  <img src="imgs/fig3_calibration_dca.jpg" width="55%"><br>
  <em>Figure 3 of the paper (CC BY 4.0). (A) LOESS calibration curve with bootstrap band. (B) Decision curve.</em>
</p>

SHAP values of the SVM (Figure 2):

<p align="center">
  <img src="imgs/fig2_shap.jpg" width="85%"><br>
  <em>Figure 2 of the paper (CC BY 4.0). (A) SHAP summary plot. (B) Mean |SHAP| per feature.</em>
</p>

## Feature Selection Comparison

Supplementary Table 1 (internal validation): Coe-Thr-Lasso against other selection methods.

| Method | Features | LR AUROC | LR AUPRC | RF AUROC | RF AUPRC | SVM AUROC | SVM AUPRC |
|---|---|---|---|---|---|---|---|
| Lasso | 21 | 0.84 | 0.85 | 0.83 | 0.86 | 0.84 | 0.87 |
| RFECV | 21 | 0.83 | 0.85 | 0.80 | 0.84 | 0.85 | 0.88 |
| RF importance | 12 | 0.82 | 0.84 | 0.80 | 0.84 | 0.81 | 0.84 |
| **Coe-Thr-Lasso** | **19** | **0.86** | **0.89** | **0.81** | **0.85** | **0.87** | **0.89** |

## Reproducibility Notes

- The evaluation code (cluster bootstrap, calibration, decision curves, DeLong) and the GWL10 map were **re-implemented from the paper's Methods**; the original analysis code of the final revision is not available. Run on the saved predictions of the published SVM, the evaluation code returns AUROC 0.87 / 0.79, AUPRC 0.89 / 0.90 (internal / external), calibration intercept 0.43 (0.10–0.75) and slope 0.93 (0.67–1.18).
- The calibration intercept and slope come from the logistic recalibration model logit(P) = a + b·logit(p) with Wald CIs, which reproduces the reported values. `evaluate.calibration_ci: bootstrap` gives patient-level bootstrap CIs instead.
- PyRadiomics: the paper used 3.0.1, which has no Python 3.9 wheels; this repository pins 3.1.0.
- Preprocessing keeps the original cast of the standardised image to 8 bits, where values outside 0–255 wrap around (`preprocess.uint8_cast: wrap`); `clip` saturates instead.
- Re-running the code on the original data will not give exactly the published numbers (library versions, random seeds, model refitting).

## Repository Structure
```
bca-wlc-radiomics/
├── main.py                   # one entry point: --step split|extract|select|train|evaluate|explain|all
├── configs/
│   ├── default.yaml          # paper settings (every parameter lives here)
│   └── toy.yaml              # small settings for the synthetic demo
├── src/bcarad/
│   ├── data.py               # region table, DICOM/PNG loading, polygon masks, cohort split
│   ├── preprocess.py         # channels; normalise -> standardise -> CLAHE
│   ├── features.py           # PyRadiomics extraction
│   ├── selection.py          # Coe-Thr-Lasso
│   ├── models.py             # five classifiers and their search spaces
│   ├── train.py              # Optuna + patient-grouped CV, refit
│   ├── evaluate.py           # bootstrap CIs, DeLong, calibration, decision curves
│   └── explain.py            # SHAP, GWL10 map
├── scripts/make_toy_data.py  # synthetic data in the real layout
├── stats/table1.R            # Table 1 (patient characteristics)
├── data/README.md            # input format
└── imgs/                     # figures from the paper
```

## Citation
```bibtex
@article{choi2026radiomics,
  title   = {A Novel Radiomics-based Interpretable Model for Bladder Cancer Grade Prediction Using White-Light Cystoscopy Images},
  author  = {Choi, Yewon and Cho, Sang Wouk and Hong, Junho and Lee, Jongsoo and Kim, Hwiyoung and Lee, Kwang Suk},
  journal = {European Urology Open Science},
  year    = {2026},
  volume  = {87},
  pages   = {71--79},
  doi     = {10.1016/j.euros.2026.03.018}
}
```

## Contact
Yewon Choi (first author). Please open a GitHub issue for questions about the code.

## License
Figures in `imgs/` are taken from the paper, published under CC BY 4.0 (© 2026 The Authors).
