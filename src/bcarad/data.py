"""Region table, image loading, polygon masks and the patient-level cohort split."""
import warnings

import cv2
import numpy as np
import pandas as pd
import pydicom
from skimage import draw
from sklearn.model_selection import StratifiedGroupKFold

REQUIRED = ["region_id", "patient_id", "centre", "image_path", "polygon", "grade", "age", "sex"]


def read_regions(cfg):
    df = pd.read_csv(cfg["paths"]["regions"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"regions.csv is missing columns: {missing} (see data/README.md)")
    df["y"] = (df["grade"] == cfg["labels"]["positive"]).astype(int)
    return df


def load_rgb(path):
    """Read a cystoscopy frame as an RGB uint8 array (DICOM or an ordinary image file)."""
    path = str(path)
    if path.lower().endswith((".dcm", ".dicom")):
        return pydicom.dcmread(path).pixel_array
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def polygon_to_mask(polygon, shape):
    """Binary mask of shape (rows, cols) from a polygon string 'x1,y1;x2,y2;...' in pixel coordinates."""
    pts = [tuple(map(float, p.split(","))) for p in polygon.split(";")]
    rr, cc = draw.polygon([p[1] for p in pts], [p[0] for p in pts], shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rr, cc] = 1
    return mask


def split_cohorts(df, cfg):
    """Assign cohort = train / internal / external.

    Centre 2 is the external cohort. Centre 1 patients are split 8:2 at the patient level with
    StratifiedGroupKFold (one fold held out), stratified on the label built from cfg["split"]["stratify_on"].
    """
    s = cfg["split"]
    df = df.copy()
    df["cohort"] = np.where(df["centre"] == 1, "train", "external")
    c1 = df[df["centre"] == 1]
    strata = []
    for col in s["stratify_on"]:
        if col == "age_group":
            strata.append((c1["age"] // s["age_bin_years"]).astype(int).astype(str))
        else:
            strata.append(c1[col].astype(str))
    label = strata[0].str.cat(strata[1:], sep="_") if len(strata) > 1 else strata[0]
    n_splits = round(1 / s["test_size"])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=s["random_state"])
    with warnings.catch_warnings():  # strata with fewer members than folds
        warnings.simplefilter("ignore", UserWarning)
        _, test_idx = next(sgkf.split(c1, label, groups=c1["patient_id"]))
    df.loc[c1.index[test_idx], "cohort"] = "internal"
    return df
