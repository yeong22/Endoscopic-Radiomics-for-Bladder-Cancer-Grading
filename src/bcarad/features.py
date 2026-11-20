"""PyRadiomics feature extraction for every tumour region (paper Section 2.3, Supplementary Section S1).

Per region: 4 channels (gray, r, g, b) x 4 image types (original, LBP2D, wavelet, LoG) x 6 feature
classes, plus 9 shape2D features from the greyscale image. Column names are "<channel>_<PyRadiomics name>",
e.g. g_wavelet-L_firstorder_10Percentile.
"""
import logging
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import radiomics
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm

from .data import load_rgb, polygon_to_mask
from .preprocess import channel_images

radiomics.setVerbosity(logging.ERROR)
_EXTRACTORS = {}


def build_extractors(rcfg):
    texture = featureextractor.RadiomicsFeatureExtractor(**rcfg["settings"])
    texture.disableAllFeatures()
    for cls in rcfg["feature_classes"]:
        texture.enableFeatureClassByName(cls)
    texture.disableAllImageTypes()
    for image_type in rcfg["image_types"]:
        texture.enableImageTypeByName(image_type)

    shape = featureextractor.RadiomicsFeatureExtractor(**rcfg["settings"])
    shape.disableAllFeatures()
    shape.enableFeatureClassByName("shape2D")
    shape.disableAllImageTypes()
    shape.enableImageTypeByName("Original")
    return {"texture": texture, "shape": shape}


def _init_worker(rcfg):
    _EXTRACTORS.update(build_extractors(rcfg))


def _values(vector, prefix):
    return {f"{prefix}_{k}": float(v) for k, v in vector.items() if not k.startswith("diagnostics")}


def extract_region(job):
    """job = (region_id, image file, polygon, cfg). Returns a dict of features, or an error string."""
    region_id, image_file, polygon, cfg = job
    try:
        rgb = load_rgb(image_file)
        mask = sitk.GetImageFromArray(polygon_to_mask(polygon, rgb.shape[:2]))
        images = channel_images(rgb, cfg["preprocess"])
        row = {"region_id": region_id}
        for channel, img in images.items():
            row.update(_values(_EXTRACTORS["texture"].execute(sitk.GetImageFromArray(img), mask), channel))
        if cfg["radiomics"]["shape2d_on_gray"]:
            row.update(_values(_EXTRACTORS["shape"].execute(sitk.GetImageFromArray(images["gray"]), mask), "gray"))
        return row
    except Exception as e:  # keep going like the original code, but report every failure
        return f"{region_id}: {type(e).__name__}: {e}"


def extract_all(regions, cfg):
    root = cfg["paths"]["image_root"]
    jobs = [(r.region_id, root / r.image_path, r.polygon, cfg) for r in regions.itertuples()]
    with ProcessPoolExecutor(max_workers=cfg["radiomics"]["n_jobs"], initializer=_init_worker,
                             initargs=(cfg["radiomics"],)) as pool:
        results = list(tqdm(pool.map(extract_region, jobs, chunksize=4), total=len(jobs), desc="radiomics"))
    rows = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, str)]
    return pd.DataFrame(rows), errors
