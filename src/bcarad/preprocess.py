"""Channel decomposition and preprocessing: min-max normalisation -> standardisation -> CLAHE (paper Section 2.3)."""
import cv2
import numpy as np


def to_uint8(x, mode):
    """Cast float pixels to uint8 like the original code.

    The original code did (z * 255).astype(np.uint8). Out-of-range values wrap modulo 256 there
    (checked on x86-64 and arm64); "wrap" reproduces that explicitly, "clip" saturates to 0..255.
    """
    if mode == "wrap":
        return np.mod(np.trunc(x).astype(np.int64), 256).astype(np.uint8)
    if mode == "clip":
        return np.clip(x, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown uint8_cast: {mode}")


def normalize_standardize_clahe(channel, pcfg, return_steps=False):
    channel = np.ascontiguousarray(channel)
    x = cv2.normalize(channel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    std = np.std(x)
    z = (x - np.mean(x)) / std if std > 0 else np.zeros_like(x)
    u8 = to_uint8(z * 255, pcfg["uint8_cast"])
    clahe = cv2.createCLAHE(clipLimit=pcfg["clahe_clip_limit"], tileGridSize=tuple(pcfg["clahe_tile_grid"]))
    out = clahe.apply(u8)
    return {"normalised": x, "standardised": z, "clahe": out} if return_steps else out


def split_channels(rgb):
    return {"gray": cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), "r": rgb[..., 0], "g": rgb[..., 1], "b": rgb[..., 2]}


def channel_images(rgb, pcfg):
    """{channel name: preprocessed uint8 image} for the channels listed in the config."""
    raw = split_channels(rgb)
    return {c: normalize_standardize_clahe(raw[c], pcfg) for c in pcfg["channels"]}
