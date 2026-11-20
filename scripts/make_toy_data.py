"""Make a small SYNTHETIC dataset with the same layout as the real one (no patient data).

    python scripts/make_toy_data.py            # -> data/toy/regions.csv + data/toy/images/*.png

Images are random pink "bladder wall" textures with elliptical "tumours". High-grade tumours are
more often darker and coarser in the green channel, so the pipeline has a weak signal to learn.
All numbers (patients, ages, prevalence) are arbitrary and unrelated to the study cohort.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def smooth_noise(rng, size, sigma):
    n = cv2.GaussianBlur(rng.normal(size=(size, size)), (0, 0), sigma)
    return n / (np.abs(n).max() + 1e-8)


def ellipse_polygon(rng, cx, cy, rx, ry, n=16):
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = 1 + rng.normal(0, 0.06, n)
    return [(cx + rx * k * np.cos(a), cy + ry * k * np.sin(a)) for a, k in zip(angles, r)]


def draw_image(rng, size, tumours):
    img = np.empty((size, size, 3))
    light = 0.85 + 0.15 * smooth_noise(rng, size, 25)
    for c, base in enumerate([190, 120, 110]):
        img[..., c] = base * light + 6 * smooth_noise(rng, size, 2)
    for poly, high_grade in tumours:
        mask = np.zeros((size, size), np.uint8)
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
        coarse_texture = rng.random() < (0.7 if high_grade else 0.35)  # grade only shifts the odds
        shift = rng.normal(-14 if high_grade else -10, 8)
        texture = 13 * smooth_noise(rng, size, 3.0 if coarse_texture else 1.0)
        fine = smooth_noise(rng, size, 1.0)
        img[..., 0] += mask * (10 + 10 * fine)
        img[..., 1] += mask * (shift + texture)
        img[..., 2] += mask * (shift / 2 + 0.5 * texture)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "data" / "toy"))
    parser.add_argument("--patients", type=int, nargs=2, default=[40, 12], help="Centre 1 and Centre 2")
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    rows, pid = [], 0
    for centre, n_patients in enumerate(args.patients, start=1):
        for _ in range(n_patients):
            pid += 1
            patient = f"P{pid:03d}"
            grade = "HG" if rng.random() < 0.5 else "LG"
            age, sex = int(rng.integers(40, 90)), "M" if rng.random() < 0.5 else "F"
            for k in range(int(rng.integers(1, 3))):  # 1-2 images per patient
                spots = [(0.3, 0.35), (0.7, 0.65)][: int(rng.integers(1, 3))]  # 1-2 tumours per image
                polys = [ellipse_polygon(rng, fx * args.size, fy * args.size, rng.uniform(14, 24), rng.uniform(12, 22))
                         for fx, fy in spots]
                name = f"{patient}_{k + 1}.png"
                cv2.imwrite(str(out / "images" / name),
                            cv2.cvtColor(draw_image(rng, args.size, [(p, grade == "HG") for p in polys]), cv2.COLOR_RGB2BGR))
                for j, poly in enumerate(polys):
                    rows.append({"region_id": f"{patient}_{k + 1}_{j + 1}", "patient_id": patient, "centre": centre,
                                 "image_path": name, "polygon": ";".join(f"{x:.1f},{y:.1f}" for x, y in poly),
                                 "grade": grade, "age": age, "sex": sex})
    df = pd.DataFrame(rows)
    df.to_csv(out / "regions.csv", index=False)
    print(f"{len(df)} regions from {df['patient_id'].nunique()} synthetic patients -> {out / 'regions.csv'}")


if __name__ == "__main__":
    main()
