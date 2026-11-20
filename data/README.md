# Input data

Nothing in `data/` except this file is tracked by git. Put your own data here, or point `paths.regions`
and `paths.image_root` in the config somewhere else.

```
data/
├── regions.csv
└── images/
    ├── <any sub-folders>/frame_001.dcm
    └── ...
```

## regions.csv

One row per tumour region. A frame can hold several regions, and a patient several frames.

| Column | Example | Meaning |
|---|---|---|
| `region_id` | `P001_1_1` | Unique id of the region |
| `patient_id` | `P001` | Pseudonymised patient id. All regions of a patient stay in the same cohort |
| `centre` | `1` | `1` = development centre (training + internal validation), `2` = external validation |
| `image_path` | `P001/frame_001.dcm` | Path relative to `paths.image_root`. DICOM (`.dcm`, RGB pixel data) or any image OpenCV reads |
| `polygon` | `412.0,233.5;430.2,240.1;...` | Tumour outline as `x,y` pixel coordinates separated by `;` (CVAT polygon format) |
| `grade` | `HG` | Pathological grade, `HG` or `LG` |
| `age` | `68` | Age in years |
| `sex` | `M` | `M` or `F` |
| `cohort` | `train` | Optional: `train`, `internal` or `external`. If present, `--step split` keeps it |

`python scripts/make_toy_data.py` writes a synthetic example in exactly this format to `data/toy/`.

## Before sharing anything

Do not commit images, DICOM files, tables or model files. DICOM headers and file names can carry
patient names and numbers, and fitted SVM models store training rows as support vectors.
