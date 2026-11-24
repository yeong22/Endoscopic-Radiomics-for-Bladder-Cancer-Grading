# Notes on reproducibility

This repository provides research companion code for the published study. Some analysis procedures differ from the original analysis code used for the manuscript:

- The original code kept the cross-validation model with the best AUROC on the internal validation cohort. This code chooses models by cross-validation on the training cohort only, so the published internal results may be optimistic.
- The original code removed internal (654 → 584) and external validation regions using an inclusion flag that the paper does not describe. The final external prediction file has 356 regions; the paper reports 358.
- The original split was stratified by tumour grade only, and cross-validation was not stratified. This code follows the paper (age, sex and grade).
- The CIs in the abstract come from a region-level bootstrap and those in Table 2 from a patient-level bootstrap. The calibration CIs are Wald intervals.
- The evaluation code and the GWL10 map were re-implemented from the paper's Methods.
