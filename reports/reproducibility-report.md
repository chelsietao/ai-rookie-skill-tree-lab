# Reproducibility report

This file is generated from machine-readable metrics. Regenerate it after every full run.

| Lab | Dataset version | Seed | Seconds | RSS MB |
|---|---|---:|---:|---:|
| classical | portfolio-20260820-v2 | 20260820 | 9.601 | 571.0 |
| deep | portfolio-20260820-v2 | 20260820 | 17.472 | 934.6 |
| graph | portfolio-20260820-v2 | 20260820 | 0.979 | 447.1 |
| nlp | portfolio-20260820-v2 | 20260820 | 3.963 | 738.7 |
| tracking | portfolio-20260820-v2 | 20260820 | 0.019 | 627.2 |
| unsupervised | portfolio-20260820-v2 | 20260820 | 13.568 | 1052.3 |
| vision | portfolio-20260820-v2 | 20260820 | 92.843 | 1136.6 |

## Selected checks

- Best classical classifier: catboost
- Classical diagnostic: artifacts/figures/classical-diagnostics-portfolio.png
- Detector mAP@0.50: 1.0
- Detector mAP@0.50:0.95: 0.7988128933776056
- GCN node ROC-AUC: 0.7000661313310973
- Logistic + degree node baseline: 0.80043421818032
- GCN link ROC-AUC: 0.555253
- Degree-popularity link baseline: 0.5912685

Generated data and model binaries are intentionally ignored; configs, code, checksums,
metrics, and figures are versioned so the artifacts can be reconstructed.
