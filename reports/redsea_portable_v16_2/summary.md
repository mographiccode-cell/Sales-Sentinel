# Sales Sentinel V16.2 — Causal Percentile Calibration

- Status: **POST_OPEN_UNSUPERVISED_CALIBRATION_DIAGNOSTIC**
- Alert budget inherited from development: **44.62%**
- Past-only percentile cutoff: **55.38%**
- Development-selected lookback / warmup: **112 / 7**

## Development policy
- Precision / Recall / F1: **25.39% / 77.78% / 38.28%**
- Balanced Accuracy / NPV / Alert: **66.25% / 92.55% / 50.66%**

## Redsea post-open diagnostic
- ROC-AUC / PR-AUC (ranking unchanged): **65.72% / 66.79%**
- Precision / Recall / F1: **70.97% / 64.71% / 67.69%**
- Accuracy / Balanced Accuracy: **65.00% / 65.05%**
- NPV / Alert rate: **58.62% / 51.67%**
- TP/FP/FN/TN: **22/9/12/17**
- Δ Alert / Precision / Recall / F1 vs V16.1: **-23.33% / +6.52% / -20.59% / -5.73%**

Scientific note: percentile calibration is causal and label-free at inference; nevertheless Redsea is already post-open and these external numbers remain diagnostic evidence only.
