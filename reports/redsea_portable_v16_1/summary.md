# Sales Sentinel V16.1 — Comparability Ablation

- Status: **POST_OPEN_EXTERNAL_DIAGNOSTIC_COMPARABILITY_ABLATION**
- Features: **116 → 96** (removed 20 semantically non-comparable features)
- Selected model: **extra_trees**
- Frozen nested threshold: **0.242**

## Development nested OOF
- ROC-AUC / PR-AUC: **69.55% / 29.00%**
- Precision / Recall / F1: **24.12% / 65.08% / 35.19%**
- NPV / Alert rate: **89.57% / 44.62%**
- Δ AUC / Δ F1 vs V16: **+3.59% / -2.37%**

## Redsea post-open diagnostic
- ROC-AUC / PR-AUC: **65.72% / 66.79%**
- Precision / Recall / F1: **64.44% / 85.29% / 73.42%**
- Accuracy / Balanced Accuracy: **65.00% / 61.88%**
- NPV / Alert rate: **66.67% / 75.00%**
- TP/FP/FN/TN: **29/16/5/10**
- Δ external AUC / Recall / Precision vs V16: **-1.02% / +32.35% / -13.82%**

## Drift after comparability ablation
- Median |SMD|: **0.150**
- Max |SMD|: **1.120**
- Features |SMD|>=1: **1**

Scientific note: this is a post-open diagnostic, not a new blind validation. The ablation does not use Redsea labels to choose the model, threshold, or removed features.
