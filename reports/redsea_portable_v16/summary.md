# Sales Sentinel V16 — Portable V7.1 → Redsea Diagnostic

- Status: **POST_OPEN_EXTERNAL_DIAGNOSTIC**
- Portable V7.1 features: **116**
- Selected model from nested development only: **hist_gb**
- Frozen median-inner threshold: **0.220**

## Nested development
- Rows / decline prevalence: **541 / 14.60%**
- ROC-AUC / PR-AUC: **65.96% / 31.54%**
- Precision / Recall / F1: **27.61% / 58.73% / 37.56%**
- NPV / Alert rate: **89.47% / 35.17%**

## Redsea post-open external diagnostic
- Eligible rows / decline prevalence: **60 / 56.67%**
- ROC-AUC / PR-AUC: **66.74% / 67.49%**
- Precision / Recall / F1: **78.26% / 52.94% / 63.16%**
- Accuracy / Balanced Accuracy: **65.00% / 66.86%**
- NPV / Alert rate: **56.76% / 38.33%**
- TP/FP/FN/TN: **18/5/16/21**

## Domain drift
- Median |SMD|: **0.157**
- Max |SMD|: **2.198**
- Features with |SMD| >= 1: **6**

Scientific note: V16 is not called blind because Redsea outcomes were already opened in V15. Its purpose is to diagnose whether V7.1's portable merchant feature family transfers better than the weak generic V15 feature set without using Redsea labels for model selection.
