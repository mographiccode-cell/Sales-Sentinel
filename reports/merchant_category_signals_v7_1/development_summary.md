# Sales Sentinel V7.1 — Merchant + Category Signals (Nested Rolling-Origin)

- Merchant rows: **541**
- Positive rate: **14.60%**
- Merchant-only features: **137**
- Added category-regime signals: **137**
- Selected scope: **merchant_only**
- Selected model: **hist_gb**
- Deployment threshold (median inner thresholds): **0.120**
- Nested OOF ROC-AUC: **76.29%**
- Nested OOF PR-AUC: **37.97%**
- Balanced Accuracy: **72.04%**
- Precision: **34.09%**
- Recall: **71.43%**
- F1: **46.15%**
- GREEN NPV: **92.77%**
- Alert rate: **34.65%**
- TP / FP / FN / TN: **45 / 87 / 18 / 231**
- Worst-fold recall: **28.57%**
- Max-fold alert rate: **52.38%**
- Category-signal AUC delta vs merchant-only: **+0.64%**
- Development gates passed: **False**
- Independent real Saudi merchant validation: **Pending**

Scientific boundary: V7.1 uses nested rolling-origin development evaluation. The previously opened V7.0 blind window is not reused or described as independent evidence.
