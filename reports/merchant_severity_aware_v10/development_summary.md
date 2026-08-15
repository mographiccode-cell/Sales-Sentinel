# Sales Sentinel V10 — Severity-Aware Decline Model

- Status: **EXPERIMENTAL_V9_2_REMAINS_BEST**
- Candidates: **36**
- Selected: **{'model': 'xgb', 'topk': 96, 'boundary': 0.78, 'growth_split': None}**

- Severity ROC-AUC / PR-AUC: **79.90% / 45.73%**
- Precision: V9.2 **42.28%** -> V10 **42.28%**
- Recall: V9.2 **82.54%** -> V10 **82.54%**
- F1: V9.2 **55.91%** -> V10 **55.91%**
- NPV: V9.2 **95.74%** -> V10 **95.74%**
- Alert rate: V9.2 **32.28%** -> V10 **32.28%**
- TP/FP/FN/TN: **52/71/11/247**
- Worst-fold recall: **60.00%**
- Strictly dominates V9.2: **False**

Scientific boundary: development evidence only; external Saudi merchant validation remains pending.
