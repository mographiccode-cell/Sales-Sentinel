# Sales Sentinel V9 — Continuous Future-Ratio Verifier

- Status: **DEVELOPMENT_BEST**
- Candidates: **30**
- Selected: **{'model': 'cb_rmse', 'topk': 96, 'target_mode': 'log_ratio'}**

- Ratio MAE / RMSE: **0.1045 / 0.1359**
- Regression-derived ROC-AUC / PR-AUC: **80.37% / 45.80%**

- Precision: V8.2 **40.94%** -> V9 **41.94%**
- Recall: V8.2 **82.54%** -> V9 **82.54%**
- F1: V8.2 **54.74%** -> V9 **55.61%**
- NPV: V8.2 **95.67%** -> V9 **95.72%**
- Alert rate: V8.2 **33.33%** -> V9 **32.55%**
- TP/FP/FN/TN: **52/72/11/246**
- Worst-fold recall: **60.00%**
- Strictly dominates V8.2: **True**

Scientific boundary: development evidence only; external real Saudi merchant validation is still pending.
