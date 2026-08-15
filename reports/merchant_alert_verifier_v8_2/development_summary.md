# Sales Sentinel V8.2 — Prequential Alert Verifier

- Status: **DEVELOPMENT_BEST**
- Candidates: **1152**
- Selected: **{'features': 'core', 'C': 0.1, 'tp_quantile': 0.1, 'margin': 0.05, 'scope': 'nonstrong', 'rank_guard': 0.4, 'rescue': 1.01}**

- Precision: V7.5 **40.31%** -> V8.2 **40.94%**
- Recall: V7.5 **82.54%** -> V8.2 **82.54%**
- F1: V7.5 **54.17%** -> V8.2 **54.74%**
- NPV: V7.5 **95.63%** -> V8.2 **95.67%**
- Alert rate: V7.5 **33.86%** -> V8.2 **33.33%**
- TP/FP/FN/TN: **52/75/11/243**
- Worst-fold recall: **60.00%**
- Strictly dominates V7.5: **True**

Scientific boundary: development-selected causal verifier; external validation remains pending.
