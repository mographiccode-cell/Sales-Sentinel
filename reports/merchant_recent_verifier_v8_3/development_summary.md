# Sales Sentinel V8.3 — Recent-Window Alert Verifier

- Status: **DEVELOPMENT_BEST**
- Candidates: **3240**
- Selected: **{'history_folds': 2, 'C': 0.05, 'tp_quantile': 0.2, 'margin': 0.02, 'scope': 'nonstrong', 'rank_guard': 0.4, 'rescue': 1.01}**

- Precision: V7.5 **40.31%** -> V8.3 **40.94%**
- Recall: V7.5 **82.54%** -> V8.3 **82.54%**
- F1: V7.5 **54.17%** -> V8.3 **54.74%**
- NPV: V7.5 **95.63%** -> V8.3 **95.67%**
- Alert rate: **33.33%**
- TP/FP/FN/TN: **52/75/11/243**
- Worst-fold recall: **60.00%**
- Strictly dominates V7.5: **True**

Scientific boundary: causal within folds, development-selected configuration; external validation pending.
