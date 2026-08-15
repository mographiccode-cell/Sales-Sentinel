# Sales Sentinel V13.1 — Frozen Development Candidate

- Status: **DEVELOPMENT_FROZEN_PENDING_FRESH_EXTERNAL_VALIDATION**
- Precision / Recall / F1: **44.44% / 82.54% / 57.78%**
- Accuracy / Balanced Accuracy: **80.05% / 81.05%**
- GREEN NPV / Alert rate: **95.83% / 30.71%**
- TP/FP/FN/TN: **52/65/11/253**
- Worst-fold recall: **60.00%**

## Incremental robustness
- Folds with FP improvement vs V11: **1/5**
- Folds with TP degradation: **0/5**
- Neighbor configs passing V11 contract: **189/243 (77.8%)**
- Neighbor configs matching/exceeding TP52 + FP<=65: **108/243 (44.4%)**
- Same selected config without calendar TP/FP: **52/69**
- With calendar TP/FP: **52/65**
- Alerts changed vs V11: **5**, all true-negative removals: **True**

Important: the incremental V13.1 gain is concentrated in one fold. Freeze this version and require fresh longitudinal Saudi merchant data or a new untouched future period before claiming further generalization.
