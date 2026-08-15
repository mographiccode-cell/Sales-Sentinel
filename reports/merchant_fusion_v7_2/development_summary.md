# Sales Sentinel V7.2 — V6.1 + V7.1 Ranking Fusion

- Selection mode: **best_experimental_candidate_contract_failed**
- Development rows: **381**
- Positives: **63**
- V7.1 category-ensemble ROC-AUC: **78.19%**
- V7.1 category-ensemble PR-AUC: **43.26%**
- Best rank-blend ROC-AUC: **78.90%**
- Best rank-blend PR-AUC: **47.60%**

## Operational comparison
- V6.1 precision: **38.81%** -> V7.2 **38.81%**
- V6.1 recall: **82.54%** -> V7.2 **82.54%**
- V6.1 F1: **52.79%** -> V7.2 **52.79%**
- V6.1 GREEN NPV: **95.55%** -> V7.2 **95.55%**
- V6.1 alert rate: **35.17%** -> V7.2 **35.17%**
- V6.1 TP/FP/FN/TN: **52/82/11/236**
- V7.2 TP/FP/FN/TN: **52/82/11/236**
- FP removed: **0**
- TP change: **+0**
- Worst-fold recall: **60.00%**
- Max-fold alert rate: **40.48%**
- Vetoed alerts: **0**
- Rescued alerts: **0**
- Robust perturbations passing: **0/81**
- Development gates passed: **False**
- Adopt over V6.1: **False**
- RED supported: **False**

Scientific boundary: this is development fusion on previously used OOF evidence, not a new blind or external validation set.
