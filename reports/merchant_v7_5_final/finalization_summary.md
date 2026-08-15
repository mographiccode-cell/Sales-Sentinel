# Sales Sentinel V7.5 — Frozen Development Candidate

- Status: **DEVELOPMENT_FROZEN_PENDING_EXTERNAL_SAUDI_VALIDATION**
- Model: **catboost / top96 / weighted=True**
- Final features: **96**

## Primary prequential evidence
- Precision: **40.31%**
- Recall: **82.54%**
- F1: **54.17%**
- GREEN NPV: **95.63%**
- Alert rate: **33.86%**
- TP/FP/FN/TN: **52/77/11/241**
- Worst-fold recall: **60.00%**

- Final development correction rule: **['market', 0.5, 'market', 0.85]**
- Threshold-neighbor robustness: **22/25 pass**
- Score-noise sigma=0.01 contract pass rate: **100%**
- Score-noise sigma=0.02 contract pass rate: **100%**
- External Saudi merchant validation: **Pending**
- RED supported: **False**

Important: final-rule tuning metrics are not independent evidence; use the prequential metrics above for academic reporting.
