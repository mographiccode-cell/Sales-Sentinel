# Sales Sentinel V8.2 — Frozen Development Candidate

- Status: **DEVELOPMENT_FROZEN_PENDING_EXTERNAL_SAUDI_VALIDATION**
- Ranking model: **XGBoost depth 2 / top 128 features**
- Alert verifier: **Logistic Regression / 14 core meta-features**

## Primary causal prequential evidence
- Accuracy: **77.43%**
- Balanced accuracy: **79.48%**
- Precision: **40.94%**
- Recall: **82.54%**
- F1: **54.74%**
- GREEN NPV: **95.67%**
- Alert rate: **33.33%**
- TP/FP/FN/TN: **52/75/11/243**
- Worst-fold recall: **60.00%**

- FP improvement vs V7.5: **77 -> 75**
- External Saudi merchant validation: **Pending**
- RED supported: **False**

Important: do not report full-development fitted performance as validation; use the prequential metrics above.
