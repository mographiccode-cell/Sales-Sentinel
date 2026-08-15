# Sales Sentinel V17 — Causal Weekly Merchant Adaptation

- Status: **POST_OPEN_CAUSAL_ADAPTATION_DIAGNOSTIC**
- Evaluation after enough past labels: **40 rows (2023-09-15 → 2023-10-24)**
- Evaluation decline prevalence: **45.00%**
- Target labels available with: **7-day delay**
- Adaptation target weight: **4.0x**

## V17 adapted
- ROC-AUC / PR-AUC: **64.65% / 55.05%**
- Precision / Recall / F1: **47.83% / 61.11% / 53.66%**
- Balanced Accuracy / NPV / Alert: **53.28% / 58.82% / 57.50%**
- TP/FP/FN/TN: **11/12/7/10**

## Same-date controls
- V16.1 static AUC / P / R / F1 / Alert: **71.97% / 54.84% / 94.44% / 69.39% / 77.50%**
- V16.2 calibrated P / R / F1 / Alert: **63.16% / 66.67% / 64.86% / 47.50%**
- Δ V17 vs static AUC / Recall / F1 / Alert: **-7.32% / -33.33% / -15.73% / -20.00%**

Scientific note: this is a post-open prequential adaptation experiment, not fresh external validation. Its value is testing whether a real deployed merchant can improve after accumulating its own labeled history without future leakage.
