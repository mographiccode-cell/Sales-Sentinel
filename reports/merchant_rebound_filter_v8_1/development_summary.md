# Sales Sentinel V8.1 — Prequential Rebound Filter

- Status: **EXPERIMENTAL_V7_5_REMAINS_BEST**
- Precision: V7.5 **40.31%** -> V8.1 **40.31%**
- Recall: V7.5 **82.54%** -> V8.1 **82.54%**
- F1: V7.5 **54.17%** -> V8.1 **54.17%**
- NPV: V7.5 **95.63%** -> V8.1 **95.63%**
- Alert rate: V7.5 **33.86%** -> V8.1 **33.86%**
- TP/FP/FN/TN: **52/77/11/241**
- Worst-fold recall: **60.00%**
- Adopt over V7.5: **False**

- Development oracle strictly dominates V7.5: **True**
- Oracle rule: **['market', 0.4, 'market', 0.8]**
- Oracle TP/FP/FN/TN: **52/63/11/255**

Scientific boundary: prequential development evidence only; oracle metrics are not deployment evidence.
