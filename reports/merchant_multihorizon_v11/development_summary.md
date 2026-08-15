# Sales Sentinel V11 — Multi-Horizon Prequential Verifier

- Status: **DEVELOPMENT_BEST**
- Frozen 7-day target reconstruction: **Exact**
- 3-day auxiliary: **{'model': 'xgb', 'topk': 64}**, mean fold AUC **69.15%**
- 14-day auxiliary: **{'model': 'xgb', 'topk': 128}**, mean fold AUC **65.19%**
- Meta candidates: **972**

- Precision: V9.2 **42.28%** -> V11 **42.62%**
- Recall: V9.2 **82.54%** -> V11 **82.54%**
- F1: V9.2 **55.91%** -> V11 **56.22%**
- NPV: V9.2 **95.74%** -> V11 **95.75%**
- Alert rate: V9.2 **32.28%** -> V11 **32.02%**
- TP/FP/FN/TN: **52/70/11/248**
- Worst-fold recall: **60.00%**
- Strictly dominates V9.2: **True**

Scientific boundary: development evidence only; fresh external Saudi merchant validation remains pending.
