# Sales Sentinel V15 — Frozen Generic Transfer to Redsea

- Status: **FROZEN_TRANSFER_EVALUATED**
- Generic scale-free features: **67**
- Development model: **logistic_c0.1**
- Frozen development threshold: **0.080**

## Development-only OOF
- Rows / decline prevalence: **350 / 13.86%**
- ROC-AUC / PR-AUC: **50.70% / 23.54%**
- Precision / Recall / F1: **17.87% / 57.81% / 27.31%**
- TP/FP/FN/TN: **37/170/27/116**

## Independent real Saudi merchant: Redsea
- Eligible dates / decline prevalence: **85 / 52.94%**
- ROC-AUC / PR-AUC: **54.44% / 62.08%**
- Precision / Recall / F1 at frozen threshold: **51.47% / 77.78% / 61.95%**
- Accuracy / Balanced Accuracy: **49.41% / 47.64%**
- NPV / Alert rate: **41.18% / 80.00%**
- TP/FP/FN/TN: **35/33/10/7**
- Prior simple recent7/28 Redsea AUC: **42.30%**

Important: Redsea was opened only after the generic feature schema, model family selection rule, and threshold were frozen on development data. No Redsea label was used to improve these reported external metrics.
