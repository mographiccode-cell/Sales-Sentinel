# Sales Sentinel V8 — Target Refinement + Hard-Negative Features

- Status: **DEVELOPMENT_BEST**
- Candidates tested: **48**
- Selected: **{'model': 'xgb2', 'topk': 128, 'weight_profile': 'baseline'}**

## Same-target prequential evidence
- ROC-AUC / PR-AUC: **79.59% / 44.85%**
- Precision: V7.5 **40.31%** -> V8 **40.80%**
- Recall: V7.5 **82.54%** -> V8 **80.95%**
- F1: V7.5 **54.17%** -> V8 **54.26%**
- NPV: V7.5 **95.63%** -> V8 **95.31%**
- Alert rate: V7.5 **33.86%** -> V8 **32.81%**
- TP/FP/FN/TN: **51/74/12/244**
- Worst-fold recall: **60.00%**
- Strictly beats V7.5: **True**

## Error audit
- V7.5 FP/FN: **77/11**
- FP share near 15% ambiguity band (future_ratio 0.82-0.90): **16.88%**

Scientific note: V8 did not remove ambiguous rows from evaluation; any gain is directly comparable with V7.5 on the original 15% target. External Saudi merchant validation is still pending.
