# Sales Sentinel v7.0 — Multi-Resolution Sector-to-Merchant Model

## Data design
- Source sector panel rows: **4,832** across **8 sectors** and **604 days**
- Sector supervised rows: **4,552**
- Merchant-total supervised rows: **541**
- Target: **next 7 days < 85% of trailing 28-day merchant baseline**
- Synthetic Region used as entity/feature: **No**
- Current/future official SAMA actuals used: **No**

## Frozen development selection
- Sector model: **ridge**
- Direct merchant model: **extra_trees**
- Blend weights — sector/direct: **0.12 / 0.88**
- Frozen decision threshold: **0.1100**
- Development OOF ROC-AUC: **73.92%**
- Development OOF PR-AUC: **44.59%**
- Development Precision / Recall / F1: **24.52% / 98.08% / 39.23%**
- Development GREEN NPV / Alert rate: **98.55% / 75.09%**

## V7 internal blind holdout — opened once after freeze
- Period: **2024-05-15 → 2024-08-19**
- Rows / positives: **97 / 15**
- ROC-AUC: **72.60%**
- PR-AUC: **35.52%**
- Accuracy / Balanced Accuracy: **17.53% / 51.22%**
- Precision: **15.79%**
- Recall: **100.00%**
- F1: **27.27%**
- GREEN NPV: **100.00%**
- Alert rate: **97.94%**
- TP / FP / FN / TN: **15 / 80 / 0 / 2**
- Performance gates passed: **False**
- Scientific integrity gates passed: **True**
- RED supported: **False**

## Scientific boundary
The V7 holdout was not used to choose its model, blend or threshold. It is still an **internal** holdout drawn from the same UCI-derived Saudi-localized longitudinal source, and its dates overlap prior project experimentation. It is not a substitute for independent real Saudi merchant validation.
