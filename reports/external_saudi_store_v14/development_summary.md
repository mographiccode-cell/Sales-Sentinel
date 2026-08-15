# Sales Sentinel V14 — External Dataset Blind 2023 Experiment

- Status: **EXTERNAL_DATASET_BLIND_EVALUATED**
- Provenance: **Saudi-labeled public Kaggle retail dataset; operational provenance unverified**
- Raw rows / daily days: **49,998 / 1,461**
- Development / blind rows: **1,040 / 358**
- Blind period: **2023-01-01 → 2023-12-24**
- Selected model from development only: **extra**
- Frozen threshold: **0.255**

## Development OOF
- ROC-AUC / PR-AUC: **71.94% / 15.95%**
- Precision / Recall / F1: **15.65% / 80.70% / 26.21%**
- NPV / Alert rate: **97.26% / 42.30%**

## Untouched 2023 blind holdout
- ROC-AUC / PR-AUC: **56.68% / 10.88%**
- Accuracy / Balanced Accuracy: **48.04% / 52.58%**
- Precision / Recall / F1: **9.42% / 58.06% / 16.22%**
- NPV / Alert rate: **92.22% / 53.35%**
- TP/FP/FN/TN: **18/173/13/154**

Important: this is independent temporal evidence on a different public Saudi-labeled dataset, but not proof of performance on verified real Saudi merchant operational data.
