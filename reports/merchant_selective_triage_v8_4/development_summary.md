# Sales Sentinel V8.4 — Prequential Selective Triage

- Fold 0: **WATCH-only calibration burn-in**
- Folds 1-4 thresholds: **learned from earlier folds only**

## Post-calibration folds 1-4
- AMBER precision: **N/A**
- GREEN NPV: **90.10%**
- Decisive accuracy: **90.10%**
- Decisive coverage: **64.86%**
- WATCH rate: **35.14%**
- AMBER recall of all declines: **0.00%**
- AMBER/GREEN/WATCH counts: **0/192/104**

- Development oracle can meet AMBER>=80% and GREEN NPV>=95%: **True**
- Oracle decisive coverage: **58.79%**

Important: selective triage does not make full-coverage binary Precision and Recall both exceed 80%; uncertain cases are explicitly routed to WATCH.
