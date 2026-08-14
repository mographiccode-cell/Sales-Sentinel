# SAMA City Risk v2.3 — Rolling Conformal Policy

- Base model: **SAMA-CITY-RISK-2.2-STATIONARY-FROZEN**
- RED rule: global non-decline p ≤ **0.75%** AND city non-decline p ≤ **5.00%**
- GREEN rule: decline p ≤ **8.00%**
- Otherwise: **AMBER / abstain**
- Historical policy rows: **286**
- RED precision: **100.00%** (15 TP / 0 FP)
- RED FPR: **0.00%**
- RED+AMBER recall: **91.67%**
- GREEN NPV: **98.40%**
- Missed declines: **3 / 36**
- All historical conformal-policy gates passed: **False**
- Fresh 2025-2026 labels used to choose policy: **No**
