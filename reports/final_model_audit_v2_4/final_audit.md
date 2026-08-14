# Sales Sentinel v2.4 — Final Model Audit

- Final decision: **NOT_FINAL_APPROVED**
- Leakage audit: **PASS**
- Generalization gates: **FAIL**
- Development RED precision: **95.00%** (19 TP / 1 FP), Wilson 95% lower **76.39%**
- Development GREEN NPV: **100.00%**, Wilson 95% lower **95.31%**
- Stress RED precision: **24.44%** (11 TP / 34 FP)
- Stress RED FPR: **5.78%**
- Stress RED+AMBER recall: **94.12%**
- Stress GREEN NPV: **99.34%**
- Stress ROC-AUC: **86.20%**
- Stress PR-AUC: **29.87%**
- App test input: `tests/fixtures/sales_sentinel_v2_4/app_test_input_sama_real_130_weeks.csv`
- Expected output: `tests/fixtures/sales_sentinel_v2_4/app_test_expected_predictions.json`

The model is not finally approved until a post-freeze prospective validation passes the frozen production contract.
