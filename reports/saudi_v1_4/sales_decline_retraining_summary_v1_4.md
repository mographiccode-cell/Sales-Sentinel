# Sales Decline Retraining v1.4

- Dataset: **SA-LOCALIZATION-1.3.1-SAMA-SAFE**
- Target: next **7-day mean sales** falls by at least **20%** versus trailing **28-day mean**.
- Final test is the last **90** prediction origins and was untouched during selection.
- Purge gap: **7 days**.
- Selected model: **ExtraTrees**
- Selected probability threshold: **0.465**
- Test Accuracy: **90.000%**
- Test Balanced Accuracy: **54.938%**
- Test Precision: **50.000%**
- Test Recall: **11.111%**
- Test F1: **18.182%**
- Test ROC-AUC: **78.601%**
- Majority baseline Accuracy: **90.000%**
- All acceptance gates passed: **False**

The final test was not used to choose the model or threshold.
