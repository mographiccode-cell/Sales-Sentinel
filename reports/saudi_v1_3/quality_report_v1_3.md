# Sales Sentinel — Saudi v1.3 SAMA-Calibrated Quality Report

- Version: **SA-LOCALIZATION-1.3-SAMA-CALIBRATED**
- UCI raw rows: **1,067,371**
- Clean/localized rows: **1,049,042**
- Unique observed source customers: **5,939**
- Invoice fallback identities excluded from customer target: **3,499**
- Training-safe days: **604**
- SAMA official national weeks available: **132**
- SAMA sector-week rows available: **2,244**
- Missing SAMA-calibration rows: **0**
- Duplicate localized line IDs: **0**
- Max full-week calibration relative error: **0.000002%**
- Maximum payment share difference: **0.000982%**
- All quality gates passed before training: **True**

## Scientific boundary

This is not observed Saudi merchant microdata. UCI Online Retail II provides the row-level transaction structure. Official SAMA weekly POS values provide Saudi market temporal and sector calibration. The calibrated merchant scale is intentionally not the national SAMA absolute scale.

## Leakage control

Current-week SAMA values are never supplied as prediction features. They are used only to calibrate the historical target series. Training, validation and test remain chronological, and model features are created from prior merchant observations.
