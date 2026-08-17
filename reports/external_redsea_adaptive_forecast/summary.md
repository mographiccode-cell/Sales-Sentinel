# Adaptive Merchant Forecast — Redsea external diagnostic

- Model: `SALES-SENTINEL-ADAPTIVE-MERCHANT-FORECAST-V1`
- Source SHA-256: `dd994d25042e54119babe058820b02e00b3443fe18150b9213f9c6929466a645`
- Source rows read: **2700**
- Calendarized daily window: **123 days** (2023-07-01 to 2023-10-31)
- Scientific status: **POST_OPEN_EXTERNAL_DIAGNOSTIC_NOT_FRESH_BLIND**

## 7-day walk-forward

- Folds: **9**
- Evaluated daily points: **63**
- Adaptive WAPE: **72.54%**
- Adaptive 1-WAPE quality proxy: **27.46%**
- Seasonal-naive reference WAPE: **97.65%**
- WAPE improvement vs seasonal naive: **25.11 percentage points**
- Prediction-interval empirical coverage: **90.48%**
- Selected-model counts: `{"moving_average_14": 4, "moving_average_7": 5}`

## 30-day walk-forward

- Folds: **3**
- Evaluated daily points: **90**
- Adaptive WAPE: **78.66%**
- Adaptive 1-WAPE quality proxy: **21.34%**
- Seasonal-naive reference WAPE: **86.56%**
- WAPE improvement vs seasonal naive: **7.90 percentage points**
- Prediction-interval empirical coverage: **87.78%**
- Selected-model counts: `{"moving_average_14": 2, "moving_average_7": 1}`

## Boundary

This is a real Saudi external-data transfer diagnostic, but it is **not fresh blind validation**: the Redsea dataset had already been inspected during development, and its time span is short. Do not present the values above as universal production accuracy. A new longitudinal Saudi merchant dataset remains required for that claim.
