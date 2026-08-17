# Adaptive Merchant Forecast — Redsea external diagnostic

- Model: `SALES-SENTINEL-ADAPTIVE-MERCHANT-FORECAST-V2`
- Source SHA-256: `dd994d25042e54119babe058820b02e00b3443fe18150b9213f9c6929466a645`
- Source rows read: **2700**
- Calendarized daily window: **123 days** (2023-07-01 to 2023-10-31)
- Scientific status: **POST_OPEN_EXTERNAL_DIAGNOSTIC_NOT_FRESH_BLIND**

## 7-day walk-forward

- Folds: **9**
- Evaluated daily points: **63**
- Adaptive WAPE: **71.27%**
- Adaptive 1-WAPE quality proxy: **28.73%**
- Seasonal-naive reference WAPE: **97.65%**
- WAPE improvement vs seasonal naive: **26.38 percentage points**
- Prediction-interval empirical coverage: **87.30%**
- Selected-model counts: `{"weekday_median_8w": 4, "median_14": 2, "moving_average_7": 1, "median_7": 2}`

## 30-day walk-forward

- Folds: **3**
- Evaluated daily points: **90**
- Adaptive WAPE: **68.96%**
- Adaptive 1-WAPE quality proxy: **31.04%**
- Seasonal-naive reference WAPE: **86.56%**
- WAPE improvement vs seasonal naive: **17.60 percentage points**
- Prediction-interval empirical coverage: **87.78%**
- Selected-model counts: `{"weekday_median_8w": 2, "median_14": 1}`

## Boundary

This is a real Saudi external-data transfer diagnostic, but it is **not fresh blind validation**: the Redsea dataset had already been inspected during development, and its time span is short. Do not present the values above as universal production accuracy. A new longitudinal Saudi merchant dataset remains required for that claim.
