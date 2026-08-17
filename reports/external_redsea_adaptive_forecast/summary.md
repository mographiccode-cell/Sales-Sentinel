# Adaptive Merchant Forecast — Redsea external diagnostic

- Model: `SALES-SENTINEL-ADAPTIVE-MERCHANT-FORECAST-V3`
- Source SHA-256: `dd994d25042e54119babe058820b02e00b3443fe18150b9213f9c6929466a645`
- Source rows read: **2700**
- Calendarized daily window: **123 days** (2023-07-01 to 2023-10-31)
- Scientific status: **POST_OPEN_EXTERNAL_DIAGNOSTIC_NOT_FRESH_BLIND**

## 7-day walk-forward

- Folds: **9**
- Daily WAPE: **72.52%**
- 7-day TOTAL WAPE: **35.65%**
- 7-day TOTAL quality proxy (1-WAPE): **64.35%**
- Seasonal-naive TOTAL WAPE: **46.45%**
- TOTAL WAPE improvement vs seasonal naive: **10.80 percentage points**
- Prediction-interval empirical coverage: **90.48%**
- Selected-model counts: `{"weekday_median_8w": 3, "weekday_mean_8w": 6}`

## 30-day walk-forward

- Folds: **3**
- Daily WAPE: **68.62%**
- 30-day TOTAL WAPE: **9.51%**
- 30-day TOTAL quality proxy (1-WAPE): **90.49%**
- Seasonal-naive TOTAL WAPE: **26.88%**
- TOTAL WAPE improvement vs seasonal naive: **17.37 percentage points**
- Prediction-interval empirical coverage: **90.00%**
- Selected-model counts: `{"weekday_median_8w": 3}`

## Boundary

Daily-value error and horizon-total error answer different questions. Sales Sentinel's decline decision is driven mainly by the total sales level over the next 7/30 days, so horizon-total WAPE is the more decision-relevant point-forecast diagnostic. This is still **not fresh blind validation** because Redsea was already inspected during development and covers only about four months.
