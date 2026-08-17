# Adaptive Merchant Forecast — Redsea external diagnostic

- Model: `SALES-SENTINEL-ADAPTIVE-MERCHANT-FORECAST-V2`
- Source SHA-256: `dd994d25042e54119babe058820b02e00b3443fe18150b9213f9c6929466a645`
- Source rows read: **2700**
- Calendarized daily window: **123 days** (2023-07-01 to 2023-10-31)
- Scientific status: **POST_OPEN_EXTERNAL_DIAGNOSTIC_NOT_FRESH_BLIND**

## 7-day walk-forward

- Folds: **9**
- Daily WAPE: **71.27%**
- 7-day TOTAL WAPE: **39.58%**
- 7-day TOTAL quality proxy (1-WAPE): **60.42%**
- Seasonal-naive TOTAL WAPE: **46.45%**
- TOTAL WAPE improvement vs seasonal naive: **6.87 percentage points**
- Prediction-interval empirical coverage: **87.30%**
- Selected-model counts: `{"weekday_median_8w": 4, "median_14": 2, "moving_average_7": 1, "median_7": 2}`

## 30-day walk-forward

- Folds: **3**
- Daily WAPE: **68.96%**
- 30-day TOTAL WAPE: **21.80%**
- 30-day TOTAL quality proxy (1-WAPE): **78.20%**
- Seasonal-naive TOTAL WAPE: **26.88%**
- TOTAL WAPE improvement vs seasonal naive: **5.09 percentage points**
- Prediction-interval empirical coverage: **87.78%**
- Selected-model counts: `{"weekday_median_8w": 2, "median_14": 1}`

## Boundary

Daily-value error and horizon-total error answer different questions. Sales Sentinel's decline decision is driven mainly by the total sales level over the next 7/30 days, so horizon-total WAPE is the more decision-relevant point-forecast diagnostic. This is still **not fresh blind validation** because Redsea was already inspected during development and covers only about four months.
