# External Saudi Dataset Temporal-Signal Diagnostics

- Transactions / days: **49,998 / 1,461**
- Daily sales autocorrelation lag1 / lag7 / lag28: **0.019 / -0.033 / -0.036**
- Variance explained by weekday / month / year: **0.18% / 1.26% / 0.16%**
- Daily row-count CV: **17.70%**
- Simple recent-7-vs-28 risk AUC / PR-AUC: **47.55% / 8.16%**

## Per year
- 2020: rows=339, decline prevalence=10.62%, sales CV=27.48%, simple AUC=39.38%
- 2021: rows=365, decline prevalence=7.12%, sales CV=24.81%, simple AUC=56.01%
- 2022: rows=365, decline prevalence=9.59%, sales CV=24.47%, simple AUC=47.33%
- 2023: rows=358, decline prevalence=8.66%, sales CV=25.68%, simple AUC=51.32%

## Cardinality / repetition
- Customer Name: unique=26, top share=6.08%, normalized entropy=0.971
- Employee Name: unique=6, top share=18.80%, normalized entropy=0.997
- Manager Name: unique=2, top share=50.65%, normalized entropy=1.000
- Product Name: unique=100, top share=1.26%, normalized entropy=0.985
- Product Category: unique=10, top share=11.19%, normalized entropy=0.999
- City: unique=12, top share=13.80%, normalized entropy=0.977
- Channel: unique=2, top share=66.49%, normalized entropy=0.920
- Customer Type: unique=3, top share=49.73%, normalized entropy=0.922
- Customer Satisfaction: unique=5, top share=33.57%, normalized entropy=0.925
- Invoice ID: unique=49,998, top share=0.00%, normalized entropy=1.000

- Transactions by year: **{'2020': 12487, '2021': 12374, '2022': 12715, '2023': 12422}**
- Transactions by weekday: **{'0': 7057, '1': 7004, '2': 7046, '3': 7210, '4': 7287, '5': 7074, '6': 7320}**

Interpretation rule: very low temporal autocorrelation and negligible calendar variance imply weak forecasting signal, regardless of whether the table is geographically labeled Saudi. This diagnostic does not by itself prove synthetic generation.
