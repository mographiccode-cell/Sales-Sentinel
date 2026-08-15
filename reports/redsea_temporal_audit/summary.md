# Redsea Real Saudi Merchant — Temporal & Accounting Audit

- SHA-256: `dd994d25042e54119babe058820b02e00b3443fe18150b9213f9c6929466a645`
- Raw / clean rows: **2,700 / 2,695**
- Exact duplicates: **5**
- Unique transactions / customers / items: **1,661 / 23 / 352**
- Active / calendar dates: **121 / 123**
- Zero-transaction days: **2**
- Missing cells by column: **{'ORG Name': 2700}**
- Type counts: **{'INV': 2579, 'CM': 121}**
- Negative quantity / net-amount rows: **121 / 121**
- Duplicate monetary effect (Net Amount): **SAR 6,257.77**

## Temporal signal
- Net sales daily CV: **88.48%**
- Autocorrelation lag1 / lag7 / lag28: **0.111 / 0.053 / 0.039**
- Frozen 7-day target usable rows: **89**
- Decline positives / prevalence: **47 / 52.81%**
- Target dates: **2023-07-28 → 2023-10-24**
- Simple recent7-vs28 AUC / PR-AUC: **42.30% / 45.05%**

- Monthly target: **{'2023-07': {'rows': 4, 'positives': 0, 'rate': 0.0}, '2023-08': {'rows': 31, 'positives': 17, 'rate': 0.5483870967741935}, '2023-09': {'rows': 30, 'positives': 13, 'rate': 0.43333333333333335}, '2023-10': {'rows': 24, 'positives': 17, 'rate': 0.7083333333333334}}**

Scientific handling: exact duplicate rows are removed; negative quantities/amounts are retained as observed returns/adjustments unless later transaction-type evidence justifies a different treatment. Net Amount (before VAT) is used for the Sales Sentinel net-sales target.
