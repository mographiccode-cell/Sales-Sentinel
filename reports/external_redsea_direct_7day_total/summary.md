# Direct 7-day total forecast — Redsea experiment

- Source rows: **2700**
- Calendar days: **123** (2023-07-01 to 2023-10-31)
- Candidate models: **12**
- Folds: **9**
- Direct-total WAPE: **36.94%**
- Direct-total quality proxy (1-WAPE): **63.06%**
- Current V3 7-day total WAPE reference: **35.65%**
- Improvement vs V3: **-1.29 percentage points**
- Winner counts: `{"mean_8w_total": 8, "mean_4w_total": 1}`

## Decision rule

This remains a post-open Redsea experiment. It may replace the V3 seven-day point-forecast path only if it improves total WAPE and then passes the full application CI. It is not fresh blind validation.
