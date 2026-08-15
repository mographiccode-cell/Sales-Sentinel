# V7 Experiment 1 vs V6.1

## Decision

**V7 Experiment 1 is not promoted as the replacement for V6.1.**

The V7 panel experiment increased supervised rows from 541 merchant-total rows to 4,328 category-day panel rows and used an untouched 84-day blind holdout. It preserved recall, but produced an operationally excessive alert rate and materially lower GREEN NPV.

| Metric | V6.1 OOF | V7 Blind Holdout |
|---|---:|---:|
| ROC-AUC | 75.83% | 67.08% |
| PR-AUC | 38.73% | 43.41%* |
| Precision | 38.81% | 37.41% |
| Recall | 82.54% | 82.50% |
| F1 | 52.79% | 51.48% |
| GREEN NPV | 95.55% | 84.85% |
| Alert rate | 35.17% | 65.62% |

*PR-AUC is not directly comparable because the V7 category-day target has a substantially different positive prevalence (200/672 = 29.76% on blind holdout) versus the V6.1 merchant-total OOF prevalence (~16.54%).

## Scientific interpretation

1. Increasing panel rows alone does not guarantee better merchant-total warning quality.
2. Category-day targets are noisier and change the class prevalence and operational meaning of an alert.
3. V7 Experiment 1 therefore demonstrates that the next iteration must separate **category-level representation learning** from the **merchant-total decision target**, rather than directly treating every category-day row as an independent alert target.
4. The 84-day V7 blind holdout has now been opened and must not be reused as an untouched final holdout for threshold/model tuning in later V7 iterations.

## Next iteration

V7.1 should use category/sector panel information to create merchant-level aggregate signals, while preserving the original merchant-total early-decline target. Model/threshold selection should use nested rolling-origin development folds. A new external Saudi merchant time series is still required for a genuinely untouched final validation.
