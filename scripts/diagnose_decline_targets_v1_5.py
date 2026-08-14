from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "saudi_v1_3" / "saudi_daily_sama_calibrated_v1_3.csv"
OUT = ROOT / "reports" / "saudi_v1_5" / "target_diagnosis_preholdout.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

BASELINE = 28
HORIZONS = [1, 3, 7]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]


def main():
    d = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    sales = d["sama_calibrated_net_sales_sar"].astype(float)
    baseline = sales.rolling(BASELINE).mean()
    cutoff = int(len(d) * 0.75)
    rows = []
    for h in HORIZONS:
        future = pd.concat([sales.shift(-i) for i in range(1, h + 1)], axis=1).mean(axis=1)
        valid = pd.DataFrame({"date": d["date"], "baseline": baseline, "future": future}).dropna()
        valid = valid[(valid.index >= 56) & (valid.index < cutoff)].copy()
        # Three contiguous development segments to measure target stability without touching holdout.
        segment_edges = np.linspace(0, len(valid), 4, dtype=int)
        for t in THRESHOLDS:
            y = (valid["future"] < (1 - t) * valid["baseline"]).astype(int)
            seg_rates = []
            seg_counts = []
            for i in range(3):
                seg = y.iloc[segment_edges[i]:segment_edges[i+1]]
                seg_rates.append(float(seg.mean()))
                seg_counts.append(int(seg.sum()))
            rate = float(y.mean())
            stability_range = max(seg_rates) - min(seg_rates)
            rows.append({
                "horizon_days": h,
                "decline_threshold": t,
                "origins": int(len(y)),
                "positive_count": int(y.sum()),
                "positive_rate": rate,
                "segment_positive_rates": seg_rates,
                "segment_positive_counts": seg_counts,
                "segment_rate_range": float(stability_range),
                "usable_15_to_40pct_positive": bool(0.15 <= rate <= 0.40),
                "each_segment_at_least_10_positives": bool(all(c >= 10 for c in seg_counts)),
            })
    eligible = [r for r in rows if r["usable_15_to_40pct_positive"] and r["each_segment_at_least_10_positives"]]
    # Prefer a 7-day operational horizon, then the strongest decline threshold, then stability.
    eligible_sorted = sorted(
        eligible,
        key=lambda r: (
            r["horizon_days"] == 7,
            r["decline_threshold"],
            -r["segment_rate_range"],
        ),
        reverse=True,
    )
    result = {
        "development_only": True,
        "rows_total": len(d),
        "development_cutoff_index": cutoff,
        "development_end_date": str(d.loc[cutoff - 1, "date"].date()),
        "holdout_not_used": True,
        "criteria": {
            "positive_rate_range": [0.15, 0.40],
            "minimum_positives_per_development_segment": 10,
            "preference": "7-day horizon first; highest meaningful threshold that satisfies learnability/stability criteria",
        },
        "candidates": rows,
        "recommended_target": eligible_sorted[0] if eligible_sorted else None,
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
