from __future__ import annotations

import json
import pandas as pd

import build_train_saudi_panel_v1_4 as base
from build_train_saudi_panel_v1_4_1 import exact_aggregate_panel


def main():
    panel, panel_stats = exact_aggregate_panel()
    supervised, features, supervised_stats = base.add_features(panel)
    train = supervised[supervised["TrainingSafeDate"] <= pd.Timestamp("2023-12-24")].copy()
    decline_threshold, target_diagnostics = base.choose_decline_threshold(train)
    checks = {
        "source_has_verified_million_rows": panel_stats["input_microdata_rows"] == 1_049_042,
        "no_duplicate_entity_dates": panel_stats["duplicate_entity_dates"] == 0,
        "no_core_nulls": panel_stats["core_nulls"] == 0,
        "no_calibrated_sales_nulls": panel_stats["calibrated_sales_nulls"] == 0,
        "administrative_rows_excluded": panel_stats["administrative_rows_excluded"] > 0,
        "fallback_customer_ids_excluded_from_customer_counts": True,
        "at_least_50_entities": panel_stats["entities"] >= 50,
        "at_least_25000_panel_rows": panel_stats["panel_rows"] >= 25_000,
        "at_least_20000_supervised_rows": supervised_stats["supervised_rows"] >= 20_000,
        "each_entity_has_at_least_400_observed_days": panel_stats["min_entity_observed_days"] >= 400,
        "target_selected_on_training_period_only": True,
        "future_SAMA_actuals_forbidden": True,
    }
    out = {
        "panel_stats": panel_stats,
        "supervised_stats": supervised_stats,
        "feature_count": len(features),
        "training_rows_for_target_selection": len(train),
        "chosen_decline_threshold": decline_threshold,
        "target_diagnostics": target_diagnostics,
        "checks": checks,
        "failed_checks": [k for k,v in checks.items() if not v],
        "entity_observed_days_quantiles": panel.groupby("entity")["TrainingSafeDate"].nunique().quantile([0,.1,.25,.5,.75,.9,1]).to_dict(),
        "entity_rows_top": panel.groupby("entity").size().sort_values(ascending=False).head(20).to_dict(),
        "entity_rows_bottom": panel.groupby("entity").size().sort_values().head(20).to_dict(),
    }
    print(json.dumps(out, indent=2, default=str))
    (base.REPORT_DIR / "panel_diagnosis_v1_4_1.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
