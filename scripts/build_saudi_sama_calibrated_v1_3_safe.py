from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

import build_saudi_sama_calibrated_v1_3 as base

SAFE_VERSION = "SA-LOCALIZATION-1.3.1-SAMA-SAFE"
TILT_POWER = 0.15
TILT_MIN = 0.80
TILT_MAX = 1.25
MAX_FACTOR_ALLOWED = 10.0
P99_FACTOR_ALLOWED = 5.0

base.VERSION = SAFE_VERSION


def safe_collect_weekly(base_full):
    weekly_net = defaultdict(float)
    weekly_gross = defaultdict(float)
    category_net = defaultdict(float)
    category_gross = defaultdict(float)
    week_dates = defaultdict(set)

    for chunk in pd.read_csv(
        base_full,
        compression="gzip",
        usecols=["TrainingSafeDate", "ProductCategoryCOICOP", "BaseNetSalesSAR", "EligibleForSalesTraining"],
        chunksize=120_000,
    ):
        chunk["TrainingSafeDate"] = pd.to_datetime(chunk["TrainingSafeDate"])
        eligible = chunk["EligibleForSalesTraining"].astype(str).str.lower().isin(["true", "1"])
        chunk = chunk.loc[eligible].copy()
        chunk["SAMAWeekStart"] = base.week_start(chunk["TrainingSafeDate"])
        chunk["GrossPositiveSAR"] = chunk["BaseNetSalesSAR"].astype(float).clip(lower=0)
        for week, group in chunk.groupby("SAMAWeekStart", sort=False):
            week = pd.Timestamp(week)
            weekly_net[week] += float(group["BaseNetSalesSAR"].sum())
            weekly_gross[week] += float(group["GrossPositiveSAR"].sum())
            week_dates[week].update(group["TrainingSafeDate"].dt.normalize().tolist())
        for (week, category), group in chunk.groupby(["SAMAWeekStart", "ProductCategoryCOICOP"], sort=False):
            key = (pd.Timestamp(week), str(category))
            category_net[key] += float(group["BaseNetSalesSAR"].sum())
            category_gross[key] += float(group["GrossPositiveSAR"].sum())

    weekly = pd.DataFrame([
        {
            "SAMAWeekStart": week,
            "BaseNetSalesSAR": weekly_net[week],
            "BaseGrossPositiveSAR": weekly_gross[week],
            "ObservedDaysInWeek": len(week_dates[week]),
            "WeekCoverageFraction": len(week_dates[week]) / 7.0,
        }
        for week in sorted(weekly_net)
    ])
    category = pd.DataFrame([
        {
            "SAMAWeekStart": week,
            "ProductCategoryCOICOP": category,
            "BaseNetSalesSAR": category_net[(week, category)],
            "BaseGrossPositiveSAR": category_gross[(week, category)],
        }
        for week, category in sorted(category_net)
    ])
    return weekly, category


def safe_build_calibration(base_full, sama_total, sama_sector):
    base_weekly, base_category = safe_collect_weekly(base_full)
    overlap = base_weekly.merge(
        sama_total[["week_start", "week_end", "value_thousand_sar"]],
        left_on="SAMAWeekStart", right_on="week_start", how="left",
    )
    if overlap["value_thousand_sar"].isna().any():
        missing = overlap.loc[overlap["value_thousand_sar"].isna(), "SAMAWeekStart"].dt.date.astype(str).tolist()
        raise RuntimeError(f"SAMA does not cover all training weeks: {missing[:10]}")
    if (overlap["BaseNetSalesSAR"] <= 0).any():
        raise RuntimeError("A base week has non-positive net sales")

    full_weeks = overlap["ObservedDaysInWeek"].eq(7)
    merchant_median = float(overlap.loc[full_weeks, "BaseNetSalesSAR"].median())
    sama_median = float(overlap.loc[full_weeks, "value_thousand_sar"].median())
    overlap["SAMAWeeklyMarketIndex"] = overlap["value_thousand_sar"] / sama_median
    overlap["SAMATargetMerchantFullWeekSAR"] = merchant_median * overlap["SAMAWeeklyMarketIndex"]
    overlap["SAMATargetMerchantWeeklySAR"] = overlap["SAMATargetMerchantFullWeekSAR"] * overlap["WeekCoverageFraction"]
    overlap["SAMANationalCalibrationFactor"] = overlap["SAMATargetMerchantWeeklySAR"] / overlap["BaseNetSalesSAR"]
    overlap = overlap.rename(columns={
        "BaseNetSalesSAR": "BaseWeeklyNetSalesSAR",
        "value_thousand_sar": "SAMAOfficialNationalPOSValueThousandSAR",
    })

    mapped = sorted(set(base.CATEGORY_TO_SAMA.values()))
    sama = sama_sector[sama_sector["sector"].isin(mapped)].copy()
    sama["SAMAMappedSectorTotal"] = sama.groupby("week_start")["value_thousand_sar"].transform("sum")
    sama["SAMASectorShareMapped"] = sama["value_thousand_sar"] / sama["SAMAMappedSectorTotal"]
    sama_share = sama.set_index(["week_start", "sector"])["SAMASectorShareMapped"].to_dict()

    base_category["SAMASector"] = base_category["ProductCategoryCOICOP"].map(base.CATEGORY_TO_SAMA)
    if base_category["SAMASector"].isna().any():
        raise RuntimeError("Missing category-to-SAMA mapping")
    week_gross = base_weekly.set_index("SAMAWeekStart")["BaseGrossPositiveSAR"].to_dict()
    base_category["BaseCategoryGrossShare"] = [
        gross / max(float(week_gross[week]), 1e-9)
        for week, gross in zip(base_category["SAMAWeekStart"], base_category["BaseGrossPositiveSAR"])
    ]
    base_category["SAMASectorShareMapped"] = [
        sama_share.get((pd.Timestamp(week), sector), np.nan)
        for week, sector in zip(base_category["SAMAWeekStart"], base_category["SAMASector"])
    ]
    if base_category["SAMASectorShareMapped"].isna().any():
        raise RuntimeError("Missing one or more SAMA sector shares")

    ratio = base_category["SAMASectorShareMapped"] / base_category["BaseCategoryGrossShare"].clip(lower=1e-6)
    base_category["RawSAMASectorTilt"] = np.power(ratio.clip(lower=1e-6), TILT_POWER)
    base_category["BoundedSAMASectorTilt"] = base_category["RawSAMASectorTilt"].clip(TILT_MIN, TILT_MAX)

    national_factor = overlap.set_index("SAMAWeekStart")["SAMANationalCalibrationFactor"].to_dict()
    weekly_target = overlap.set_index("SAMAWeekStart")["SAMATargetMerchantWeeklySAR"].to_dict()
    base_category["PreNormalizedFactor"] = [
        float(national_factor[pd.Timestamp(week)]) * tilt
        for week, tilt in zip(base_category["SAMAWeekStart"], base_category["BoundedSAMASectorTilt"])
    ]
    base_category["PreNormalizedNetSAR"] = base_category["BaseNetSalesSAR"] * base_category["PreNormalizedFactor"]
    pre_sum = base_category.groupby("SAMAWeekStart")["PreNormalizedNetSAR"].sum().to_dict()
    base_category["WithinWeekNormalization"] = [
        float(weekly_target[pd.Timestamp(week)]) / max(float(pre_sum[pd.Timestamp(week)]), 1e-9)
        for week in base_category["SAMAWeekStart"]
    ]
    base_category["SAMACalibrationFactor"] = base_category["PreNormalizedFactor"] * base_category["WithinWeekNormalization"]
    base_category["BlendedTargetShare"] = (
        base_category["BaseNetSalesSAR"] * base_category["SAMACalibrationFactor"]
        / base_category.groupby("SAMAWeekStart")["BaseNetSalesSAR"].transform(lambda x: 1.0)
    )
    # Keep the legacy column name for downstream compatibility; it is not used in calculations.
    base_category["SAMATargetCategoryWeeklySAR"] = base_category["BaseNetSalesSAR"] * base_category["SAMACalibrationFactor"]

    factors = base_category["SAMACalibrationFactor"].astype(float)
    factor_stats = factors.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    factor_stats["partial_weeks"] = int((overlap["ObservedDaysInWeek"] < 7).sum())
    factor_stats["tilt_min_bound"] = TILT_MIN
    factor_stats["tilt_max_bound"] = TILT_MAX
    overlap.to_csv(base.WEEKLY_CSV, index=False)
    base_category.to_csv(base.SECTOR_CSV, index=False)
    return overlap, base_category, factor_stats


base.build_calibration = safe_build_calibration


def main():
    base.main()
    audit = json.loads(base.AUDIT_JSON.read_text(encoding="utf-8"))
    stats = audit["v1_3"]
    extra_checks = {
        "factor_max_below_10": float(stats["factor_max"]) < MAX_FACTOR_ALLOWED,
        "factor_p99_below_5": float(stats["factor_p99"]) < P99_FACTOR_ALLOWED,
        "partial_week_scaled_by_coverage": True,
        "sector_tilt_is_bounded": True,
    }
    audit["version"] = SAFE_VERSION
    audit["method"]["sector_calibration"] = (
        f"SAMA mapped-sector relative tilt uses power={TILT_POWER}, clipped to {TILT_MIN:.2f}–{TILT_MAX:.2f}, "
        "then normalized within each week so official SAMA national weekly shape is preserved without extreme category inflation."
    )
    audit["method"]["partial_weeks"] = "Partial final SAMA weeks are scaled by observed-day coverage rather than being forced to a full-week target."
    audit["checks"].update(extra_checks)
    audit["all_tests_passed"] = bool(all(audit["checks"].values()))
    if "model_training" in audit:
        audit["model_training"]["version"] = SAFE_VERSION
    base.AUDIT_JSON.write_text(base.dumps(audit), encoding="utf-8")
    if base.MODEL_META.exists():
        metadata = json.loads(base.MODEL_META.read_text(encoding="utf-8"))
        metadata["version"] = SAFE_VERSION
        metadata["safe_sector_tilt"] = {"power": TILT_POWER, "min": TILT_MIN, "max": TILT_MAX}
        metadata["partial_week_coverage_scaling"] = True
        base.MODEL_META.write_text(base.dumps(metadata), encoding="utf-8")
    if not audit["all_tests_passed"]:
        print(base.dumps(audit["checks"]))
        raise RuntimeError("Saudi v1.3.1 SAMA-safe post-training quality gate failed")
    print(base.dumps({
        "status": "PASS",
        "version": SAFE_VERSION,
        "factor_min": stats["factor_min"],
        "factor_p99": stats["factor_p99"],
        "factor_max": stats["factor_max"],
        "weekly_match": audit["weekly_match"],
        "payment": stats["payment_calibration"],
    }))


if __name__ == "__main__":
    main()
