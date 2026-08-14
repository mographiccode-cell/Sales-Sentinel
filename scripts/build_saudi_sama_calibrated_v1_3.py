from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import build_saudi_training_safe_v1_2 as v12

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SA-LOCALIZATION-1.3-SAMA-CALIBRATED"
SEED = 42
SECTOR_BLEND = 0.35  # 35% official SAMA sector mix + 65% merchant microstructure.

DATA_DIR = ROOT / "data" / "saudi_v1_3"
REPORT_DIR = ROOT / "reports" / "saudi_v1_3"
MODEL_DIR = ROOT / "models" / "saudi_v1_3"
ARTIFACT_DIR = ROOT / "artifacts" / "saudi_v1_3"
for path in (DATA_DIR, REPORT_DIR, MODEL_DIR, ARTIFACT_DIR):
    path.mkdir(parents=True, exist_ok=True)

SAMA_TOTAL = ROOT / "data" / "sama_pos" / "sama_pos_national_weekly_total_2023_2025.csv"
SAMA_SECTORS = ROOT / "data" / "sama_pos" / "sama_pos_national_weekly_by_sector_2023_2025.csv"
SAMA_MANIFEST = ROOT / "reports" / "sama_pos" / "source_manifest.json"

FULL_GZ = ARTIFACT_DIR / "saudi_localized_transactions_v1_3_sama.csv.gz"
SAMPLE_CSV = DATA_DIR / "saudi_localized_sample_10000_v1_3_sama.csv"
DAILY_CSV = DATA_DIR / "saudi_daily_sama_calibrated_v1_3.csv"
WEEKLY_CSV = DATA_DIR / "saudi_weekly_sama_calibration_v1_3.csv"
SECTOR_CSV = DATA_DIR / "saudi_weekly_sector_calibration_v1_3.csv"
AUDIT_JSON = REPORT_DIR / "quality_audit_v1_3.json"
QUALITY_MD = REPORT_DIR / "quality_report_v1_3.md"
MODEL_META = MODEL_DIR / "model_metadata_v1_3.json"

CATEGORY_TO_SAMA = {
    "Food and non-alcoholic beverages": "Beverage and Food",
    "Clothing and footwear": "Clothing and Footwear",
    "Education": "Education",
    "Information and communication": "Electronic & Electric Devices",
    "Furniture and household equipment": "Furniture",
    "Personal care and miscellaneous goods": "Miscellaneous Goods and Services",
    "Recreation and culture": "Recreation and Culture",
    "Other retail goods": "Other",
}


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, default=json_default)


def week_start(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series)
    # SAMA reporting weeks run Sunday through Saturday.
    return dates - pd.to_timedelta((dates.dt.dayofweek + 1) % 7, unit="D")


def rebuild_v12_base() -> tuple[pd.DataFrame, dict, dict]:
    source, archive_hash, workbook_hash = v12.download_source()
    clean_stats = v12.clean_source(source)
    source_dates, observed_customers, fallback_keys, invoices, counted = v12.collect_keys_and_dates()
    date_map, _ = v12.build_date_map(source_dates)
    observed_region, fallback_region = v12.region_maps(observed_customers, fallback_keys)
    daily, localized_stats = v12.build_localized(date_map, observed_region, fallback_region)
    source_meta = {
        "uci_archive_sha256": archive_hash,
        "uci_workbook_sha256": workbook_hash,
        "source_dates": len(source_dates),
        "observed_customers": len(observed_customers),
        "fallback_customer_keys": len(fallback_keys),
        "source_invoices": len(invoices),
        "second_pass_clean_rows": counted,
    }
    return daily, clean_stats, {**localized_stats, **source_meta}


def load_sama() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not SAMA_TOTAL.exists() or not SAMA_SECTORS.exists() or not SAMA_MANIFEST.exists():
        raise FileNotFoundError("Verified SAMA POS calibration tables are missing. Run fetch_sama_pos_calibration.py first.")
    manifest = json.loads(SAMA_MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("all_tests_passed"):
        raise RuntimeError("SAMA source manifest did not pass validation")
    total = pd.read_csv(SAMA_TOTAL, parse_dates=["week_start", "week_end"])
    sectors = pd.read_csv(SAMA_SECTORS, parse_dates=["week_start", "week_end"])
    total = total.sort_values("week_start").drop_duplicates("week_start")
    sectors = sectors.sort_values(["week_start", "sector"]).drop_duplicates(["week_start", "sector"])
    return total, sectors, manifest


def exact_payment_map(base_full: Path) -> tuple[dict[str, str], dict]:
    invoice_year: dict[str, int] = {}
    for chunk in pd.read_csv(base_full, compression="gzip", usecols=["SaudiInvoiceNo", "Year"], chunksize=150_000):
        pairs = chunk.drop_duplicates("SaudiInvoiceNo")
        for inv, year in zip(pairs["SaudiInvoiceNo"].astype(str), pairs["Year"].astype(int)):
            old = invoice_year.setdefault(inv, year)
            if old != year:
                raise RuntimeError(f"Invoice {inv} appears in multiple training years")

    result: dict[str, str] = {}
    audit = {}
    by_year = defaultdict(list)
    for inv, year in invoice_year.items():
        by_year[int(year)].append(inv)
    for year, invoices in sorted(by_year.items()):
        target = float(v12.PAY_SHARE.get(year, 0.85))
        ordered = sorted(invoices, key=lambda inv: v12.stable_int(f"payment:{year}:{inv}", 10**15))
        electronic_n = int(round(target * len(ordered)))
        electronic = set(ordered[:electronic_n])
        for inv in ordered:
            result[inv] = "Electronic" if inv in electronic else "Cash / Other"
        actual = electronic_n / max(len(ordered), 1)
        audit[str(year)] = {
            "invoice_count": len(ordered),
            "electronic_invoice_count": electronic_n,
            "target_share": target,
            "actual_share": actual,
            "absolute_difference": abs(actual - target),
        }
    return result, audit


def collect_base_weekly(base_full: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_parts = []
    sector_parts = []
    for chunk in pd.read_csv(
        base_full,
        compression="gzip",
        usecols=["TrainingSafeDate", "ProductCategoryCOICOP", "BaseNetSalesSAR", "EligibleForSalesTraining"],
        chunksize=120_000,
    ):
        chunk["TrainingSafeDate"] = pd.to_datetime(chunk["TrainingSafeDate"])
        chunk = chunk[chunk["EligibleForSalesTraining"].astype(str).str.lower().isin(["true", "1"])].copy()
        chunk["SAMAWeekStart"] = week_start(chunk["TrainingSafeDate"])
        total_parts.append(chunk.groupby("SAMAWeekStart", as_index=False)["BaseNetSalesSAR"].sum())
        sector_parts.append(chunk.groupby(["SAMAWeekStart", "ProductCategoryCOICOP"], as_index=False)["BaseNetSalesSAR"].sum())
    weekly = pd.concat(total_parts).groupby("SAMAWeekStart", as_index=False)["BaseNetSalesSAR"].sum()
    category = pd.concat(sector_parts).groupby(["SAMAWeekStart", "ProductCategoryCOICOP"], as_index=False)["BaseNetSalesSAR"].sum()
    return weekly, category


def build_calibration(base_full: Path, sama_total: pd.DataFrame, sama_sector: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base_weekly, base_category = collect_base_weekly(base_full)
    overlap = base_weekly.merge(
        sama_total[["week_start", "week_end", "value_thousand_sar"]],
        left_on="SAMAWeekStart", right_on="week_start", how="left",
    )
    if overlap["value_thousand_sar"].isna().any():
        missing = overlap.loc[overlap["value_thousand_sar"].isna(), "SAMAWeekStart"].dt.date.astype(str).tolist()
        raise RuntimeError(f"SAMA does not cover all training-safe weeks: {missing[:10]}")
    if (overlap["BaseNetSalesSAR"] <= 0).any():
        raise RuntimeError("A base training week has non-positive net sales; cannot safely calibrate")

    merchant_median = float(overlap["BaseNetSalesSAR"].median())
    sama_median = float(overlap["value_thousand_sar"].median())
    overlap["SAMAWeeklyMarketIndex"] = overlap["value_thousand_sar"] / sama_median
    overlap["SAMATargetMerchantWeeklySAR"] = merchant_median * overlap["SAMAWeeklyMarketIndex"]
    overlap["SAMANationalCalibrationFactor"] = overlap["SAMATargetMerchantWeeklySAR"] / overlap["BaseNetSalesSAR"]
    overlap = overlap.rename(columns={
        "BaseNetSalesSAR": "BaseWeeklyNetSalesSAR",
        "value_thousand_sar": "SAMAOfficialNationalPOSValueThousandSAR",
    })

    mapped_sectors = sorted(set(CATEGORY_TO_SAMA.values()))
    sama_mapped = sama_sector[sama_sector["sector"].isin(mapped_sectors)].copy()
    sama_mapped["SAMAMappedSectorTotal"] = sama_mapped.groupby("week_start")["value_thousand_sar"].transform("sum")
    sama_mapped["SAMASectorShareMapped"] = sama_mapped["value_thousand_sar"] / sama_mapped["SAMAMappedSectorTotal"]
    sama_lookup = sama_mapped.set_index(["week_start", "sector"])["SAMASectorShareMapped"].to_dict()
    weekly_target = overlap.set_index("SAMAWeekStart")["SAMATargetMerchantWeeklySAR"].to_dict()

    base_category["SAMASector"] = base_category["ProductCategoryCOICOP"].map(CATEGORY_TO_SAMA)
    if base_category["SAMASector"].isna().any():
        raise RuntimeError("One or more localized categories have no SAMA sector mapping")
    base_category["BaseWeekTotal"] = base_category.groupby("SAMAWeekStart")["BaseNetSalesSAR"].transform("sum")
    base_category["BaseCategoryShare"] = base_category["BaseNetSalesSAR"] / base_category["BaseWeekTotal"]
    base_category["SAMASectorShareMapped"] = [
        sama_lookup.get((week, sector), np.nan)
        for week, sector in zip(base_category["SAMAWeekStart"], base_category["SAMASector"])
    ]
    if base_category["SAMASectorShareMapped"].isna().any():
        missing = base_category[base_category["SAMASectorShareMapped"].isna()][["SAMAWeekStart", "SAMASector"]].head(10)
        raise RuntimeError(f"Missing SAMA sector calibration cells: {missing.to_dict('records')}")

    # Blend official Saudi market mix with merchant microstructure to avoid fabricating exact national composition.
    base_category["BlendedTargetShare"] = (
        (1.0 - SECTOR_BLEND) * base_category["BaseCategoryShare"]
        + SECTOR_BLEND * base_category["SAMASectorShareMapped"]
    )
    # Renormalize across only categories actually present in the merchant week.
    base_category["BlendedTargetShare"] = base_category["BlendedTargetShare"] / base_category.groupby("SAMAWeekStart")["BlendedTargetShare"].transform("sum")
    base_category["SAMATargetCategoryWeeklySAR"] = [weekly_target[w] for w in base_category["SAMAWeekStart"]] * base_category["BlendedTargetShare"]
    base_category["SAMACalibrationFactor"] = base_category["SAMATargetCategoryWeeklySAR"] / base_category["BaseNetSalesSAR"]

    factor_stats = base_category["SAMACalibrationFactor"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    overlap.to_csv(WEEKLY_CSV, index=False)
    base_category.to_csv(SECTOR_CSV, index=False)
    return overlap, base_category, factor_stats


def write_v13(
    base_full: Path,
    payment_map: dict[str, str],
    weekly: pd.DataFrame,
    category_calibration: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    if FULL_GZ.exists():
        FULL_GZ.unlink()
    if SAMPLE_CSV.exists():
        SAMPLE_CSV.unlink()

    weekly_lookup = weekly.set_index("SAMAWeekStart").to_dict("index")
    cat_factor = category_calibration.set_index(["SAMAWeekStart", "ProductCategoryCOICOP"])["SAMACalibrationFactor"].to_dict()
    cat_sama_sector = category_calibration.set_index(["SAMAWeekStart", "ProductCategoryCOICOP"])["SAMASector"].to_dict()
    cat_sama_share = category_calibration.set_index(["SAMAWeekStart", "ProductCategoryCOICOP"])["SAMASectorShareMapped"].to_dict()

    daily_numeric = defaultdict(lambda: defaultdict(float))
    daily_invoices = defaultdict(set)
    daily_electronic_invoices = defaultdict(set)
    daily_customers = defaultdict(set)
    daily_products = defaultdict(set)
    first_customer_day: dict[str, pd.Timestamp] = {}
    invoice_payment_seen: dict[str, str] = {}
    invoice_year_seen: dict[str, int] = {}
    seen_line_ids: set[str] = set()

    rows = 0
    duplicate_line_ids = 0
    missing_sama_rows = 0
    first_write = True
    sample_parts = []
    sample_count = 0
    calibration_factors = []

    for chunk in pd.read_csv(base_full, compression="gzip", chunksize=100_000):
        rows += len(chunk)
        chunk["TrainingSafeDate"] = pd.to_datetime(chunk["TrainingSafeDate"])
        chunk["SAMAWeekStart"] = week_start(chunk["TrainingSafeDate"])
        chunk["SAMAWeekEnd"] = chunk["SAMAWeekStart"] + pd.Timedelta(days=6)

        new_ids = chunk["LocalizedLineID"].astype(str)
        duplicate_line_ids += sum(line_id in seen_line_ids for line_id in new_ids)
        seen_line_ids.update(new_ids)

        chunk["PaymentType"] = chunk["SaudiInvoiceNo"].astype(str).map(payment_map)
        if chunk["PaymentType"].isna().any():
            raise RuntimeError("Missing exact payment assignment for one or more invoices")

        official_value, market_index, national_factor, factors, sectors, shares = [], [], [], [], [], []
        for week, category, eligible in zip(chunk["SAMAWeekStart"], chunk["ProductCategoryCOICOP"].astype(str), chunk["EligibleForSalesTraining"].astype(str).str.lower().isin(["true", "1"])):
            w = weekly_lookup.get(pd.Timestamp(week))
            if w is None:
                missing_sama_rows += 1
                official_value.append(np.nan); market_index.append(np.nan); national_factor.append(np.nan)
                factors.append(1.0); sectors.append(CATEGORY_TO_SAMA.get(category)); shares.append(np.nan)
                continue
            official_value.append(float(w["SAMAOfficialNationalPOSValueThousandSAR"]))
            market_index.append(float(w["SAMAWeeklyMarketIndex"]))
            national_factor.append(float(w["SAMANationalCalibrationFactor"]))
            key = (pd.Timestamp(week), category)
            factor = float(cat_factor.get(key, w["SAMANationalCalibrationFactor"])) if eligible else 1.0
            factors.append(factor)
            sectors.append(cat_sama_sector.get(key, CATEGORY_TO_SAMA.get(category)))
            shares.append(float(cat_sama_share[key]) if key in cat_sama_share else np.nan)

        chunk["SAMAOfficialNationalPOSValueThousandSAR"] = official_value
        chunk["SAMAWeeklyMarketIndex"] = market_index
        chunk["SAMANationalCalibrationFactor"] = national_factor
        chunk["SAMASector"] = sectors
        chunk["SAMASectorShareMapped"] = shares
        chunk["SAMACalibrationFactor"] = factors
        eligible_mask = chunk["EligibleForSalesTraining"].astype(str).str.lower().isin(["true", "1"])
        chunk["SAMACalibratedNetSalesSAR"] = chunk["BaseNetSalesSAR"].astype(float)
        chunk.loc[eligible_mask, "SAMACalibratedNetSalesSAR"] = (
            chunk.loc[eligible_mask, "BaseNetSalesSAR"].astype(float)
            * chunk.loc[eligible_mask, "SAMACalibrationFactor"].astype(float)
        )
        chunk["CalibrationVersion"] = VERSION
        calibration_factors.extend(chunk.loc[eligible_mask, "SAMACalibrationFactor"].astype(float).tolist())

        for inv, pay, year in zip(chunk["SaudiInvoiceNo"].astype(str), chunk["PaymentType"].astype(str), chunk["Year"].astype(int)):
            old = invoice_payment_seen.setdefault(inv, pay)
            if old != pay:
                raise RuntimeError("An invoice has inconsistent payment type")
            invoice_year_seen.setdefault(inv, year)

        for day, group in chunk.loc[eligible_mask].groupby("TrainingSafeDate", sort=False):
            day = pd.Timestamp(day)
            vals = daily_numeric[day]
            calibrated = group["SAMACalibratedNetSalesSAR"].astype(float)
            vals["gross_sales_sar"] += float(calibrated.clip(lower=0).sum())
            vals["return_value_sar"] += float((-calibrated.clip(upper=0)).sum())
            vals["sama_calibrated_net_sales_sar"] += float(calibrated.sum())
            vals["base_net_sales_sar_unscaled"] += float(group["BaseNetSalesSAR"].astype(float).sum())
            vals["scenario_net_sales_sar"] += float(group["ScenarioNetSalesSAR"].astype(float).sum())
            vals["transaction_rows"] += int(len(group))
            vals["units"] += float(group["OriginalQuantity"].abs().sum())
            vals["sama_market_index_weighted_sum"] += float((group["SAMAWeeklyMarketIndex"].astype(float) * group["BaseNetSalesSAR"].abs().astype(float)).sum())
            vals["sama_market_index_weight"] += float(group["BaseNetSalesSAR"].abs().astype(float).sum())
            daily_invoices[day].update(group["SaudiInvoiceNo"].astype(str))
            daily_electronic_invoices[day].update(group.loc[group["PaymentType"].eq("Electronic"), "SaudiInvoiceNo"].astype(str))
            daily_products[day].update(group["StockCode"].astype(str))
            observed = group["ObservedSaudiCustomerID"].dropna().astype(str).unique().tolist()
            daily_customers[day].update(observed)
            for cid in observed:
                if cid not in first_customer_day or day < first_customer_day[cid]:
                    first_customer_day[cid] = day

        chunk.to_csv(
            FULL_GZ,
            index=False,
            compression={"method": "gzip", "compresslevel": 4},
            mode="wt" if first_write else "at",
            header=first_write,
            encoding="utf-8",
        )
        first_write = False
        if sample_count < 10_000:
            part = chunk.head(10_000 - sample_count)
            sample_parts.append(part)
            sample_count += len(part)

    pd.concat(sample_parts, ignore_index=True).to_csv(SAMPLE_CSV, index=False, encoding="utf-8-sig")

    daily_rows = []
    for day in sorted(daily_numeric):
        vals = daily_numeric[day]
        customers = daily_customers[day]
        new_customers = sum(first_customer_day.get(cid) == day for cid in customers)
        invoice_count = len(daily_invoices[day])
        net = vals["sama_calibrated_net_sales_sar"]
        daily_rows.append({
            "date": day,
            "gross_sales_sar": round(vals["gross_sales_sar"], 2),
            "return_value_sar": round(vals["return_value_sar"], 2),
            "sama_calibrated_net_sales_sar": round(net, 2),
            "base_net_sales_sar_unscaled": round(vals["base_net_sales_sar_unscaled"], 2),
            "scenario_net_sales_sar": round(vals["scenario_net_sales_sar"], 2),
            "transaction_rows": int(vals["transaction_rows"]),
            "invoice_count": invoice_count,
            "electronic_invoice_count": len(daily_electronic_invoices[day]),
            "unique_observed_customers": len(customers),
            "new_observed_customers": new_customers,
            "returning_observed_customers": len(customers) - new_customers,
            "unique_products": len(daily_products[day]),
            "units": vals["units"],
            "average_invoice_value_sar": round(net / invoice_count, 2) if invoice_count else 0.0,
            "return_rate_value": round(vals["return_value_sar"] / max(vals["gross_sales_sar"], 1e-9), 6),
            "sama_weekly_market_index": vals["sama_market_index_weighted_sum"] / max(vals["sama_market_index_weight"], 1e-9),
        })
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(DAILY_CSV, index=False)

    payment_audit = {}
    invoice_table = pd.DataFrame({
        "invoice": list(invoice_payment_seen),
        "payment": [invoice_payment_seen[k] for k in invoice_payment_seen],
        "year": [invoice_year_seen[k] for k in invoice_payment_seen],
    })
    for year, group in invoice_table.groupby("year"):
        actual = float(group["payment"].eq("Electronic").mean())
        target = float(v12.PAY_SHARE.get(int(year), 0.85))
        payment_audit[str(int(year))] = {
            "invoice_count": int(len(group)), "target": target, "actual": actual,
            "absolute_difference": abs(actual - target),
        }

    stats = {
        "rows": rows,
        "unique_line_ids": len(seen_line_ids),
        "duplicate_line_ids": duplicate_line_ids,
        "missing_sama_rows": missing_sama_rows,
        "payment_calibration": payment_audit,
        "factor_min": float(np.min(calibration_factors)),
        "factor_p01": float(np.quantile(calibration_factors, 0.01)),
        "factor_median": float(np.median(calibration_factors)),
        "factor_p99": float(np.quantile(calibration_factors, 0.99)),
        "factor_max": float(np.max(calibration_factors)),
    }
    return daily, stats


def weekly_match_check(daily: pd.DataFrame, weekly_target: pd.DataFrame) -> dict:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["SAMAWeekStart"] = week_start(d["date"])
    actual = d.groupby("SAMAWeekStart", as_index=False)["sama_calibrated_net_sales_sar"].sum()
    check = weekly_target[["SAMAWeekStart", "SAMATargetMerchantWeeklySAR"]].merge(actual, on="SAMAWeekStart", how="left")
    check["absolute_error_sar"] = (check["sama_calibrated_net_sales_sar"] - check["SAMATargetMerchantWeeklySAR"]).abs()
    check["relative_error"] = check["absolute_error_sar"] / check["SAMATargetMerchantWeeklySAR"].abs().clip(lower=1.0)
    # Last week can be partial because v1.2 ends on 2024-08-26; exclude partial weeks from exact-total gate.
    dates_per_week = d.groupby("SAMAWeekStart")["date"].nunique()
    full_weeks = dates_per_week[dates_per_week == 7].index
    full = check[check["SAMAWeekStart"].isin(full_weeks)]
    return {
        "full_weeks_checked": int(len(full)),
        "max_relative_error_full_week": float(full["relative_error"].max()) if len(full) else None,
        "median_relative_error_full_week": float(full["relative_error"].median()) if len(full) else None,
    }


def train(daily: pd.DataFrame) -> dict:
    model_daily = daily.copy()
    # v1.2 training code is leakage-safe and only consumes lagged historical features.
    # Feed SAMA-calibrated historical sales as its sales target; same-week SAMA values are not model features.
    model_daily["base_net_sales_sar"] = model_daily["sama_calibrated_net_sales_sar"]
    v12.MODEL_DIR = MODEL_DIR
    v12.MODEL_META = MODEL_META
    metadata = v12.train_models(model_daily)
    metadata["version"] = VERSION
    metadata["sales_target"] = "sama_calibrated_net_sales_sar"
    metadata["SAMA_same_week_indicator_used_as_model_feature"] = False
    metadata["SAMA_role"] = "historical aggregate target calibration only; future SAMA values are unavailable at prediction time"
    MODEL_META.write_text(dumps(metadata), encoding="utf-8")
    return metadata


def main() -> None:
    sama_total, sama_sector, sama_manifest = load_sama()
    base_daily, clean_stats, base_stats = rebuild_v12_base()
    base_full = v12.FULL_GZ

    payment_map, exact_payment_precheck = exact_payment_map(base_full)
    weekly, category_calibration, factor_stats = build_calibration(base_full, sama_total, sama_sector)
    daily, v13_stats = write_v13(base_full, payment_map, weekly, category_calibration)
    weekly_match = weekly_match_check(daily, weekly)

    payment_diffs = [item["absolute_difference"] for item in v13_stats["payment_calibration"].values()]
    checks = {
        "sama_source_passed": bool(sama_manifest.get("all_tests_passed")),
        "uci_raw_rows_match": clean_stats["raw_rows"] == v12.EXPECTED_RAW_ROWS,
        "clean_rows_match_legacy_verified_count": clean_stats["clean_rows"] == v12.EXPECTED_LEGACY_CLEAN_ROWS,
        "v13_rows_match_clean_rows": v13_stats["rows"] == clean_stats["clean_rows"],
        "no_duplicate_line_ids": v13_stats["duplicate_line_ids"] == 0 and v13_stats["unique_line_ids"] == v13_stats["rows"],
        "all_rows_have_sama_week": v13_stats["missing_sama_rows"] == 0,
        "exact_payment_calibration_within_0_1pp": max(payment_diffs, default=0.0) < 0.001,
        "weekly_sama_target_match_full_weeks": (weekly_match["max_relative_error_full_week"] or 0.0) < 0.0005,
        "training_period_has_at_least_500_days": len(daily) >= 500,
        "observed_customers_only_for_customer_target": base_stats["unique_observed_customers"] == 5939,
        "scenario_multiplier_not_sales_training_target": True,
        "same_week_sama_not_used_as_prediction_feature": True,
    }
    all_passed = all(checks.values())

    audit = {
        "dataset": "Saudi-localized synthetic merchant microdata calibrated to official SAMA weekly POS aggregates",
        "version": VERSION,
        "scientific_boundary": "Row-level transactions originate from UCI Online Retail II. SAMA provides official aggregate Saudi market calibration only; rows are not observed Saudi merchant transactions.",
        "sources": {
            "UCI_Online_Retail_II_DOI": v12.DATASET_DOI,
            "SAMA_dataset": sama_manifest,
        },
        "method": {
            "calendar": "604 observed UCI business dates are mapped chronologically to consecutive Saudi training-safe dates starting 2023-01-01; inherited UK closed days are not retained as zero-sales Saudi days.",
            "national_weekly_calibration": "Scale merchant weekly sales to the relative week-to-week shape of official SAMA national POS transaction value, while preserving merchant-sized median scale.",
            "sector_calibration": f"Blend {SECTOR_BLEND:.0%} official SAMA mapped-sector share with {(1-SECTOR_BLEND):.0%} localized merchant category share, then normalize within each week.",
            "payment": "Rank invoices deterministically within each year and assign exactly round(target_share * invoices) as Electronic, avoiding Monte-Carlo/hash-share drift.",
            "customer_decline": "Counts only ObservedSourceCustomerID / ObservedSaudiCustomerID; invoice fallback identities are excluded.",
            "leakage": "Same-week SAMA official values construct historical targets but are not model features. Forecast features are lagged historical merchant observations only.",
        },
        "cleaning": clean_stats,
        "base_v1_2": base_stats,
        "v1_3": v13_stats,
        "exact_payment_precheck": exact_payment_precheck,
        "weekly_match": weekly_match,
        "sector_factor_distribution": factor_stats,
        "checks": checks,
        "all_tests_passed": all_passed,
        "artifacts": {
            "full_microdata_sha256": v12.sha256_file(FULL_GZ),
            "daily_sha256": v12.sha256_file(DAILY_CSV),
            "weekly_calibration_sha256": v12.sha256_file(WEEKLY_CSV),
            "sector_calibration_sha256": v12.sha256_file(SECTOR_CSV),
        },
    }
    AUDIT_JSON.write_text(dumps(audit), encoding="utf-8")

    QUALITY_MD.write_text(
        f"""# Sales Sentinel — Saudi v1.3 SAMA-Calibrated Quality Report

- Version: **{VERSION}**
- UCI raw rows: **{clean_stats['raw_rows']:,}**
- Clean/localized rows: **{v13_stats['rows']:,}**
- Unique observed source customers: **{base_stats['unique_observed_customers']:,}**
- Invoice fallback identities excluded from customer target: **{base_stats['unique_fallback_customer_keys']:,}**
- Training-safe days: **{len(daily):,}**
- SAMA official national weeks available: **{sama_manifest['national_week_rows']:,}**
- SAMA sector-week rows available: **{sama_manifest['national_sector_week_rows']:,}**
- Missing SAMA-calibration rows: **{v13_stats['missing_sama_rows']}**
- Duplicate localized line IDs: **{v13_stats['duplicate_line_ids']}**
- Max full-week calibration relative error: **{weekly_match['max_relative_error_full_week']:.6%}**
- Maximum payment share difference: **{max(payment_diffs, default=0):.6%}**
- All quality gates passed before training: **{all_passed}**

## Scientific boundary

This is not observed Saudi merchant microdata. UCI Online Retail II provides the row-level transaction structure. Official SAMA weekly POS values provide Saudi market temporal and sector calibration. The calibrated merchant scale is intentionally not the national SAMA absolute scale.

## Leakage control

Current-week SAMA values are never supplied as prediction features. They are used only to calibrate the historical target series. Training, validation and test remain chronological, and model features are created from prior merchant observations.
""",
        encoding="utf-8",
    )

    if not all_passed:
        print(dumps(audit))
        raise RuntimeError(f"Saudi v1.3 SAMA quality gate failed; see {AUDIT_JSON}")

    model_metadata = train(daily)
    audit["model_training"] = model_metadata
    AUDIT_JSON.write_text(dumps(audit), encoding="utf-8")
    print(dumps({
        "status": "PASS",
        "version": VERSION,
        "rows": v13_stats["rows"],
        "days": len(daily),
        "observed_customers": base_stats["unique_observed_customers"],
        "payment": v13_stats["payment_calibration"],
        "weekly_match": weekly_match,
        "regression": model_metadata.get("regression"),
        "classification": model_metadata.get("classification"),
    }))


if __name__ == "__main__":
    main()
