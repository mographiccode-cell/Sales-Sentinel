from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sama_pos"
REPORTS = ROOT / "reports" / "sama_pos"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

EXPORT_URL = "https://data.kapsarc.org/api/explore/v2.1/catalog/datasets/point-of-sale-transactions-by-sector-and-city/exports/csv"
SAMA_PAGE = "https://www.sama.gov.sa/en-US/Statistics/Indices/Pages/POS.aspx"
DATASET_PAGE = "https://data.kapsarc.org/explore/dataset/point-of-sale-transactions-by-sector-and-city/analyze/?flg=en-gb"

RAW = OUT / "sama_pos_full_export.csv"
NORMALIZED = OUT / "sama_pos_2023_2025_normalized.csv"
NATIONAL = OUT / "sama_pos_national_weekly_total_2023_2025.csv"
SECTORS = OUT / "sama_pos_national_weekly_by_sector_2023_2025.csv"
HISTORY_NORMALIZED = OUT / "sama_pos_2020_2025_normalized.csv"
HISTORY_VALUE = OUT / "sama_pos_national_weekly_value_2020_2025.csv"
HISTORY_COUNT = OUT / "sama_pos_national_weekly_count_2020_2025.csv"
MANIFEST = REPORTS / "source_manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_export(data: bytes) -> pd.DataFrame:
    text = data.decode("utf-8-sig")
    for sep in (";", ",", "\t"):
        try:
            frame = pd.read_csv(io.StringIO(text), sep=sep)
            if len(frame.columns) >= 5:
                return frame
        except Exception:
            pass
    raise RuntimeError("Unable to parse SAMA/KAPSARC POS CSV export")


def resolve_column(columns: list[str], candidates: list[str]) -> str:
    normalized = {c.strip().lower().replace(" ", "_"): c for c in columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    for col in columns:
        low = col.lower()
        if any(candidate.lower() in low for candidate in candidates):
            return col
    raise KeyError(f"Missing expected column. Tried {candidates}. Available={columns}")


def national_total_for_indicator(frame: pd.DataFrame, mask: pd.Series, value_name: str) -> pd.DataFrame:
    national = frame[
        mask
        & frame["city"].astype(str).str.strip().str.lower().eq("total")
        & frame["sector"].astype(str).str.strip().str.lower().eq("total")
    ].copy()
    if national.empty:
        raise RuntimeError(f"No national Total/Total SAMA rows found for {value_name}")
    national = national[["week_start", "week_end", "value", "source", "data_status"]].rename(columns={"value": value_name})
    return national.sort_values("week_start").drop_duplicates("week_start", keep="last")


def main() -> None:
    response = requests.get(EXPORT_URL, timeout=180, headers={"User-Agent": "Sales-Sentinel-Academic/1.0"})
    response.raise_for_status()
    payload = response.content
    if len(payload) < 10_000:
        raise RuntimeError(f"SAMA/KAPSARC export is unexpectedly small: {len(payload)} bytes")
    RAW.write_bytes(payload)

    raw = read_export(payload)
    cols = list(raw.columns)
    date_col = resolve_column(cols, ["starting_date", "starting date"])
    indicator_col = resolve_column(cols, ["number_value_change_transactions", "indicator"])
    sector_col = resolve_column(cols, ["sectors", "sector"])
    city_col = resolve_column(cols, ["city"])
    value_col = resolve_column(cols, ["value"])

    base = raw[[date_col, indicator_col, sector_col, city_col, value_col]].copy()
    base.columns = ["week_start", "indicator", "sector", "city", "value"]
    base["week_start"] = pd.to_datetime(base["week_start"], errors="coerce")
    base["value"] = pd.to_numeric(base["value"], errors="coerce")
    base = base.dropna(subset=["week_start", "indicator", "sector", "city", "value"])
    base["week_end"] = base["week_start"] + pd.Timedelta(days=6)
    base["source"] = "Saudi Central Bank (SAMA) weekly POS via KAPSARC Data Portal"
    base["data_status"] = "REAL_OFFICIAL_AGGREGATE"
    base = base.drop_duplicates().sort_values(["week_start", "city", "sector", "indicator"]).reset_index(drop=True)

    min_year = max(2020, int(base["week_start"].dt.year.min()))
    history = base[base["week_start"].dt.year.between(min_year, 2025)].copy()
    history.to_csv(HISTORY_NORMALIZED, index=False)

    indicator_norm = history["indicator"].astype(str).str.strip().str.lower()
    value_mask_hist = indicator_norm.str.contains("value") & indicator_norm.str.contains("transaction") & ~indicator_norm.str.contains("change")
    number_mask_hist = indicator_norm.str.contains("number") & indicator_norm.str.contains("transaction") & ~indicator_norm.str.contains("change")

    national_value_history = national_total_for_indicator(history, value_mask_hist, "value_thousand_sar")
    national_count_history = national_total_for_indicator(history, number_mask_hist, "transaction_count")
    national_value_history.to_csv(HISTORY_VALUE, index=False)
    national_count_history.to_csv(HISTORY_COUNT, index=False)

    # Keep existing 2023-2025 products unchanged for v1.3 reproducibility.
    frame = history[history["week_start"].dt.year.between(2023, 2025)].copy()
    frame.to_csv(NORMALIZED, index=False)
    indicator_norm_23 = frame["indicator"].astype(str).str.strip().str.lower()
    value_mask = indicator_norm_23.str.contains("value") & indicator_norm_23.str.contains("transaction") & ~indicator_norm_23.str.contains("change")
    national = frame[value_mask & frame["city"].astype(str).str.strip().str.lower().eq("total")].copy()
    national_total = national[national["sector"].astype(str).str.strip().str.lower().eq("total")].copy()
    if national_total.empty:
        raise RuntimeError("No national Total/Total SAMA POS value rows found")
    national_total = national_total[["week_start", "week_end", "value", "source", "data_status"]].rename(columns={"value": "value_thousand_sar"})
    national_total = national_total.sort_values("week_start").drop_duplicates("week_start", keep="last")
    national_total.to_csv(NATIONAL, index=False)

    sector_values = national[~national["sector"].astype(str).str.strip().str.lower().eq("total")].copy()
    sector_values = sector_values[["week_start", "week_end", "sector", "value", "source", "data_status"]].rename(columns={"value": "value_thousand_sar"})
    sector_values = sector_values.sort_values(["week_start", "sector"]).drop_duplicates(["week_start", "sector"], keep="last")
    sector_values.to_csv(SECTORS, index=False)

    years = national_total["week_start"].dt.year.value_counts().sort_index().to_dict()
    history_years = national_value_history["week_start"].dt.year.value_counts().sort_index().to_dict()
    aligned = national_value_history[["week_start"]].merge(national_count_history[["week_start"]], on="week_start", how="inner")
    checks = {
        "download_nonempty": len(payload) >= 10_000,
        "normalized_rows_positive": len(frame) > 0,
        "national_total_rows_positive": len(national_total) > 0,
        "contains_2023": int(years.get(2023, 0)) > 0,
        "contains_2024": int(years.get(2024, 0)) > 0,
        "no_duplicate_national_weeks": not national_total["week_start"].duplicated().any(),
        "national_values_positive": bool((national_total["value_thousand_sar"] > 0).all()),
        "history_starts_2020_or_earlier_available_floor": int(national_value_history["week_start"].dt.year.min()) <= 2020,
        "history_value_has_at_least_250_weeks": len(national_value_history) >= 250,
        "history_count_has_at_least_250_weeks": len(national_count_history) >= 250,
        "history_value_count_alignment_at_least_250_weeks": len(aligned) >= 250,
        "history_counts_positive": bool((national_count_history["transaction_count"] > 0).all()),
    }
    manifest = {
        "dataset": "Point of Sale Transactions by Sector and City",
        "publisher": "Saudi Central Bank (SAMA)",
        "distribution": "KAPSARC Data Portal Opendatasoft export",
        "sama_official_page": SAMA_PAGE,
        "dataset_page": DATASET_PAGE,
        "export_url": EXPORT_URL,
        "download_sha256": sha256_bytes(payload),
        "download_bytes": len(payload),
        "raw_rows": int(len(raw)),
        "normalized_rows_2023_2025": int(len(frame)),
        "national_week_rows": int(len(national_total)),
        "national_sector_week_rows": int(len(sector_values)),
        "national_weeks_by_year": {str(k): int(v) for k, v in years.items()},
        "date_min": national_total["week_start"].min().date().isoformat(),
        "date_max": national_total["week_start"].max().date().isoformat(),
        "history": {
            "min_year": int(national_value_history["week_start"].dt.year.min()),
            "max_year": int(national_value_history["week_start"].dt.year.max()),
            "value_weeks": int(len(national_value_history)),
            "count_weeks": int(len(national_count_history)),
            "aligned_weeks": int(len(aligned)),
            "value_weeks_by_year": {str(k): int(v) for k, v in history_years.items()},
        },
        "checks": checks,
        "all_tests_passed": bool(all(checks.values())),
        "scientific_boundary": "Official Saudi aggregate POS data are used as external calibration and forecast inputs; they are never represented as merchant-level observed transactions.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not manifest["all_tests_passed"]:
        raise RuntimeError("SAMA POS source validation failed")


if __name__ == "__main__":
    main()
