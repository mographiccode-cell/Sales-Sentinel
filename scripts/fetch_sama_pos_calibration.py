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
MANIFEST = REPORTS / "source_manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_export(data: bytes) -> pd.DataFrame:
    # Opendatasoft exports are commonly semicolon-delimited; sniff first, then fall back.
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

    frame = raw[[date_col, indicator_col, sector_col, city_col, value_col]].copy()
    frame.columns = ["week_start", "indicator", "sector", "city", "value"]
    frame["week_start"] = pd.to_datetime(frame["week_start"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["week_start", "indicator", "sector", "city", "value"])
    frame = frame[frame["week_start"].dt.year.between(2023, 2025)].copy()
    frame["week_end"] = frame["week_start"] + pd.Timedelta(days=6)
    frame["source"] = "Saudi Central Bank (SAMA) weekly POS via KAPSARC Data Portal"
    frame["data_status"] = "REAL_OFFICIAL_AGGREGATE"
    frame = frame.drop_duplicates().sort_values(["week_start", "city", "sector", "indicator"]).reset_index(drop=True)
    frame.to_csv(NORMALIZED, index=False)

    indicator_norm = frame["indicator"].astype(str).str.strip().str.lower()
    value_mask = indicator_norm.str.contains("value") & indicator_norm.str.contains("transaction") & ~indicator_norm.str.contains("change")
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
    checks = {
        "download_nonempty": len(payload) >= 10_000,
        "normalized_rows_positive": len(frame) > 0,
        "national_total_rows_positive": len(national_total) > 0,
        "contains_2023": int(years.get(2023, 0)) > 0,
        "contains_2024": int(years.get(2024, 0)) > 0,
        "no_duplicate_national_weeks": not national_total["week_start"].duplicated().any(),
        "national_values_positive": bool((national_total["value_thousand_sar"] > 0).all()),
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
        "checks": checks,
        "all_tests_passed": bool(all(checks.values())),
        "scientific_boundary": "Official Saudi aggregate POS data are used only as external calibration targets; they are not copied as merchant-level transactions.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not manifest["all_tests_passed"]:
        raise RuntimeError("SAMA POS source validation failed")


if __name__ == "__main__":
    main()
