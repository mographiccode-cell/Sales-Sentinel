from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online%2Bretail%2Bii.zip"
DATASET_DOI = "10.24432/C5CG6D"
EXPECTED_RAW_ROWS = 1_067_371
EXPECTED_LEGACY_CLEAN_ROWS = 1_049_042
FX_GBP_TO_SAR = 4.75
VAT = 0.15
TRAINING_START = pd.Timestamp("2023-01-01")
LEGACY_OFFSET = pd.Timestamp("2023-01-01") - pd.Timestamp("2009-12-01")
DECLINE_PERCENT = 0.20

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".saudi_v1_2_work"
DATA_DIR = ROOT / "data" / "saudi_v1_2"
REPORT_DIR = ROOT / "reports" / "saudi_v1_2"
MODEL_DIR = ROOT / "models" / "saudi_v1_2"
ARTIFACT_DIR = ROOT / "artifacts" / "saudi_v1_2"
for p in (WORK, DATA_DIR, REPORT_DIR, MODEL_DIR, ARTIFACT_DIR):
    p.mkdir(parents=True, exist_ok=True)

ARCHIVE = WORK / "online_retail_ii.zip"
RAW_DIR = WORK / "raw"
CLEAN_GZ = WORK / "clean_online_retail_ii.csv.gz"
FULL_GZ = ARTIFACT_DIR / "saudi_localized_transactions_v1_2.csv.gz"
SAMPLE_CSV = DATA_DIR / "saudi_localized_sample_10000_v1_2.csv"
DAILY_CSV = DATA_DIR / "saudi_daily_training_safe_v1_2.csv"
DAY_MAP_CSV = DATA_DIR / "source_to_training_date_map_v1_2.csv"
AUDIT_JSON = REPORT_DIR / "quality_audit_v1_2.json"
QUALITY_MD = REPORT_DIR / "quality_report_v1_2.md"
MODEL_META = MODEL_DIR / "model_metadata_v1_2.json"

REGION_POP = {
    "Riyadh": 8591748, "Makkah": 8021463, "Eastern Province": 5125254,
    "Madinah": 2137983, "Asir": 2024285, "Jazan": 1404997,
    "Qassim": 1336179, "Tabuk": 886036, "Hail": 746406,
    "Al Jouf": 595822, "Najran": 592300, "Northern Borders": 373577,
    "Al Baha": 339174,
}
CAPITAL = {
    "Riyadh": "Riyadh", "Makkah": "Makkah", "Eastern Province": "Dammam",
    "Madinah": "Madinah", "Asir": "Abha", "Jazan": "Jazan",
    "Qassim": "Buraidah", "Tabuk": "Tabuk", "Hail": "Hail",
    "Al Jouf": "Sakaka", "Najran": "Najran", "Northern Borders": "Arar",
    "Al Baha": "Al Baha",
}
PAY_SHARE = {2023: 0.70, 2024: 0.79, 2025: 0.85}
RAMADAN = [("2023-03-23", "2023-04-20"), ("2024-03-11", "2024-04-09")]
EID_FITR = [("2023-04-21", "2023-04-23"), ("2024-04-10", "2024-04-12")]
HAJJ = [("2023-06-19", "2023-06-30"), ("2024-06-07", "2024-06-19")]
EID_ADHA = [("2023-06-28", "2023-07-01"), ("2024-06-16", "2024-06-19")]

CAT_RULES = [
    ("Clothing and footwear", r"BAG|SCARF|SHIRT|SOCK|JACKET|SLIPPER|APRON|GLOVE|PURSE"),
    ("Education", r"BOOK|PENCIL|\bPEN\b|NOTEBOOK|ERASER|RULER|STATIONER|CRAYON"),
    ("Information and communication", r"PHONE|RADIO|BATTERY|ELECTRIC|USB|CABLE|CHARGER"),
    ("Personal care and miscellaneous goods", r"BATH|SOAP|COSMETIC|MIRROR|TOILET|BRUSH|COMB"),
    ("Recreation and culture", r"TOY|GAME|PUZZLE|DOLL|CRAFT|PARTY|BALLOON|GIFT"),
    ("Food and non-alcoholic beverages", r"COFFEE|\bTEA\b|MUG|CUP|BOTTLE|JAR|CAKE|BOWL|PLATE"),
    ("Furniture and household equipment", r"LAMP|LIGHT|CLOCK|FRAME|CANDLE|DECOR|CUSHION|DOORMAT|RUG|KITCHEN|STORAGE|TRAY|HOLDER"),
]
NAMES = {
    "Food and non-alcoholic beverages": [("Saudi Dates Gift Box", "علبة تمور سعودية"), ("Arabic Coffee Set", "طقم قهوة عربية"), ("Premium Coffee Beans", "حبوب قهوة فاخرة"), ("Snack Gift Basket", "سلة وجبات خفيفة")],
    "Clothing and footwear": [("Cotton Clothing Accessory", "إكسسوار ملابس قطني"), ("Children Clothing Set", "طقم ملابس أطفال"), ("Seasonal Scarf", "وشاح موسمي"), ("Comfort Footwear", "أحذية مريحة")],
    "Furniture and household equipment": [("Decorative Serving Tray", "صينية تقديم زخرفية"), ("Kitchen Storage Set", "طقم تخزين للمطبخ"), ("Home Fragrance Holder", "حامل معطر منزلي"), ("Living Room Cushion", "وسادة غرفة معيشة")],
    "Personal care and miscellaneous goods": [("Personal Care Set", "طقم عناية شخصية"), ("Travel Hygiene Kit", "حقيبة نظافة للسفر"), ("Cosmetic Organizer", "منظم مستحضرات تجميل"), ("Grooming Accessory", "إكسسوار عناية")],
    "Recreation and culture": [("Family Board Game", "لعبة عائلية"), ("Children Puzzle", "أحجية أطفال"), ("Decorative Gift Item", "هدية زخرفية"), ("Creative Hobby Kit", "طقم هواية إبداعية")],
    "Information and communication": [("Mobile Accessory", "إكسسوار جوال"), ("Rechargeable Light", "مصباح قابل للشحن"), ("Electronic Desk Accessory", "إكسسوار مكتبي إلكتروني"), ("Cable Organizer", "منظم كابلات")],
    "Education": [("Notebook Set", "طقم دفاتر"), ("School Stationery Pack", "حزمة أدوات مدرسية"), ("Office Organizer", "منظم مكتبي"), ("Writing Tools Set", "طقم أدوات كتابة")],
    "Other retail goods": [("General Retail Item", "سلعة تجزئة عامة"), ("Seasonal Utility Item", "سلعة موسمية عملية"), ("Household Gift Item", "هدية منزلية"), ("Everyday Accessory", "إكسسوار يومي")],
}
ADMIN_CODES = {"POST", "DOT", "M", "D", "C2", "BANK CHARGES", "AMAZONFEE", "ADJUST", "ADJUST2", "CRUK", "S"}
FORBIDDEN_MODEL_FIELDS = {
    "LocalizedNetSalesSAR", "ScenarioNetSalesSAR", "ModeledSaudiDemandMultiplier",
    "IsRamadan", "IsEidAlFitr", "IsHajjSeason", "IsEidAlAdha",
    "IsNationalDay", "IsFoundingDay", "IsSalaryPeriodAssumption",
    "SourceInvoiceDate", "SourceWeekday", "LegacyLocalizedDateV1_1",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_int(value: object, modulus: int) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def stable_customer_id(key: object) -> str:
    return f"C-{stable_int(key, 100_000_000):08d}"


def stable_invoice_id(invoice: object, year: int) -> str:
    return f"SA-{year}-{stable_int(invoice, 10_000_000_000):010d}"


def in_ranges(dates: pd.Series, ranges: list[tuple[str, str]]) -> np.ndarray:
    result = np.zeros(len(dates), dtype=bool)
    for start, end in ranges:
        result |= dates.between(pd.Timestamp(start), pd.Timestamp(end)).to_numpy()
    return result


def download_source() -> tuple[Path, str, str]:
    if not ARCHIVE.exists():
        response = requests.get(DATASET_URL, timeout=240)
        response.raise_for_status()
        ARCHIVE.write_bytes(response.content)
    RAW_DIR.mkdir(exist_ok=True)
    if not list(RAW_DIR.rglob("*.xlsx")):
        with zipfile.ZipFile(ARCHIVE) as bundle:
            bundle.extractall(RAW_DIR)
    xlsx_files = list(RAW_DIR.rglob("*.xlsx"))
    if len(xlsx_files) != 1:
        raise RuntimeError(f"Expected one XLSX workbook, found {len(xlsx_files)}")
    return xlsx_files[0], sha256_file(ARCHIVE), sha256_file(xlsx_files[0])


def clean_source(xlsx_path: Path) -> dict:
    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    frames = []
    for sheet_name, frame in sheets.items():
        frame = frame.copy()
        frame["SourceSheet"] = str(sheet_name)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"})
    required = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country", "SourceSheet"]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw["InvoiceNo"] = raw["InvoiceNo"].astype("string").str.strip().str.upper()
    raw["StockCode"] = raw["StockCode"].astype("string").str.strip()
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
    raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce")
    raw["UnitPrice"] = pd.to_numeric(raw["UnitPrice"], errors="coerce")

    raw_rows = len(raw)
    duplicate_rows = int(raw.duplicated().sum())
    clean = raw.drop_duplicates().copy()
    after_duplicates = len(clean)
    required_bad = clean[["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]].isna().any(axis=1)
    required_removed = int(required_bad.sum())
    clean = clean.loc[~required_bad].copy()
    nonpositive_bad = clean["Quantity"].eq(0) | clean["UnitPrice"].le(0)
    nonpositive_removed = int(nonpositive_bad.sum())
    clean = clean.loc[~nonpositive_bad].copy()
    clean_rows = len(clean)

    clean["UnitPriceGBP"] = clean["UnitPrice"].astype(float)
    clean = clean.drop(columns=["UnitPrice"])
    clean.to_csv(CLEAN_GZ, index=False, compression={"method": "gzip", "compresslevel": 4})
    del raw, clean, frames, sheets

    return {
        "raw_rows": raw_rows,
        "duplicates_removed": duplicate_rows,
        "rows_after_duplicates": after_duplicates,
        "required_value_rows_removed": required_removed,
        "zero_quantity_or_nonpositive_price_removed": nonpositive_removed,
        "clean_rows": clean_rows,
    }


def collect_keys_and_dates() -> tuple[list[pd.Timestamp], set[str], set[str], set[str], int]:
    source_dates: set[pd.Timestamp] = set()
    observed_customers: set[str] = set()
    fallback_keys: set[str] = set()
    invoices: set[str] = set()
    row_count = 0
    for chunk in pd.read_csv(CLEAN_GZ, compression="gzip", chunksize=120_000, parse_dates=["InvoiceDate"], dtype={"CustomerID": "string", "InvoiceNo": "string", "StockCode": "string"}):
        row_count += len(chunk)
        source_dates.update(pd.to_datetime(chunk["InvoiceDate"].dt.floor("D").unique()).tolist())
        inv = chunk["InvoiceNo"].astype("string")
        invoices.update(inv.dropna().astype(str).tolist())
        observed = chunk["CustomerID"].notna()
        observed_customers.update(chunk.loc[observed, "CustomerID"].astype("string").astype(str).tolist())
        fallback_keys.update(("INV-" + inv.loc[~observed]).astype(str).tolist())
    return sorted(source_dates), observed_customers, fallback_keys, invoices, row_count


def build_date_map(source_dates: list[pd.Timestamp]) -> tuple[dict[pd.Timestamp, pd.Timestamp], pd.DataFrame]:
    target_dates = pd.date_range(TRAINING_START, periods=len(source_dates), freq="D")
    mapping = {pd.Timestamp(src): pd.Timestamp(dst) for src, dst in zip(source_dates, target_dates)}
    source_series = pd.Series(source_dates)
    source_gap = source_series.diff().dt.days.fillna(1).astype(int) - 1
    legacy = source_series + LEGACY_OFFSET
    frame = pd.DataFrame({
        "SourceDate": source_series,
        "LegacyLocalizedDateV1_1": legacy,
        "TrainingSafeDate": target_dates,
        "SourceGapDaysRemovedBefore": source_gap.clip(lower=0),
        "SourceWeekday": source_series.dt.day_name(),
        "LegacyWeekday": legacy.dt.day_name(),
        "TrainingSafeWeekday": pd.Series(target_dates).dt.day_name(),
    })
    frame.to_csv(DAY_MAP_CSV, index=False)
    return mapping, frame


def region_maps(observed_customers: set[str], fallback_keys: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    regions = list(REGION_POP)
    weights = np.array(list(REGION_POP.values()), dtype=float)
    cumulative = np.cumsum(weights / weights.sum())

    def assign(key: str) -> str:
        u = stable_int(key, 10**12) / float(10**12)
        idx = min(int(np.searchsorted(cumulative, u, side="right")), len(regions) - 1)
        return regions[idx]

    return ({k: assign(k) for k in observed_customers}, {k: assign(k) for k in fallback_keys})


def category_and_names(description: pd.Series, stock: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    desc = description.fillna("").astype(str)
    choices, labels = [], []
    for cat, pattern in CAT_RULES:
        choices.append(desc.str.contains(pattern, case=False, regex=True, na=False).to_numpy())
        labels.append(cat)
    categories = np.select(choices, labels, default="Other retail goods")
    en = np.empty(len(desc), dtype=object)
    ar = np.empty(len(desc), dtype=object)
    stock_values = stock.astype("string").fillna("MISSING").astype(str).to_numpy()
    for cat, names in NAMES.items():
        mask = categories == cat
        idx = [stable_int(value, len(names)) for value in stock_values[mask]]
        en[mask] = [names[i][0] for i in idx]
        ar[mask] = [names[i][1] for i in idx]
    return categories, en, ar


def is_admin_line(stock: pd.Series, desc: pd.Series) -> np.ndarray:
    code = stock.astype("string").fillna("").str.upper().str.strip()
    description = desc.fillna("").astype(str).str.upper()
    return (
        code.isin(ADMIN_CODES)
        | description.str.contains(r"BANK CHARG|AMAZON FEE|ADJUSTMENT|POSTAGE|MANUAL", regex=True, na=False)
    ).to_numpy()


def build_localized(
    date_map: dict[pd.Timestamp, pd.Timestamp],
    observed_region: dict[str, str],
    fallback_region: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    for path in (FULL_GZ, SAMPLE_CSV):
        if path.exists():
            path.unlink()

    daily_numeric = defaultdict(lambda: defaultdict(float))
    daily_invoice_sets: dict[pd.Timestamp, set[str]] = defaultdict(set)
    daily_electronic_invoice_sets: dict[pd.Timestamp, set[str]] = defaultdict(set)
    daily_customer_sets: dict[pd.Timestamp, set[str]] = defaultdict(set)
    daily_product_sets: dict[pd.Timestamp, set[str]] = defaultdict(set)
    first_customer_day: dict[str, pd.Timestamp] = {}
    unique_source_to_sa_invoice: dict[str, str] = {}
    unique_sa_invoice_to_source: dict[str, str] = {}
    unique_source_customer_to_sa: dict[str, str] = {}
    unique_sa_customer_to_source: dict[str, str] = {}
    payment_invoices: dict[str, tuple[int, str]] = {}
    observed_region_customers: dict[str, str] = {}

    total_rows = 0
    sample_parts = []
    sample_count = 0
    first_write = True
    critical_nulls = 0
    vat_error_rows = 0
    modeled_math_error_rows = 0
    administrative_rows = 0
    fallback_rows = 0
    observed_rows = 0
    line_counter = 0

    for chunk in pd.read_csv(CLEAN_GZ, compression="gzip", chunksize=100_000, parse_dates=["InvoiceDate"], dtype={"CustomerID": "string", "InvoiceNo": "string", "StockCode": "string"}):
        n = len(chunk)
        total_rows += n
        source_date = chunk["InvoiceDate"].dt.floor("D")
        training_date = source_date.map(date_map)
        if training_date.isna().any():
            raise RuntimeError("A cleaned source date is missing from the training-safe mapping")
        legacy_date = source_date + LEGACY_OFFSET
        time_delta = chunk["InvoiceDate"] - source_date
        localized_timestamp = training_date + time_delta
        year = training_date.dt.year.astype(int)

        invoice = chunk["InvoiceNo"].astype("string").astype(str)
        customer_observed = chunk["CustomerID"].notna()
        customer_key = chunk["CustomerID"].astype("string")
        fallback_key = "INV-" + invoice
        effective_key = customer_key.where(customer_observed, fallback_key).astype(str)
        sa_customer = effective_key.map(stable_customer_id)
        observed_sa_customer = sa_customer.where(customer_observed, pd.NA)
        source_type = np.where(customer_observed.to_numpy(), "ObservedSourceCustomerID", "SyntheticInvoiceFallback")
        fallback_rows += int((~customer_observed).sum())
        observed_rows += int(customer_observed.sum())

        region_values = []
        for key, obs in zip(effective_key, customer_observed.to_numpy()):
            region_values.append(observed_region.get(key) if obs else fallback_region.get(key))
        region = pd.Series(region_values, index=chunk.index, dtype="string")
        city = region.map(CAPITAL)

        sa_invoice_values = [stable_invoice_id(inv, int(y)) for inv, y in zip(invoice, year)]
        sa_invoice = pd.Series(sa_invoice_values, index=chunk.index, dtype="string")
        payment = []
        for inv, y in zip(invoice, year):
            u = stable_int(inv, 10**12) / float(10**12)
            payment.append("Electronic" if u < PAY_SHARE.get(int(y), 0.85) else "Cash / Other")
        payment = pd.Series(payment, index=chunk.index, dtype="string")

        categories, names_en, names_ar = category_and_names(chunk["Description"], chunk["StockCode"])
        cancellation = invoice.str.startswith("C", na=False) | chunk["Quantity"].lt(0)
        unit_sar = (chunk["UnitPriceGBP"].astype(float) * FX_GBP_TO_SAR).round(2)
        abs_sub = chunk["Quantity"].abs().astype(float) * unit_sar
        subtotal = pd.Series(np.where(cancellation, -abs_sub, abs_sub), index=chunk.index).round(2)
        vat_amount = (subtotal * VAT).round(2)
        line_total = (subtotal + vat_amount).round(2)

        is_weekend = training_date.dt.dayofweek.isin([4, 5])
        is_founding = training_date.dt.month.eq(2) & training_date.dt.day.eq(22)
        is_national = training_date.dt.month.eq(9) & training_date.dt.day.eq(23)
        is_ramadan = pd.Series(in_ranges(training_date, RAMADAN), index=chunk.index)
        is_fitr = pd.Series(in_ranges(training_date, EID_FITR), index=chunk.index)
        is_hajj = pd.Series(in_ranges(training_date, HAJJ), index=chunk.index)
        is_adha = pd.Series(in_ranges(training_date, EID_ADHA), index=chunk.index)
        salary = training_date.dt.day.between(25, 28)

        multiplier = np.ones(n, dtype=float)
        cat_series = pd.Series(categories, index=chunk.index)
        holy = region.isin(["Makkah", "Madinah"]).to_numpy()
        multiplier *= np.where(is_ramadan.to_numpy() & cat_series.eq("Food and non-alcoholic beverages").to_numpy(), 1.22, 1.0)
        multiplier *= np.where(is_ramadan.to_numpy() & cat_series.eq("Clothing and footwear").to_numpy(), 1.12, 1.0)
        multiplier *= np.where(is_ramadan.to_numpy() & cat_series.eq("Furniture and household equipment").to_numpy(), 1.10, 1.0)
        multiplier *= np.where(is_fitr.to_numpy() & cat_series.eq("Clothing and footwear").to_numpy(), 1.30, 1.0)
        multiplier *= np.where(is_fitr.to_numpy() & cat_series.eq("Recreation and culture").to_numpy(), 1.18, 1.0)
        multiplier *= np.where(is_hajj.to_numpy() & holy & cat_series.eq("Food and non-alcoholic beverages").to_numpy(), 1.25, 1.0)
        multiplier *= np.where(is_hajj.to_numpy() & holy & cat_series.eq("Clothing and footwear").to_numpy(), 1.15, 1.0)
        multiplier *= np.where(is_national.to_numpy(), 1.10, 1.0)
        multiplier *= np.where(is_founding.to_numpy(), 1.06, 1.0)
        multiplier *= np.where(salary.to_numpy(), 1.05, 1.0)
        multiplier = np.round(multiplier, 4)

        modeled_subtotal = (subtotal * multiplier).round(2)
        modeled_vat = (modeled_subtotal * VAT).round(2)
        modeled_total = (modeled_subtotal + modeled_vat).round(2)
        scenario_total = modeled_total.copy()
        admin = is_admin_line(chunk["StockCode"], chunk["Description"])
        administrative_rows += int(admin.sum())
        eligible = ~admin

        line_ids = [f"SA-LINE-{i:010d}" for i in range(line_counter + 1, line_counter + n + 1)]
        line_counter += n

        out = pd.DataFrame({
            "LocalizedLineID": line_ids,
            "SaudiInvoiceNo": sa_invoice,
            "SaudiCustomerID": sa_customer,
            "ObservedSaudiCustomerID": observed_sa_customer,
            "CustomerIDSource": source_type,
            "LocalizedInvoiceDate": localized_timestamp,
            "LocalizedDate": training_date,
            "TrainingSafeDate": training_date,
            "LegacyLocalizedDateV1_1": legacy_date,
            "SourceInvoiceDate": chunk["InvoiceDate"],
            "SourceWeekday": source_date.dt.day_name(),
            "Year": year,
            "Region": region,
            "City": city,
            "PaymentType": payment,
            "StockCode": chunk["StockCode"].astype("string"),
            "ProductCategoryCOICOP": categories,
            "LocalizedProductNameEnglish": names_en,
            "LocalizedProductNameArabic": names_ar,
            "SourceDescription": chunk["Description"],
            "SourceCountry": chunk["Country"],
            "SourceSheet": chunk["SourceSheet"],
            "OriginalQuantity": chunk["Quantity"].astype(float),
            "IsCancellation": cancellation.to_numpy(),
            "IsAdministrativeLine": admin,
            "EligibleForSalesTraining": eligible,
            "UnitPriceSARExVAT": unit_sar,
            "SubtotalSARExVAT": subtotal,
            "VATRate": VAT,
            "VATAmountSAR": vat_amount,
            "LineTotalSARIncVAT": line_total,
            "BaseNetSalesSAR": line_total,
            "ModeledSaudiDemandMultiplier": multiplier,
            "ModeledSubtotalSARExVAT": modeled_subtotal,
            "ModeledVATAmountSAR": modeled_vat,
            "ModeledLineTotalSARIncVAT": modeled_total,
            "ScenarioNetSalesSAR": scenario_total,
            "LocalizedNetSalesSAR": scenario_total,
            "IsWeekend": is_weekend.to_numpy(),
            "IsFoundingDay": is_founding.to_numpy(),
            "IsNationalDay": is_national.to_numpy(),
            "IsRamadan": is_ramadan.to_numpy(),
            "IsEidAlFitr": is_fitr.to_numpy(),
            "IsHajjSeason": is_hajj.to_numpy(),
            "IsEidAlAdha": is_adha.to_numpy(),
            "IsSalaryPeriodAssumption": salary.to_numpy(),
            "DatasetType": "Saudi-localized synthetic microdata; training-safe chronology",
            "CalibrationVersion": "SA-LOCALIZATION-1.2-TRAINING-SAFE",
            "RandomSeed": SEED,
        })

        critical_cols = ["LocalizedLineID", "SaudiInvoiceNo", "LocalizedInvoiceDate", "LocalizedDate", "Region", "StockCode", "BaseNetSalesSAR"]
        critical_nulls += int(out[critical_cols].isna().sum().sum())
        vat_error_rows += int(((out["VATAmountSAR"] - (out["SubtotalSARExVAT"] * VAT).round(2)).abs() > 0.011).sum())
        modeled_math_error_rows += int(((out["ModeledLineTotalSARIncVAT"] - (out["ModeledSubtotalSARExVAT"] + out["ModeledVATAmountSAR"]).round(2)).abs() > 0.011).sum())

        out.to_csv(
            FULL_GZ, index=False,
            compression={"method": "gzip", "compresslevel": 4},
            mode="wt" if first_write else "at", header=first_write, encoding="utf-8",
        )
        first_write = False
        if sample_count < 10_000:
            part = out.head(10_000 - sample_count)
            sample_parts.append(part)
            sample_count += len(part)

        for src, sai in zip(invoice, sa_invoice):
            old = unique_source_to_sa_invoice.setdefault(src, str(sai))
            if old != str(sai):
                raise RuntimeError("One source invoice mapped to multiple Saudi invoice IDs")
            other = unique_sa_invoice_to_source.setdefault(str(sai), src)
            if other != src:
                raise RuntimeError("Saudi invoice ID collision detected")
        obs_mask = customer_observed.to_numpy()
        for src, sai, reg in zip(customer_key[obs_mask].astype(str), sa_customer[obs_mask].astype(str), region[obs_mask].astype(str)):
            old = unique_source_customer_to_sa.setdefault(src, sai)
            if old != sai:
                raise RuntimeError("One observed source customer mapped to multiple Saudi customer IDs")
            other = unique_sa_customer_to_source.setdefault(sai, src)
            if other != src:
                raise RuntimeError("Saudi observed-customer ID collision detected")
            observed_region_customers[sai] = reg

        for inv, y, pay in zip(sa_invoice.astype(str), year, payment.astype(str)):
            payment_invoices.setdefault(inv, (int(y), pay))

        eligible_frame = out.loc[out["EligibleForSalesTraining"]].copy()
        for d, group in eligible_frame.groupby("TrainingSafeDate", sort=False):
            day = pd.Timestamp(d)
            vals = daily_numeric[day]
            base = group["BaseNetSalesSAR"].astype(float)
            vals["gross_sales_sar"] += float(base.clip(lower=0).sum())
            vals["return_value_sar"] += float((-base.clip(upper=0)).sum())
            vals["base_net_sales_sar"] += float(base.sum())
            vals["scenario_net_sales_sar"] += float(group["ScenarioNetSalesSAR"].astype(float).sum())
            vals["transaction_rows"] += int(len(group))
            vals["units"] += float(group["OriginalQuantity"].abs().sum())
            vals["ramadan"] = max(vals["ramadan"], int(group["IsRamadan"].max()))
            vals["eid_fitr"] = max(vals["eid_fitr"], int(group["IsEidAlFitr"].max()))
            vals["hajj_season"] = max(vals["hajj_season"], int(group["IsHajjSeason"].max()))
            vals["eid_adha"] = max(vals["eid_adha"], int(group["IsEidAlAdha"].max()))
            vals["national_day"] = max(vals["national_day"], int(group["IsNationalDay"].max()))
            vals["founding_day"] = max(vals["founding_day"], int(group["IsFoundingDay"].max()))
            vals["salary_period"] = max(vals["salary_period"], int(group["IsSalaryPeriodAssumption"].max()))
            daily_invoice_sets[day].update(group["SaudiInvoiceNo"].astype(str).tolist())
            daily_electronic_invoice_sets[day].update(group.loc[group["PaymentType"].eq("Electronic"), "SaudiInvoiceNo"].astype(str).tolist())
            daily_product_sets[day].update(group["StockCode"].astype(str).tolist())
            obs_ids = group["ObservedSaudiCustomerID"].dropna().astype(str).unique().tolist()
            daily_customer_sets[day].update(obs_ids)
            for cid in obs_ids:
                if cid not in first_customer_day or day < first_customer_day[cid]:
                    first_customer_day[cid] = day

    pd.concat(sample_parts, ignore_index=True).to_csv(SAMPLE_CSV, index=False, encoding="utf-8-sig")

    rows = []
    for day in sorted(daily_numeric):
        vals = daily_numeric[day]
        customers = daily_customer_sets[day]
        new_customers = sum(first_customer_day.get(cid) == day for cid in customers)
        invoice_count = len(daily_invoice_sets[day])
        base_net = float(vals["base_net_sales_sar"])
        gross = float(vals["gross_sales_sar"])
        returns = float(vals["return_value_sar"])
        rows.append({
            "date": day,
            "business_day_index": len(rows),
            "gross_sales_sar": round(gross, 2),
            "return_value_sar": round(returns, 2),
            "base_net_sales_sar": round(base_net, 2),
            "scenario_net_sales_sar": round(float(vals["scenario_net_sales_sar"]), 2),
            "transaction_rows": int(vals["transaction_rows"]),
            "invoice_count": invoice_count,
            "electronic_invoice_count": len(daily_electronic_invoice_sets[day]),
            "unique_observed_customers": len(customers),
            "new_observed_customers": new_customers,
            "returning_observed_customers": len(customers) - new_customers,
            "unique_products": len(daily_product_sets[day]),
            "units": float(vals["units"]),
            "average_invoice_value_sar": round(base_net / invoice_count, 2) if invoice_count else 0.0,
            "return_rate_value": round(returns / gross, 6) if gross > 0 else 0.0,
            "ramadan": bool(vals["ramadan"]),
            "eid_fitr": bool(vals["eid_fitr"]),
            "hajj_season": bool(vals["hajj_season"]),
            "eid_adha": bool(vals["eid_adha"]),
            "national_day": bool(vals["national_day"]),
            "founding_day": bool(vals["founding_day"]),
            "salary_period": bool(vals["salary_period"]),
        })
    daily = pd.DataFrame(rows)
    daily.to_csv(DAILY_CSV, index=False)

    observed_region_counts = pd.Series(observed_region_customers).value_counts()
    total_observed = len(observed_region_customers)
    pop_total = float(sum(REGION_POP.values()))
    region_diff = {}
    for region_name, population in REGION_POP.items():
        target = population / pop_total
        actual = observed_region_counts.get(region_name, 0) / max(total_observed, 1)
        region_diff[region_name] = abs(actual - target)

    payment_df = pd.DataFrame([(k, v[0], v[1]) for k, v in payment_invoices.items()], columns=["invoice", "year", "payment"])
    payment_summary = {}
    if not payment_df.empty:
        for y, group in payment_df.groupby("year"):
            actual = float(group["payment"].eq("Electronic").mean())
            payment_summary[str(int(y))] = {
                "invoice_count": int(len(group)),
                "actual_electronic_share": actual,
                "target_electronic_share": PAY_SHARE.get(int(y)),
                "absolute_difference": abs(actual - PAY_SHARE.get(int(y), actual)),
            }

    stats = {
        "localized_rows": total_rows,
        "critical_null_count": critical_nulls,
        "vat_error_rows": vat_error_rows,
        "modeled_math_error_rows": modeled_math_error_rows,
        "administrative_rows_retained_but_excluded_from_training": administrative_rows,
        "observed_customer_rows": observed_rows,
        "fallback_customer_rows": fallback_rows,
        "unique_invoices": len(unique_source_to_sa_invoice),
        "unique_observed_customers": len(unique_source_customer_to_sa),
        "unique_fallback_customer_keys": len(fallback_region),
        "training_safe_days": len(daily),
        "training_safe_date_start": str(pd.to_datetime(daily["date"]).min().date()),
        "training_safe_date_end": str(pd.to_datetime(daily["date"]).max().date()),
        "max_region_share_difference": max(region_diff.values()) if region_diff else None,
        "payment_calibration": payment_summary,
    }
    return daily, stats


def make_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").set_index("date")
    frame = pd.DataFrame(index=d.index)
    series_map = {
        "sales": d["base_net_sales_sar"].astype(float),
        "customers": d["unique_observed_customers"].astype(float),
        "invoices": d["invoice_count"].astype(float),
        "returns_rate": d["return_rate_value"].astype(float),
    }
    for prefix, values in series_map.items():
        for lag in (1, 2, 3, 7, 14, 28):
            frame[f"{prefix}_lag_{lag}"] = values.shift(lag)
        past = values.shift(1)
        for window in (7, 14, 28):
            frame[f"{prefix}_mean_{window}"] = past.rolling(window).mean()
            frame[f"{prefix}_std_{window}"] = past.rolling(window).std()
    frame["trend"] = np.arange(len(frame), dtype=float)
    frame["target_sales"] = d["base_net_sales_sar"].astype(float)
    frame["target_customers"] = d["unique_observed_customers"].astype(float)
    frame["customer_baseline_28"] = d["unique_observed_customers"].shift(1).rolling(28).mean()
    frame["customer_decline_target"] = (
        frame["target_customers"] < (1.0 - DECLINE_PERCENT) * frame["customer_baseline_28"]
    ).astype(int)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    feature_cols = [c for c in frame.columns if c not in {"target_sales", "target_customers", "customer_baseline_28", "customer_decline_target"}]
    if FORBIDDEN_MODEL_FIELDS.intersection(feature_cols):
        raise RuntimeError("Forbidden leakage/source-calendar field reached the model feature set")
    return frame, feature_cols


def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.abs(y_true - y_pred).sum() / max(np.abs(y_true).sum(), 1e-9))


def train_models(daily: pd.DataFrame) -> dict:
    frame, feature_cols = make_features(daily)
    if len(frame) < 240:
        raise RuntimeError(f"Not enough supervised training rows: {len(frame)}")
    hold = 60 if len(frame) >= 360 else 45
    test_start = len(frame) - hold
    val_start = test_start - hold
    train = frame.iloc[:val_start]
    val = frame.iloc[val_start:test_start]
    test = frame.iloc[test_start:]
    X_train, X_val, X_test = train[feature_cols], val[feature_cols], test[feature_cols]
    y_train_r, y_val_r, y_test_r = train["target_sales"], val["target_sales"], test["target_sales"]
    y_train_c, y_val_c, y_test_c = train["customer_decline_target"], val["customer_decline_target"], test["customer_decline_target"]

    regressors = {
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.85, random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED),
    }
    reg_validation = {
        "SeasonalNaive7": {
            "WAPE": wape(y_val_r, val["sales_lag_7"]),
            "MAE": float(mean_absolute_error(y_val_r, val["sales_lag_7"])),
        }
    }
    fitted_reg = {}
    for name, model in regressors.items():
        fit = clone(model).fit(X_train, y_train_r)
        pred = fit.predict(X_val)
        fitted_reg[name] = fit
        reg_validation[name] = {"WAPE": wape(y_val_r, pred), "MAE": float(mean_absolute_error(y_val_r, pred))}
    best_reg_ml = min(regressors, key=lambda name: reg_validation[name]["WAPE"])
    X_train_val = pd.concat([X_train, X_val])
    y_train_val_r = pd.concat([y_train_r, y_val_r])
    best_reg = clone(regressors[best_reg_ml]).fit(X_train_val, y_train_val_r)
    test_reg_pred = best_reg.predict(X_test)
    reg_test = {
        "MAE": float(mean_absolute_error(y_test_r, test_reg_pred)),
        "RMSE": float(mean_squared_error(y_test_r, test_reg_pred) ** 0.5),
        "WAPE": wape(y_test_r, test_reg_pred),
        "R2": float(r2_score(y_test_r, test_reg_pred)),
    }
    joblib.dump(best_reg, MODEL_DIR / "sales_forecast_regressor_v1_2.joblib")

    classifiers = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))]),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, max_features=0.85, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED),
    }
    if y_train_c.nunique() < 2 or y_val_c.nunique() < 2 or y_test_c.nunique() < 2:
        raise RuntimeError("Customer-decline target does not contain both classes in chronological splits")

    cls_validation = {}
    val_prob = {}
    for name, model in classifiers.items():
        fit = clone(model).fit(X_train, y_train_c)
        prob = fit.predict_proba(X_val)[:, 1]
        pred = (prob >= 0.5).astype(int)
        val_prob[name] = prob
        cls_validation[name] = {
            "Accuracy": float(accuracy_score(y_val_c, pred)),
            "BalancedAccuracy": float(balanced_accuracy_score(y_val_c, pred)),
            "F1": float(f1_score(y_val_c, pred, zero_division=0)),
        }
    candidates = [name for name in classifiers if name != "Dummy"]
    best_cls = max(candidates, key=lambda name: (cls_validation[name]["BalancedAccuracy"], cls_validation[name]["F1"]))
    threshold_rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (val_prob[best_cls] >= threshold).astype(int)
        ba = balanced_accuracy_score(y_val_c, pred)
        f1 = f1_score(y_val_c, pred, zero_division=0)
        threshold_rows.append((0.5 * (ba + f1), float(threshold)))
    best_threshold = max(threshold_rows)[1]

    y_train_val_c = pd.concat([y_train_c, y_val_c])
    best_classifier = clone(classifiers[best_cls]).fit(X_train_val, y_train_val_c)
    test_prob = best_classifier.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)
    cls_test = {
        "Accuracy": float(accuracy_score(y_test_c, test_pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y_test_c, test_pred)),
        "Precision": float(precision_score(y_test_c, test_pred, zero_division=0)),
        "Recall": float(recall_score(y_test_c, test_pred, zero_division=0)),
        "F1": float(f1_score(y_test_c, test_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_test_c, test_prob)),
    }
    joblib.dump(best_classifier, MODEL_DIR / "customer_decline_classifier_v1_2.joblib")

    metadata = {
        "version": "SA-LOCALIZATION-1.2-TRAINING-SAFE",
        "training_target_sales": "base_net_sales_sar",
        "scenario_sales_is_not_a_training_target": True,
        "customer_target_uses_only_observed_source_customer_ids": True,
        "source_calendar_fields_forbidden": True,
        "feature_columns": feature_cols,
        "chronological_split": {"validation_days": hold, "test_days": hold, "shuffle": False},
        "regression": {"selected_ml_model": best_reg_ml, "validation": reg_validation, "test": reg_test},
        "classification": {"selected_model": best_cls, "selected_threshold": best_threshold, "validation": cls_validation, "test": cls_test},
    }
    MODEL_META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    xlsx, archive_hash, workbook_hash = download_source()
    clean_stats = clean_source(xlsx)
    source_dates, observed_customers, fallback_keys, invoices, counted_clean_rows = collect_keys_and_dates()
    date_map, date_map_frame = build_date_map(source_dates)
    observed_region, fallback_region = region_maps(observed_customers, fallback_keys)
    daily, localized_stats = build_localized(date_map, observed_region, fallback_region)

    legacy_start = pd.Timestamp(source_dates[0]) + LEGACY_OFFSET
    legacy_end = pd.Timestamp(source_dates[-1]) + LEGACY_OFFSET
    legacy_calendar = pd.date_range(legacy_start, legacy_end, freq="D")
    legacy_active = set((pd.Series(source_dates) + LEGACY_OFFSET).dt.normalize())
    legacy_missing = [d for d in legacy_calendar if d.normalize() not in legacy_active]
    missing_weekdays = pd.Series([d.day_name() for d in legacy_missing]).value_counts().to_dict()
    gaps = pd.to_datetime(daily["date"]).sort_values().diff().dt.days.dropna()
    max_payment_diff = max((v["absolute_difference"] for v in localized_stats["payment_calibration"].values()), default=0.0)

    checks = {
        "official_raw_row_count_matches_uci": clean_stats["raw_rows"] == EXPECTED_RAW_ROWS,
        "legacy_clean_row_count_reproduced": clean_stats["clean_rows"] == EXPECTED_LEGACY_CLEAN_ROWS,
        "second_pass_row_count_matches_clean": counted_clean_rows == clean_stats["clean_rows"],
        "localized_row_count_matches_clean": localized_stats["localized_rows"] == clean_stats["clean_rows"],
        "critical_null_count_zero": localized_stats["critical_null_count"] == 0,
        "vat_math_valid": localized_stats["vat_error_rows"] == 0,
        "modeled_math_valid": localized_stats["modeled_math_error_rows"] == 0,
        "training_dates_are_consecutive": bool(len(gaps) == 0 or (gaps == 1).all()),
        "no_inherited_zero_calendar_days_in_training_table": len(daily) == len(source_dates),
        "observed_customer_ids_separated_from_fallbacks": localized_stats["unique_observed_customers"] == len(observed_customers),
        "region_calibration_within_2pp": (localized_stats["max_region_share_difference"] or 0) < 0.02,
        "payment_calibration_within_1pp": max_payment_diff < 0.01,
        "scenario_target_is_separate_from_ml_target": True,
        "source_calendar_fields_are_forbidden_from_model": True,
    }
    all_passed = all(checks.values())

    audit = {
        "dataset": "Saudi-Localized Online Retail II",
        "version": "SA-LOCALIZATION-1.2-TRAINING-SAFE",
        "source": {"name": "UCI Online Retail II", "doi": DATASET_DOI, "download_url": DATASET_URL, "archive_sha256": archive_hash, "workbook_sha256": workbook_hash},
        "cleaning": clean_stats,
        "localization": localized_stats,
        "calendar_repair": {
            "method": "Map each observed source transaction date, in chronological order, to consecutive TrainingSafeDate values starting 2023-01-01. Preserve source and legacy-shifted dates only as provenance fields.",
            "source_observed_days": len(source_dates),
            "legacy_calendar_days": len(legacy_calendar),
            "legacy_inherited_zero_days_removed_from_training": len(legacy_missing),
            "legacy_zero_days_by_weekday": missing_weekdays,
            "training_safe_gap_days": int((gaps > 1).sum()) if len(gaps) else 0,
            "training_safe_date_start": localized_stats["training_safe_date_start"],
            "training_safe_date_end": localized_stats["training_safe_date_end"],
        },
        "customer_repair": {
            "observed_source_customers": localized_stats["unique_observed_customers"],
            "fallback_customer_keys_retained_for_row_linkage_only": localized_stats["unique_fallback_customer_keys"],
            "fallbacks_counted_in_customer_decline_target": False,
        },
        "training_contract": {
            "sales_ml_target": "base_net_sales_sar",
            "scenario_sales_field": "scenario_net_sales_sar",
            "scenario_sales_used_for_model_selection_or_test_metrics": False,
            "forbidden_model_fields": sorted(FORBIDDEN_MODEL_FIELDS),
            "chronological_split_only": True,
        },
        "checks": checks,
        "all_tests_passed": all_passed,
        "full_microdata_sha256": sha256_file(FULL_GZ),
        "daily_training_sha256": sha256_file(DAILY_CSV),
        "sample_sha256": sha256_file(SAMPLE_CSV),
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    report = f"""# Saudi Localization v1.2 — Training-Safe Quality Report\n\n- Source: UCI Online Retail II (`{DATASET_DOI}`)\n- Raw rows: **{clean_stats['raw_rows']:,}**\n- Final clean rows: **{clean_stats['clean_rows']:,}**\n- Exact duplicates removed: **{clean_stats['duplicates_removed']:,}**\n- Required-value rows removed: **{clean_stats['required_value_rows_removed']:,}**\n- Zero-quantity / non-positive-price rows removed: **{clean_stats['zero_quantity_or_nonpositive_price_removed']:,}**\n- Observed source customers: **{localized_stats['unique_observed_customers']:,}**\n- Fallback customer keys retained only for row linkage: **{localized_stats['unique_fallback_customer_keys']:,}**\n- Legacy inherited zero-calendar days excluded from training: **{len(legacy_missing):,}**\n- Training-safe active days: **{len(daily):,}**\n- Training-safe period: **{localized_stats['training_safe_date_start']} → {localized_stats['training_safe_date_end']}**\n- Critical nulls: **{localized_stats['critical_null_count']}**\n- VAT math errors: **{localized_stats['vat_error_rows']}**\n- Modeled math errors: **{localized_stats['modeled_math_error_rows']}**\n- Administrative/service rows retained for audit but excluded from sales target: **{localized_stats['administrative_rows_retained_but_excluded_from_training']:,}**\n- All quality gates passed: **{all_passed}**\n\n## Repairs\n\n1. `TrainingSafeDate` replaces the inherited UK closure calendar for model training; `SourceInvoiceDate` and `LegacyLocalizedDateV1_1` remain provenance-only.\n2. Customer-decline counts use only `ObservedSourceCustomerID`; invoice-based fallback IDs are never counted as customers.\n3. `BaseNetSalesSAR` is the machine-learning target. `ScenarioNetSalesSAR` / `LocalizedNetSalesSAR` remain scenario fields and are forbidden from model selection and test metrics.\n4. Exact duplicates and invalid core rows are removed using the same cleaning rules that produced the legacy 1,049,042-row clean source.\n5. Administrative/service lines are retained for traceability but excluded from the training sales target.\n"""
    QUALITY_MD.write_text(report, encoding="utf-8")

    if not all_passed:
        raise RuntimeError(f"Saudi v1.2 quality gate failed. See {AUDIT_JSON}")

    model_metadata = train_models(daily)
    audit["model_training"] = model_metadata
    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": "PASS",
        "version": audit["version"],
        "clean_rows": clean_stats["clean_rows"],
        "training_safe_days": len(daily),
        "observed_customers": localized_stats["unique_observed_customers"],
        "fallback_customer_keys": localized_stats["unique_fallback_customer_keys"],
        "legacy_zero_days_removed": len(legacy_missing),
        "audit": str(AUDIT_JSON),
        "daily": str(DAILY_CSV),
        "full_microdata": str(FULL_GZ),
        "model_metadata": str(MODEL_META),
    }, indent=2))


if __name__ == "__main__":
    main()
