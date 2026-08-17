from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from typing import Iterable, Mapping, Any


@dataclass(frozen=True)
class ValidationGateResult:
    eligible: bool
    grade: str
    reasons: list[str]
    warnings: list[str]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in (
        "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S", "%d-%m-%Y",
    ):
        try:
            return datetime.strptime(text[:19], fmt).date()
        except ValueError:
            continue
    return None


def _pick_column(columns: set[str], aliases: Iterable[str]) -> str | None:
    lowered = {column.lower().strip(): column for column in columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def assess_external_merchant_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    minimum_span_days: int = 365,
    minimum_calendar_days: int = 300,
    minimum_transactions: int = 1000,
    maximum_missing_day_ratio: float = 0.10,
) -> ValidationGateResult:
    """Assess whether fresh merchant data is strong enough for blind validation.

    This gate deliberately evaluates data suitability only. It does not train or
    tune a model and therefore must run before any external-data model changes.
    """
    materialized = list(rows)
    if not materialized:
        return ValidationGateResult(
            eligible=False,
            grade="REJECT",
            reasons=["dataset_has_no_rows"],
            warnings=[],
            metrics={"row_count": 0},
        )

    columns = {str(key).strip() for row in materialized for key in row.keys()}
    date_col = _pick_column(columns, (
        "TRX DATE", "Invoice Date", "InvoiceDate", "Order_Date", "date", "sale_date",
    ))
    sales_col = _pick_column(columns, (
        "Net Amount", "Total Sales", "total_sales", "net_sales", "TOTAL AMOUNT", "total_amount",
    ))
    transaction_col = _pick_column(columns, (
        "TRX NUMBER", "Invoice ID", "InvoiceNo", "Order_ID", "transaction_number", "sale_id",
    ))
    customer_col = _pick_column(columns, (
        "CUSTOMER NUMBER", "CustomerID", "customer_id", "customer_key",
    ))
    product_col = _pick_column(columns, (
        "ITEM CODE", "StockCode", "product_id", "sku", "shoe_id",
    ))

    reasons: list[str] = []
    warnings: list[str] = []
    if date_col is None:
        reasons.append("missing_date_column")
    if sales_col is None:
        reasons.append("missing_sales_value_column")
    if transaction_col is None:
        reasons.append("missing_transaction_identifier")
    if date_col is None:
        return ValidationGateResult(False, "REJECT", reasons, warnings, {
            "row_count": len(materialized), "columns": sorted(columns),
        })

    parsed_dates = [_parse_date(row.get(date_col)) for row in materialized]
    valid_dates = [value for value in parsed_dates if value is not None]
    invalid_date_rows = len(parsed_dates) - len(valid_dates)
    if not valid_dates:
        reasons.append("no_parseable_dates")
        return ValidationGateResult(False, "REJECT", reasons, warnings, {
            "row_count": len(materialized), "invalid_date_rows": invalid_date_rows,
        })

    start, end = min(valid_dates), max(valid_dates)
    span_days = (end - start).days + 1
    distinct_days = len(set(valid_dates))
    missing_days = max(0, span_days - distinct_days)
    missing_day_ratio = missing_days / span_days if span_days else 1.0

    transaction_values = {
        str(row.get(transaction_col)).strip()
        for row in materialized
        if transaction_col and row.get(transaction_col) not in (None, "")
    }
    transaction_count = len(transaction_values) if transaction_col else 0

    if span_days < minimum_span_days:
        reasons.append("insufficient_time_span")
    if distinct_days < minimum_calendar_days:
        reasons.append("insufficient_distinct_calendar_days")
    if transaction_count < minimum_transactions:
        reasons.append("insufficient_unique_transactions")
    if missing_day_ratio > maximum_missing_day_ratio:
        reasons.append("excessive_calendar_gaps")
    if invalid_date_rows:
        ratio = invalid_date_rows / len(materialized)
        if ratio > 0.01:
            reasons.append("too_many_invalid_dates")
        else:
            warnings.append("some_invalid_dates")
    if customer_col is None:
        warnings.append("missing_customer_identifier_v18_may_be_limited")
    if product_col is None:
        warnings.append("missing_product_identifier_v18_may_be_limited")

    # Freshness/provenance cannot be proven from tabular rows alone. Keep this
    # explicit so a technically good file is never automatically called blind.
    warnings.append("provenance_and_freshness_must_be_verified_separately")

    eligible = not reasons
    grade = "ELIGIBLE_FOR_BLIND_EVALUATION" if eligible else "REJECT"
    return ValidationGateResult(
        eligible=eligible,
        grade=grade,
        reasons=reasons,
        warnings=warnings,
        metrics={
            "row_count": len(materialized),
            "date_column": date_col,
            "sales_column": sales_col,
            "transaction_column": transaction_col,
            "customer_column": customer_col,
            "product_column": product_col,
            "date_start": start.isoformat(),
            "date_end": end.isoformat(),
            "span_days": span_days,
            "distinct_calendar_days": distinct_days,
            "missing_calendar_days": missing_days,
            "missing_day_ratio": missing_day_ratio,
            "unique_transactions": transaction_count,
            "invalid_date_rows": invalid_date_rows,
            "thresholds": {
                "minimum_span_days": minimum_span_days,
                "minimum_calendar_days": minimum_calendar_days,
                "minimum_transactions": minimum_transactions,
                "maximum_missing_day_ratio": maximum_missing_day_ratio,
            },
        },
    )
