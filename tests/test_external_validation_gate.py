from __future__ import annotations

from datetime import date, timedelta

from app.services.external_validation_gate import assess_external_merchant_rows


def _rows(days: int, transactions_per_day: int = 4, *, skip_every: int | None = None):
    start = date(2025, 1, 1)
    rows = []
    tx = 0
    for i in range(days):
        if skip_every and i % skip_every == 0:
            continue
        day = start + timedelta(days=i)
        for j in range(transactions_per_day):
            tx += 1
            rows.append({
                "TRX DATE": day.isoformat(),
                "TRX NUMBER": f"TX-{tx}",
                "CUSTOMER NUMBER": f"C-{j % 20}",
                "ITEM CODE": f"SKU-{j % 50}",
                "Net Amount": 100 + j,
            })
    return rows


def test_rejects_short_redsea_like_window_even_with_many_transactions():
    result = assess_external_merchant_rows(_rows(123, transactions_per_day=25))
    assert result.eligible is False
    assert result.grade == "REJECT"
    assert "insufficient_time_span" in result.reasons
    assert "insufficient_distinct_calendar_days" in result.reasons


def test_accepts_long_dense_transaction_history_for_blind_evaluation_gate():
    result = assess_external_merchant_rows(_rows(400, transactions_per_day=4))
    assert result.eligible is True
    assert result.grade == "ELIGIBLE_FOR_BLIND_EVALUATION"
    assert result.metrics["span_days"] == 400
    assert result.metrics["distinct_calendar_days"] == 400
    assert result.metrics["unique_transactions"] == 1600
    assert result.metrics["missing_day_ratio"] == 0.0
    assert "provenance_and_freshness_must_be_verified_separately" in result.warnings


def test_rejects_large_calendar_gaps():
    result = assess_external_merchant_rows(
        _rows(420, transactions_per_day=4, skip_every=5),
        minimum_calendar_days=300,
    )
    assert result.eligible is False
    assert "excessive_calendar_gaps" in result.reasons
    assert result.metrics["missing_day_ratio"] > 0.10


def test_rejects_missing_transaction_identifier():
    rows = [
        {"date": (date(2025, 1, 1) + timedelta(days=i)).isoformat(), "net_sales": 500.0}
        for i in range(400)
    ]
    result = assess_external_merchant_rows(rows)
    assert result.eligible is False
    assert "missing_transaction_identifier" in result.reasons
    assert "insufficient_unique_transactions" in result.reasons


def test_customer_and_product_are_warnings_not_point_forecast_blockers():
    start = date(2025, 1, 1)
    rows = []
    for i in range(400):
        for j in range(3):
            rows.append({
                "Invoice Date": (start + timedelta(days=i)).isoformat(),
                "Invoice ID": f"INV-{i}-{j}",
                "Total Sales": 250.0,
            })
    result = assess_external_merchant_rows(rows)
    assert result.eligible is True
    assert "missing_customer_identifier_v18_may_be_limited" in result.warnings
    assert "missing_product_identifier_v18_may_be_limited" in result.warnings
