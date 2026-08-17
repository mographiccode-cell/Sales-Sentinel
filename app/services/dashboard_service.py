from __future__ import annotations

from datetime import timedelta

from sqlalchemy import case, desc, func, select, text

from app.models import Alert, Forecast, ModelRun, Sale
from app.services.data_scope import preferred_sales_condition


def dashboard_summary(db, allowed_branch_ids: set[int] | None = None) -> dict:
    sales_condition, data_mode, _ = preferred_sales_condition(db)

    def scoped(stmt):
        stmt = stmt.where(sales_condition)
        return stmt.where(Sale.branch_id.in_(allowed_branch_ids)) if allowed_branch_ids else stmt

    total_records = int(db.scalar(scoped(select(func.count(Sale.id)))) or 0)
    max_date = db.scalar(scoped(select(func.max(Sale.sale_date))))
    if not max_date:
        return {
            "current_sales": 0,
            "previous_sales": 0,
            "change_percent": 0,
            "forecast_sales": 0,
            "decline_probability": 0,
            "active_alerts": 0,
            "transactions": 0,
            "active_customers": 0,
            "products": 0,
            "channels": 0,
            "returns": 0,
            "discounts": 0,
            "avg_transaction": 0,
            "series": [],
            "forecast_series": [],
            "latest_run": None,
            "top_alerts": [],
            "total_records": total_records,
            "data_end": None,
            "data_mode": data_mode,
        }

    current_start = max_date - timedelta(days=29)
    previous_start = current_start - timedelta(days=30)
    current_sales = float(
        db.scalar(scoped(select(func.sum(Sale.net_sales)).where(Sale.sale_date.between(current_start, max_date)))) or 0
    )
    previous_sales = float(
        db.scalar(
            scoped(
                select(func.sum(Sale.net_sales)).where(
                    Sale.sale_date.between(previous_start, current_start - timedelta(days=1))
                )
            )
        )
        or 0
    )
    change = ((current_sales - previous_sales) / previous_sales * 100) if previous_sales else 0

    transactions = int(
        db.scalar(
            scoped(
                select(func.count(func.distinct(Sale.transaction_number))).where(
                    Sale.sale_date.between(current_start, max_date)
                )
            )
        )
        or 0
    )
    products = int(
        db.scalar(
            scoped(
                select(func.count(func.distinct(Sale.product_id))).where(
                    Sale.sale_date.between(current_start, max_date)
                )
            )
        )
        or 0
    )
    channels = int(
        db.scalar(
            scoped(
                select(func.count(func.distinct(Sale.channel))).where(
                    Sale.sale_date.between(current_start, max_date)
                )
            )
        )
        or 0
    )

    # customer_key is an additive runtime column retained by the transaction
    # importer. Query it directly so older databases remain migration-safe.
    source_clause = "source_import_id IS NOT NULL" if data_mode == "imported" else "source_import_id IS NULL"
    customer_sql = (
        "SELECT COUNT(DISTINCT customer_key) FROM sales "
        "WHERE customer_key IS NOT NULL AND customer_key <> '' "
        "AND sale_date BETWEEN :start AND :end AND " + source_clause
    )
    if allowed_branch_ids:
        safe_ids = ",".join(str(int(branch_id)) for branch_id in sorted(allowed_branch_ids))
        customer_sql += f" AND branch_id IN ({safe_ids})"
    try:
        active_customers = int(
            db.execute(text(customer_sql), {"start": current_start, "end": max_date}).scalar_one() or 0
        )
    except Exception:
        active_customers = 0

    returns = float(
        db.scalar(
            scoped(
                select(
                    func.sum(
                        case(
                            (Sale.gross_sales > Sale.net_sales, Sale.gross_sales - Sale.net_sales),
                            else_=0,
                        )
                    )
                ).where(Sale.sale_date.between(current_start, max_date))
            )
        )
        or 0
    )
    discounts = abs(
        float(
            db.scalar(
                scoped(
                    select(func.sum(Sale.discount_amount)).where(
                        Sale.sale_date.between(current_start, max_date)
                    )
                )
            )
            or 0
        )
    )
    avg_transaction = current_sales / transactions if transactions else 0

    series_stmt = scoped(
        select(Sale.sale_date, func.sum(Sale.net_sales))
        .where(Sale.sale_date >= max_date - timedelta(days=89))
        .group_by(Sale.sale_date)
        .order_by(Sale.sale_date)
    )
    series = [{"date": row[0].isoformat(), "value": float(row[1])} for row in db.execute(series_stmt)]

    # The executive dashboard's risk and headline forecast are canonical 7-day
    # outputs. A later 30-day point forecast must not overwrite the 7-day risk.
    latest_run = db.scalar(
        select(ModelRun)
        .where(
            ModelRun.status == "completed",
            ModelRun.data_end == max_date,
            ModelRun.horizon_days == 7,
        )
        .order_by(desc(ModelRun.completed_at))
        .limit(1)
    )

    forecasts = []
    forecast_sales = 0.0
    probability = 0.0
    active_alerts = 0
    top_alerts = []
    if latest_run:
        rows = db.scalars(
            select(Forecast)
            .where(Forecast.model_run_id == latest_run.id)
            .order_by(Forecast.forecast_date)
        ).all()
        forecasts = [
            {
                "date": row.forecast_date.isoformat(),
                "median": float(row.predicted_sales),
                "lower": float(row.lower_bound),
                "upper": float(row.upper_bound),
            }
            for row in rows[:7]
        ]
        forecast_sales = sum(float(row.predicted_sales) for row in rows[:7])
        probability = max((row.decline_probability for row in rows[:7]), default=0.0)
        active_alerts = int(
            db.scalar(
                select(func.count(Alert.id))
                .join(Forecast, Alert.forecast_id == Forecast.id)
                .where(
                    Forecast.model_run_id == latest_run.id,
                    Alert.is_resolved.is_(False),
                )
            )
            or 0
        )
        top_alerts = db.scalars(
            select(Alert)
            .join(Forecast, Alert.forecast_id == Forecast.id)
            .where(
                Forecast.model_run_id == latest_run.id,
                Alert.is_resolved.is_(False),
            )
            .order_by(desc(Alert.created_at))
            .limit(5)
        ).all()

    return {
        "current_sales": current_sales,
        "previous_sales": previous_sales,
        "change_percent": change,
        "forecast_sales": forecast_sales,
        "decline_probability": probability,
        "active_alerts": active_alerts,
        "transactions": transactions,
        "active_customers": active_customers,
        "products": products,
        "channels": channels,
        "returns": returns,
        "discounts": discounts,
        "avg_transaction": avg_transaction,
        "series": series,
        "forecast_series": forecasts,
        "latest_run": latest_run,
        "top_alerts": top_alerts,
        "total_records": total_records,
        "data_end": max_date,
        "data_mode": data_mode,
    }
