from __future__ import annotations

from datetime import timedelta

from sqlalchemy import case, desc, func, select

from app.models import Alert, Forecast, ModelRun, Sale


def dashboard_summary(db, allowed_branch_ids: set[int] | None = None) -> dict:
    def scoped(stmt):
        return stmt.where(Sale.branch_id.in_(allowed_branch_ids)) if allowed_branch_ids else stmt

    total_records = int(db.scalar(scoped(select(func.count(Sale.id)))) or 0)
    max_date = db.scalar(scoped(select(func.max(Sale.sale_date))))
    if not max_date:
        return {"current_sales": 0, "previous_sales": 0, "change_percent": 0, "forecast_sales": 0, "decline_probability": 0, "active_alerts": 0, "transactions": 0, "returns": 0, "discounts": 0, "avg_transaction": 0, "series": [], "forecast_series": [], "latest_run": None, "top_alerts": [], "total_records": total_records, "data_end": None}
    current_start = max_date - timedelta(days=29)
    previous_start = current_start - timedelta(days=30)
    current_sales = float(db.scalar(scoped(select(func.sum(Sale.net_sales)).where(Sale.sale_date.between(current_start, max_date)))) or 0)
    previous_sales = float(db.scalar(scoped(select(func.sum(Sale.net_sales)).where(Sale.sale_date.between(previous_start, current_start - timedelta(days=1))))) or 0)
    change = ((current_sales - previous_sales) / previous_sales * 100) if previous_sales else 0
    transactions = int(db.scalar(scoped(select(func.count(func.distinct(Sale.transaction_number))).where(Sale.sale_date.between(current_start, max_date)))) or 0)
    returns = float(db.scalar(scoped(select(func.sum(case((Sale.transaction_type != "INV", func.abs(Sale.net_sales)), else_=0))).where(Sale.sale_date.between(current_start, max_date)))) or 0)
    discounts = abs(float(db.scalar(scoped(select(func.sum(Sale.discount_amount)).where(Sale.sale_date.between(current_start, max_date)))) or 0))
    avg_transaction = current_sales / transactions if transactions else 0
    series_stmt = scoped(select(Sale.sale_date, func.sum(Sale.net_sales)).where(Sale.sale_date >= max_date - timedelta(days=89)).group_by(Sale.sale_date).order_by(Sale.sale_date))
    series = [{"date": row[0].isoformat(), "value": float(row[1])} for row in db.execute(series_stmt)]
    latest_run = db.scalar(select(ModelRun).where(ModelRun.status == "completed").order_by(desc(ModelRun.completed_at)).limit(1))
    forecasts = []
    forecast_sales = 0.0
    probability = 0.0
    if latest_run:
        rows = db.scalars(select(Forecast).where(Forecast.model_run_id == latest_run.id).order_by(Forecast.forecast_date)).all()
        forecasts = [{"date": row.forecast_date.isoformat(), "median": float(row.predicted_sales), "lower": float(row.lower_bound), "upper": float(row.upper_bound)} for row in rows]
        forecast_sales = sum(float(row.predicted_sales) for row in rows[:30])
        probability = max((row.decline_probability for row in rows), default=0.0)
    active_alerts = int(db.scalar(select(func.count(Alert.id)).where(Alert.is_resolved.is_(False))) or 0)
    top_alerts = db.scalars(select(Alert).where(Alert.is_resolved.is_(False)).order_by(desc(Alert.created_at)).limit(5)).all()
    return {"current_sales": current_sales, "previous_sales": previous_sales, "change_percent": change, "forecast_sales": forecast_sales, "decline_probability": probability, "active_alerts": active_alerts, "transactions": transactions, "returns": returns, "discounts": discounts, "avg_transaction": avg_transaction, "series": series, "forecast_series": forecasts, "latest_run": latest_run, "top_alerts": top_alerts, "total_records": total_records, "data_end": max_date}
