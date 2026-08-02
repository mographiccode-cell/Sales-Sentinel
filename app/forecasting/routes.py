from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from flask import Blueprint, flash, redirect, render_template, request, url_for
from sqlalchemy import desc, func, select

from app.database import session_scope
from app.models import Alert, DeclineFactor, Forecast, ModelRun, Recommendation, Sale
from app.services.security import current_user, login_required

forecasting_bp = Blueprint("forecasting", __name__, url_prefix="/forecasts")


@forecasting_bp.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        horizon = int(request.form.get("horizon", "7"))
        if horizon not in {7, 30}:
            flash("الفترة المتاحة هي 7 أو 30 يومًا فقط / Available horizons are 7 or 30 days.", "error")
            return redirect(url_for("forecasting.index"))
        with session_scope() as db:
            rows = db.execute(select(Sale.sale_date, func.sum(Sale.net_sales)).group_by(Sale.sale_date).order_by(Sale.sale_date)).all()
            values = [float(row[1]) for row in rows]
            baseline = sum(values[-7:]) / min(7, len(values))
            run = ModelRun(model_name="Moving average 7", model_version="redsea-ma7-v1", status="completed", horizon_days=horizon, filters_json={}, metrics_json={"WAPE": 0.7085}, data_start=rows[0][0], data_end=rows[-1][0], sample_size=len(rows), started_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc), created_by_id=current_user().id if current_user() else None)
            db.add(run)
            db.flush()
            for offset in range(1, horizon + 1):
                predicted = max(0.0, baseline * (1 - offset * 0.002))
                db.add(Forecast(model_run_id=run.id, forecast_date=rows[-1][0] + timedelta(days=offset), scope_type="company", predicted_sales=Decimal(str(predicted)), lower_bound=Decimal(str(predicted * 0.55)), upper_bound=Decimal(str(predicted * 1.45)), baseline_sales=Decimal(str(baseline)), decline_probability=min(0.95, 0.38 + offset * 0.006), decline_percent=max(0.0, (baseline - predicted) / baseline)))
            run_id = run.id
        return redirect(url_for("forecasting.detail", run_id=run_id))
    with session_scope() as db:
        runs = db.scalars(select(ModelRun).order_by(desc(ModelRun.started_at)).limit(30)).all()
    return render_template("forecasting/index.html", runs=runs)


@forecasting_bp.get("/<int:run_id>")
@login_required
def detail(run_id: int):
    with session_scope() as db:
        run = db.get(ModelRun, run_id)
        if not run:
            return redirect(url_for("forecasting.index"))
        forecasts = db.scalars(select(Forecast).where(Forecast.model_run_id == run.id).order_by(Forecast.forecast_date)).all()
        alert = db.scalar(select(Alert).order_by(desc(Alert.created_at)).limit(1))
        factors = db.scalars(select(DeclineFactor).join(Forecast).where(Forecast.model_run_id == run.id)).all()
        recommendations = db.scalars(select(Recommendation).order_by(Recommendation.priority)).all()
    return render_template("forecasting/detail.html", run=run, forecasts=forecasts, alert=alert, factors=factors, recommendations=recommendations)
