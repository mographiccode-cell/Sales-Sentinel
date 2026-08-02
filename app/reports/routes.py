from __future__ import annotations

import csv
import io

from flask import Blueprint, Response, render_template
from sqlalchemy import desc, select

from app.database import session_scope
from app.models import Forecast, ModelRun
from app.services.security import login_required

reports_bp = Blueprint("reports", __name__, url_prefix="/reports")


@reports_bp.get("/")
@login_required
def index():
    with session_scope() as db:
        runs = db.scalars(select(ModelRun).where(ModelRun.status == "completed").order_by(desc(ModelRun.completed_at)).limit(30)).all()
    return render_template("reports/index.html", runs=runs)


@reports_bp.get("/<int:run_id>.<fmt>")
@login_required
def download(run_id: int, fmt: str):
    with session_scope() as db:
        forecasts = db.scalars(select(Forecast).where(Forecast.model_run_id == run_id).order_by(Forecast.forecast_date)).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["date", "predicted_sales", "lower_bound", "upper_bound", "decline_probability"])
    for item in forecasts:
        writer.writerow([item.forecast_date.isoformat(), item.predicted_sales, item.lower_bound, item.upper_bound, item.decline_probability])
    payload = output.getvalue().encode("utf-8-sig")
    response = Response(payload, mimetype="text/csv")
    response.headers["Content-Disposition"] = f'attachment; filename="forecast-{run_id}.csv"'
    return response
