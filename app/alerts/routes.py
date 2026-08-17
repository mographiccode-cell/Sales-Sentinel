from __future__ import annotations

from flask import Blueprint, abort, redirect, render_template, request, url_for
from sqlalchemy import desc, select

from app.database import session_scope
from app.models import Alert, Forecast, Recommendation
from app.services.security import login_required, permission_required

alerts_bp = Blueprint("alerts", __name__, url_prefix="/alerts")


@alerts_bp.get("/")
@login_required
@permission_required("alerts.view")
def index():
    status = request.args.get("status", "active").strip().lower()
    if status not in {"active", "resolved", "all"}:
        status = "active"

    with session_scope() as db:
        stmt = select(Alert).order_by(desc(Alert.created_at)).limit(100)
        if status == "active":
            stmt = stmt.where(Alert.is_resolved.is_(False))
        elif status == "resolved":
            stmt = stmt.where(Alert.is_resolved.is_(True))
        alerts = db.scalars(stmt).all()

        recommendation_rows = db.scalars(
            select(Recommendation)
            .where(Recommendation.alert_id.in_([item.id for item in alerts]))
            .order_by(Recommendation.priority)
        ).all() if alerts else []
        recommendation_map: dict[int, list[Recommendation]] = {}
        for recommendation in recommendation_rows:
            recommendation_map.setdefault(int(recommendation.alert_id), []).append(recommendation)

        forecast_run_map: dict[int, int] = {}
        forecast_ids = [item.forecast_id for item in alerts if item.forecast_id]
        if forecast_ids:
            for forecast in db.scalars(select(Forecast).where(Forecast.id.in_(forecast_ids))).all():
                forecast_run_map[forecast.id] = forecast.model_run_id

    return render_template(
        "alerts/index.html",
        alerts=alerts,
        status=status,
        recommendation_map=recommendation_map,
        forecast_run_map=forecast_run_map,
    )


@alerts_bp.post("/<int:alert_id>/read")
@login_required
@permission_required("alerts.view")
def mark_read(alert_id: int):
    with session_scope() as db:
        alert = db.get(Alert, alert_id)
        if not alert:
            abort(404)
        alert.is_read = True
    return redirect(url_for("alerts.index", status=request.form.get("status", "active")))


@alerts_bp.post("/<int:alert_id>/resolve")
@login_required
@permission_required("alerts.view")
def resolve(alert_id: int):
    with session_scope() as db:
        alert = db.get(Alert, alert_id)
        if not alert:
            abort(404)
        alert.is_read = True
        alert.is_resolved = True
    return redirect(url_for("alerts.index", status=request.form.get("status", "active")))


@alerts_bp.post("/<int:alert_id>/reopen")
@login_required
@permission_required("alerts.view")
def reopen(alert_id: int):
    with session_scope() as db:
        alert = db.get(Alert, alert_id)
        if not alert:
            abort(404)
        alert.is_resolved = False
    return redirect(url_for("alerts.index", status=request.form.get("status", "resolved")))
