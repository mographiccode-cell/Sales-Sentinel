from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from flask import Blueprint, flash, redirect, render_template, request, url_for
from sqlalchemy import desc, func, select

from app.database import session_scope
from app.models import Alert, DeclineFactor, Forecast, ModelRun, Recommendation, Sale
from app.services.forecasting_engine import forecast
from app.services.security import current_user, login_required, permission_required

forecasting_bp = Blueprint("forecasting", __name__, url_prefix="/forecasts")


@forecasting_bp.route("/", methods=["GET", "POST"])
@login_required
@permission_required("forecasts.run")
def index():
    if request.method == "POST":
        try:
            horizon = int(request.form.get("horizon", "7"))
            with session_scope() as db:
                rows = db.execute(
                    select(Sale.sale_date, func.sum(Sale.net_sales))
                    .group_by(Sale.sale_date).order_by(Sale.sale_date)
                ).all()
                if len(rows) < 28:
                    raise ValueError("Insufficient daily history")
                values = [float(row[1] or 0) for row in rows]
                generated = forecast(values, rows[-1][0], horizon)
                first = generated[0]
                run = ModelRun(
                    model_name=first["model_name"], model_version=first["model_version"],
                    status="completed", horizon_days=horizon, filters_json={},
                    metrics_json=first["metrics"], data_start=rows[0][0], data_end=rows[-1][0],
                    sample_size=len(rows), started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    created_by_id=current_user().id if current_user() else None,
                )
                db.add(run); db.flush()
                highest = None
                for item in generated:
                    row = Forecast(
                        model_run_id=run.id, forecast_date=item["date"], scope_type="company",
                        predicted_sales=Decimal(str(item["predicted"])),
                        lower_bound=Decimal(str(item["lower"])), upper_bound=Decimal(str(item["upper"])),
                        baseline_sales=Decimal(str(item["baseline"])),
                        decline_probability=item["decline_probability"], decline_percent=item["decline_percent"],
                    )
                    db.add(row); db.flush()
                    highest = row if highest is None or row.decline_probability > highest.decline_probability else highest
                if highest and highest.decline_probability >= 0.55:
                    severity = "critical" if highest.decline_probability >= 0.85 else "high" if highest.decline_probability >= 0.70 else "medium"
                    alert = Alert(
                        forecast_id=highest.id, severity=severity,
                        title_ar="احتمال انخفاض مبيعات", title_en="Sales decline probability",
                        message_ar=f"اكتشف النموذج احتمال انخفاض قدره {highest.decline_probability:.0%}.",
                        message_en=f"The model detected a {highest.decline_probability:.0%} decline probability.",
                    )
                    db.add(alert); db.flush()
                    db.add(DeclineFactor(
                        forecast_id=highest.id, factor_code="trend_vs_baseline",
                        factor_name_ar="الاتجاه مقابل خط الأساس", factor_name_en="Trend versus baseline",
                        impact_value=highest.decline_percent, direction="negative", method="validated_residual_distribution",
                    ))
                    db.add(Recommendation(
                        alert_id=alert.id, factor_code="trend_vs_baseline",
                        text_ar="راجع الأيام والمنتجات والعملاء المساهمة في الانخفاض قبل اتخاذ إجراء.",
                        text_en="Review contributing days, products and customers before acting.",
                        rationale_ar="التوصية مرتبطة بتوقع زمني وفاصل عدم يقين وليست قاعدة ثابتة.",
                        rationale_en="The recommendation is linked to a validated forecast and uncertainty interval.", priority=1,
                    ))
                run_id = run.id
            return redirect(url_for("forecasting.detail", run_id=run_id))
        except (ValueError, RuntimeError) as exc:
            flash(str(exc), "error")
            return redirect(url_for("forecasting.index"))
    with session_scope() as db:
        runs = db.scalars(select(ModelRun).order_by(desc(ModelRun.started_at)).limit(30)).all()
    return render_template("forecasting/index.html", runs=runs)


@forecasting_bp.get("/<int:run_id>")
@login_required
@permission_required("forecasts.run")
def detail(run_id: int):
    with session_scope() as db:
        run = db.get(ModelRun, run_id)
        if not run:
            return redirect(url_for("forecasting.index"))
        forecasts = db.scalars(select(Forecast).where(Forecast.model_run_id == run.id).order_by(Forecast.forecast_date)).all()
        forecast_ids = [item.id for item in forecasts]
        alert = db.scalar(select(Alert).where(Alert.forecast_id.in_(forecast_ids)).order_by(desc(Alert.created_at)).limit(1)) if forecast_ids else None
        factors = db.scalars(select(DeclineFactor).where(DeclineFactor.forecast_id.in_(forecast_ids))).all() if forecast_ids else []
        recommendations = db.scalars(select(Recommendation).where(Recommendation.alert_id == alert.id).order_by(Recommendation.priority)).all() if alert else []
    return render_template("forecasting/detail.html", run=run, forecasts=forecasts, alert=alert, factors=factors, recommendations=recommendations)
