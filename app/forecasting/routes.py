from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from flask import Blueprint, flash, redirect, render_template, request, url_for
from sqlalchemy import desc, func, select

from app.database import session_scope
from app.models import Alert, DeclineFactor, Forecast, ModelRun, Recommendation, Sale
from app.services.adaptive_forecasting_engine import forecast
from app.services.decline_explainer import explain_decline_drivers
from app.services.portable_decline_engine import assess_decline_risk
from app.services.security import current_user, login_required, permission_required

forecasting_bp = Blueprint("forecasting", __name__, url_prefix="/forecasts")


def _grouped_sales(db, condition):
    return db.execute(
        select(Sale.sale_date, func.sum(Sale.net_sales))
        .where(condition)
        .group_by(Sale.sale_date)
        .order_by(Sale.sale_date)
    ).all()


def _daily_sales_rows(db):
    """Choose one coherent history source and never mix seed/demo with user data."""
    rich = _grouped_sales(
        db,
        Sale.transaction_type.notin_(["DAILY_AGGREGATE", "DAILY_IMPORT"]),
    )
    daily_import = _grouped_sales(db, Sale.transaction_type == "DAILY_IMPORT")

    if len(rich) >= 28:
        return rich, "transaction_level"
    if len(daily_import) >= 28:
        return daily_import, "daily_import"
    if rich:
        return rich, "transaction_level_insufficient"
    if daily_import:
        return daily_import, "daily_import_insufficient"

    seed = _grouped_sales(db, Sale.transaction_type == "DAILY_AGGREGATE")
    return seed, "seed_aggregate"


@forecasting_bp.route("/", methods=["GET", "POST"])
@login_required
@permission_required("forecasts.run")
def index():
    if request.method == "POST":
        try:
            horizon = int(request.form.get("horizon", "7"))
            with session_scope() as db:
                rows, data_mode = _daily_sales_rows(db)
                if len(rows) < 28:
                    raise ValueError("Insufficient daily history")
                values = [float(row[1] or 0) for row in rows]
                generated = forecast(values, rows[-1][0], horizon)
                explanation = explain_decline_drivers(db, window=7) if data_mode.startswith("transaction_level") else {"available": False, "drivers": []}
                point_model_name = generated[0]["model_name"]
                point_model_version = generated[0]["model_version"]

                decline_risk = assess_decline_risk(db) if horizon == 7 else {
                    "available": False,
                    "reason": "The validated decline-risk engine supports the canonical 7-day target only. No 30-day decline alert is generated.",
                }
                if decline_risk.get("available"):
                    for item in generated:
                        item["decline_probability"] = float(decline_risk["score"])
                    model_name = decline_risk["model_name"]
                    model_version = decline_risk["model_version"]
                else:
                    for item in generated:
                        item["decline_probability"] = 0.0
                    model_name = point_model_name
                    model_version = point_model_version

                metrics = {
                    "decline_engine": decline_risk,
                    "decline_probability_supported": bool(decline_risk.get("available")),
                    "point_forecast_engine": {
                        "name": point_model_name,
                        "version": point_model_version,
                        "metrics": generated[0]["metrics"],
                        "calibration": generated[0].get("calibration", {}),
                    },
                    "explanation": explanation,
                }

                run = ModelRun(
                    model_name=model_name,
                    model_version=model_version,
                    status="completed",
                    horizon_days=horizon,
                    filters_json={
                        "data_mode": data_mode,
                        "decline_engine_available": bool(decline_risk.get("available")),
                        "point_forecast_selection": "merchant_rolling_wape",
                    },
                    metrics_json=metrics,
                    data_start=rows[0][0],
                    data_end=rows[-1][0],
                    sample_size=len(rows),
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    created_by_id=current_user().id if current_user() else None,
                )
                db.add(run)
                db.flush()
                highest = None
                for item in generated:
                    forecast_row = Forecast(
                        model_run_id=run.id,
                        forecast_date=item["date"],
                        scope_type="company",
                        predicted_sales=Decimal(str(item["predicted"])),
                        lower_bound=Decimal(str(item["lower"])),
                        upper_bound=Decimal(str(item["upper"])),
                        baseline_sales=Decimal(str(item["baseline"])),
                        decline_probability=item["decline_probability"],
                        decline_percent=item["decline_percent"],
                    )
                    db.add(forecast_row)
                    db.flush()
                    highest = forecast_row if highest is None or forecast_row.decline_probability > highest.decline_probability else highest

                should_alert = bool(decline_risk.get("available") and decline_risk.get("alert"))
                if highest and should_alert:
                    severity = "high" if highest.decline_probability >= 0.70 else "medium"
                    mode = decline_risk.get("policy_mode", "static")
                    alert = Alert(
                        forecast_id=highest.id,
                        severity=severity,
                        title_ar="إنذار مبكر لاحتمال انخفاض المبيعات",
                        title_en="Early sales-decline warning",
                        message_ar=f"اكتشف Sales Sentinel V18 خطر انخفاض خلال 7 أيام بدرجة {highest.decline_probability:.0%} باستخدام سياسة {mode}.",
                        message_en=f"Sales Sentinel V18 detected 7-day decline risk of {highest.decline_probability:.0%} using the {mode} policy.",
                    )
                    db.add(alert)
                    db.flush()
                    db.add(DeclineFactor(
                        forecast_id=highest.id,
                        factor_code="trend_vs_baseline",
                        factor_name_ar="الاتجاه مقابل خط الأساس",
                        factor_name_en="Trend versus baseline",
                        impact_value=highest.decline_percent,
                        direction="negative",
                        method="v18_portable_extratrees",
                    ))
                    for driver in explanation.get("drivers", []):
                        db.add(DeclineFactor(
                            forecast_id=highest.id,
                            factor_code=str(driver.get("code") or "decline_signal")[:80],
                            factor_name_ar=str(driver.get("title_ar") or "إشارة انخفاض")[:150],
                            factor_name_en=str(driver.get("title_en") or "Decline signal")[:150],
                            impact_value=float(driver.get("strength_pct") or 0.0) / 100.0,
                            direction="negative",
                            method="recent_window_explanation",
                        ))

                    primary_code = str(explanation.get("primary_driver_code") or "trend_vs_baseline")[:80]
                    recommendation_ar = str(explanation.get("recommended_action_ar") or "راجع الأيام والمنتجات والعملاء المساهمة في الانخفاض قبل اتخاذ إجراء.")
                    recommendation_en = str(explanation.get("recommended_action_en") or "Review contributing days, products and customers before acting.")
                    db.add(Recommendation(
                        alert_id=alert.id,
                        factor_code=primary_code,
                        text_ar=recommendation_ar,
                        text_en=recommendation_en,
                        rationale_ar="التوصية مبنية على أقوى إشارة تفسيرية متاحة وهي دعم قرار وليست قرارًا آليًا؛ قناة RED الحرجة معطلة حتى تتوفر أدلة كافية.",
                        rationale_en="The recommendation follows the strongest available explanatory signal and is decision support, not an automated action; the RED/critical channel remains disabled pending sufficient evidence.",
                        priority=1,
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
        forecasts = db.scalars(
            select(Forecast).where(Forecast.model_run_id == run.id).order_by(Forecast.forecast_date)
        ).all()
        forecast_ids = [item.id for item in forecasts]
        alert = (
            db.scalar(select(Alert).where(Alert.forecast_id.in_(forecast_ids)).order_by(desc(Alert.created_at)).limit(1))
            if forecast_ids else None
        )
        factors = db.scalars(select(DeclineFactor).where(DeclineFactor.forecast_id.in_(forecast_ids))).all() if forecast_ids else []
        recommendations = (
            db.scalars(select(Recommendation).where(Recommendation.alert_id == alert.id).order_by(Recommendation.priority)).all()
            if alert else []
        )
    return render_template(
        "forecasting/detail.html",
        run=run,
        forecasts=forecasts,
        alert=alert,
        factors=factors,
        recommendations=recommendations,
    )
