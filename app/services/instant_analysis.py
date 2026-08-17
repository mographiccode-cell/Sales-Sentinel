from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import func, select

from app.models import Alert, DeclineFactor, Forecast, ModelRun, Recommendation, Sale
from app.services.adaptive_forecasting_engine import forecast
from app.services.decline_explainer import explain_decline_drivers
from app.services.portable_decline_engine import assess_decline_risk


def _grouped_sales(db, condition):
    rows = db.execute(
        select(Sale.sale_date, func.sum(Sale.net_sales))
        .where(condition)
        .group_by(Sale.sale_date)
        .order_by(Sale.sale_date)
    ).all()
    if not rows:
        return []
    by_date = {row[0]: float(row[1] or 0.0) for row in rows}
    start = rows[0][0]
    end = rows[-1][0]
    output = []
    cursor = start
    while cursor <= end:
        output.append((cursor, by_date.get(cursor, 0.0)))
        cursor += timedelta(days=1)
    return output


def daily_sales_rows(db):
    """Prefer imported merchant history, keep calendar days contiguous, never mix sources."""
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


def _recent_change(rows, window: int = 7) -> dict:
    values = [float(row[1] or 0.0) for row in rows]
    if len(values) < window * 2:
        return {
            "current_sales": sum(values[-window:]),
            "previous_sales": 0.0,
            "change_pct": 0.0,
            "decline_pct": 0.0,
        }
    current = sum(values[-window:])
    previous = sum(values[-window * 2:-window])
    change = ((current - previous) / abs(previous) * 100.0) if previous else 0.0
    return {
        "current_sales": current,
        "previous_sales": previous,
        "change_pct": change,
        "decline_pct": max(0.0, -change),
    }


def _point_quality(generated: list[dict]) -> dict:
    first = generated[0]
    metrics = first.get("metrics") or {}
    daily_wape = metrics.get("wape")
    daily_wape = float(daily_wape) if daily_wape is not None else None
    horizon_wape = metrics.get("horizon_total_wape")
    horizon_wape = float(horizon_wape) if horizon_wape is not None else None
    primary_wape = horizon_wape if horizon_wape is not None else daily_wape
    return {
        "model_name": first.get("model_name", "—"),
        "model_version": first.get("model_version", "—"),
        "accuracy_proxy_pct": max(0.0, min(100.0, (1.0 - primary_wape) * 100.0)) if primary_wape is not None else None,
        "error_wape_pct": max(0.0, primary_wape * 100.0) if primary_wape is not None else None,
        "daily_wape_pct": max(0.0, daily_wape * 100.0) if daily_wape is not None else None,
        "horizon_total_wape_pct": max(0.0, horizon_wape * 100.0) if horizon_wape is not None else None,
        "horizon_backtest_folds": int(metrics.get("horizon_backtest_folds") or 0),
        "interval_coverage_pct": None,
        "interval_method": (first.get("calibration") or {}).get("interval_method"),
        "mae": float(metrics.get("mae")) if metrics.get("mae") is not None else None,
        "rmse": float(metrics.get("rmse")) if metrics.get("rmse") is not None else None,
        "backtest_points": int(metrics.get("backtest_points") or 0),
        "selection_metric": metrics.get("selection_metric"),
        "candidate_wape": metrics.get("candidate_wape") or {},
        "evidence_scope": metrics.get("evidence_scope"),
    }


def _decline_quality(decline_risk: dict) -> dict:
    accuracy = decline_risk.get("diagnostic_accuracy")
    tp = int(decline_risk.get("diagnostic_tp") or 0)
    tn = int(decline_risk.get("diagnostic_tn") or 0)
    fp = int(decline_risk.get("diagnostic_fp") or 0)
    fn = int(decline_risk.get("diagnostic_fn") or 0)
    total = tp + tn + fp + fn
    return {
        "accuracy_pct": (float(accuracy) * 100.0) if accuracy is not None else None,
        "error_pct": ((1.0 - float(accuracy)) * 100.0) if accuracy is not None else None,
        "correct_count": tp + tn if total else None,
        "wrong_count": fp + fn if total else None,
        "sample_size": total or None,
        "precision_pct": (float(decline_risk.get("diagnostic_precision")) * 100.0) if decline_risk.get("diagnostic_precision") is not None else None,
        "recall_pct": (float(decline_risk.get("diagnostic_recall")) * 100.0) if decline_risk.get("diagnostic_recall") is not None else None,
        "f1_pct": (float(decline_risk.get("diagnostic_f1")) * 100.0) if decline_risk.get("diagnostic_f1") is not None else None,
        "roc_auc_pct": (float(decline_risk.get("diagnostic_roc_auc")) * 100.0) if decline_risk.get("diagnostic_roc_auc") is not None else None,
        "evidence_label": decline_risk.get("diagnostic_evidence_label"),
        "scientific_status": decline_risk.get("scientific_status"),
    }


def run_instant_analysis(db, *, horizon: int = 7, created_by_id: int | None = None) -> dict:
    """Run the same adaptive point forecast + dedicated V18 risk logic used by the forecast page."""
    rows, data_mode = daily_sales_rows(db)
    if len(rows) < 28:
        return {
            "available": False,
            "reason": f"At least 28 daily observations are required; found {len(rows)}.",
            "history_days": len(rows),
            "data_mode": data_mode,
        }

    values = [float(row[1] or 0.0) for row in rows]
    generated = forecast(values, rows[-1][0], horizon)
    point = _point_quality(generated)
    recent = _recent_change(rows)
    explanation = explain_decline_drivers(db, window=7) if data_mode.startswith("transaction_level") else {"available": False, "drivers": []}

    if horizon == 7:
        decline_risk = assess_decline_risk(db)
    else:
        decline_risk = {
            "available": False,
            "reason": "The dedicated decline-risk runtime supports the canonical 7-day target only.",
        }

    decline_supported = bool(decline_risk.get("available"))
    if decline_supported:
        risk_score = float(decline_risk.get("score") or 0.0)
        model_name = str(decline_risk.get("model_name") or generated[0]["model_name"])
        model_version = str(decline_risk.get("model_version") or generated[0]["model_version"])
        for item in generated:
            item["decline_probability"] = risk_score
    else:
        risk_score = 0.0
        model_name = str(generated[0]["model_name"])
        model_version = str(generated[0]["model_version"])
        for item in generated:
            item["decline_probability"] = 0.0

    baseline_daily = sum(values[-28:]) / min(28, len(values))
    forecast_total = sum(float(item["predicted"]) for item in generated)
    forecast_average = forecast_total / len(generated)
    predicted_change_pct = (
        ((forecast_average - baseline_daily) / abs(baseline_daily)) * 100.0
        if baseline_daily else 0.0
    )
    predicted_decline_pct = max(0.0, -predicted_change_pct)

    # Operational decline alerts may only be emitted by the dedicated V18 policy.
    should_alert = bool(decline_supported and decline_risk.get("alert"))
    severity = "high" if should_alert and risk_score >= 0.70 else ("medium" if should_alert else "low")
    alert_source = "model" if should_alert else "none"

    decline_quality = _decline_quality(decline_risk)
    metrics = {
        "decline_engine": decline_risk,
        "decline_probability_supported": decline_supported,
        "point_forecast_engine": {
            "name": point["model_name"],
            "version": point["model_version"],
            "metrics": generated[0].get("metrics") or {},
            "calibration": generated[0].get("calibration") or {},
        },
        "quality": {"point": point, "decline": decline_quality},
        "observed_recent": recent,
        "predicted_change_pct": predicted_change_pct,
        "explanation": explanation,
    }

    now = datetime.now(timezone.utc)
    run = ModelRun(
        model_name=model_name,
        model_version=model_version,
        status="completed",
        horizon_days=horizon,
        filters_json={
            "data_mode": data_mode,
            "decline_engine_available": decline_supported,
            "auto_after_import": True,
            "point_forecast_selection": "merchant_horizon_total_wape_then_daily_wape",
        },
        metrics_json=metrics,
        data_start=rows[0][0],
        data_end=rows[-1][0],
        sample_size=len(rows),
        started_at=now,
        completed_at=now,
        created_by_id=created_by_id,
    )
    db.add(run)
    db.flush()

    forecast_models: list[Forecast] = []
    for item in generated:
        forecast_row = Forecast(
            model_run_id=run.id,
            forecast_date=item["date"],
            scope_type="company",
            predicted_sales=Decimal(str(item["predicted"])),
            lower_bound=Decimal(str(item["lower"])),
            upper_bound=Decimal(str(item["upper"])),
            baseline_sales=Decimal(str(item["baseline"])),
            decline_probability=float(item["decline_probability"]),
            decline_percent=float(item["decline_percent"]),
        )
        db.add(forecast_row)
        db.flush()
        forecast_models.append(forecast_row)

    alert_model = None
    recommendation_ar = "استمر في المراقبة اليومية؛ لا يوجد تنبيه صادر من محرك خطر الانخفاض المعتمد."
    recommendation_en = "Continue daily monitoring; the dedicated decline-risk engine has not issued an alert."
    if should_alert and forecast_models:
        highest = max(forecast_models, key=lambda item: float(item.decline_probability or 0.0))
        alert_model = Alert(
            forecast_id=highest.id,
            severity=severity,
            title_ar="إنذار مبكر لاحتمال انخفاض المبيعات",
            title_en="Early sales-decline warning",
            message_ar=f"اكتشف Sales Sentinel V18 خطر انخفاض خلال 7 أيام بدرجة {risk_score:.1%}.",
            message_en=f"Sales Sentinel V18 detected a 7-day sales-decline risk score of {risk_score:.1%}.",
        )
        db.add(alert_model)
        db.flush()
        db.add(DeclineFactor(
            forecast_id=highest.id,
            factor_code="v18_decline_risk",
            factor_name_ar="خطر الانخفاض وفق V18",
            factor_name_en="V18 decline risk",
            impact_value=risk_score,
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
        recommendation_ar = explanation.get("recommended_action_ar") or "راجع المنتجات والعملاء والقنوات الأكثر تراجعًا قبل اتخاذ الإجراء."
        recommendation_en = explanation.get("recommended_action_en") or "Review the most-declining products, customers and channels before acting."
        db.add(Recommendation(
            alert_id=alert_model.id,
            factor_code=str(explanation.get("primary_driver_code") or "v18_decline_risk")[:80],
            text_ar=recommendation_ar,
            text_en=recommendation_en,
            rationale_ar="التوصية مبنية على تنبيه V18 وإشارات البيانات، وهي دعم قرار وليست قرارًا آليًا.",
            rationale_en="The recommendation follows the V18 alert and data signals; it is decision support, not an automated action.",
            priority=1,
        ))

    db.flush()

    return {
        "available": True,
        "run_id": run.id,
        "generated_at": now.isoformat(),
        "horizon_days": horizon,
        "data_mode": data_mode,
        "history_days": len(rows),
        "data_start": rows[0][0].isoformat(),
        "data_end": rows[-1][0].isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "point_model_name": point["model_name"],
        "point_model_version": point["model_version"],
        "risk_probability_pct": risk_score * 100.0,
        "decision_threshold_pct": (float(decline_risk.get("decision_threshold")) * 100.0) if decline_risk.get("decision_threshold") is not None else None,
        "decline_probability_supported": decline_supported,
        "v18_available": decline_supported,
        "v18_reason": decline_risk.get("reason"),
        "alert": should_alert,
        "alert_source": alert_source,
        "severity": severity,
        "observed_current_7d_sales": recent["current_sales"],
        "observed_previous_7d_sales": recent["previous_sales"],
        "observed_change_pct": recent["change_pct"],
        "observed_decline_pct": recent["decline_pct"],
        "baseline_daily_sales": baseline_daily,
        "forecast_total": forecast_total,
        "forecast_average": forecast_average,
        "predicted_change_pct": predicted_change_pct,
        "predicted_decline_pct": predicted_decline_pct,
        "forecast_accuracy_pct": point["accuracy_proxy_pct"] or 0.0,
        "forecast_error_pct": point["error_wape_pct"] or 0.0,
        "forecast_backtest_points": point["backtest_points"],
        "forecast_selection_metric": point["selection_metric"],
        "forecast_candidate_wape": point["candidate_wape"],
        "interval_coverage_pct": 0.0,
        "interval_method": point["interval_method"],
        "forecast_mae": point["mae"] or 0.0,
        "forecast_rmse": point["rmse"] or 0.0,
        "decline_diagnostic_accuracy_pct": decline_quality["accuracy_pct"],
        "decline_diagnostic_error_pct": decline_quality["error_pct"],
        "decline_correct_count": decline_quality["correct_count"],
        "decline_wrong_count": decline_quality["wrong_count"],
        "decline_diagnostic_sample_size": decline_quality["sample_size"],
        "decline_precision_pct": decline_quality["precision_pct"],
        "decline_recall_pct": decline_quality["recall_pct"],
        "decline_f1_pct": decline_quality["f1_pct"],
        "decline_roc_auc_pct": decline_quality["roc_auc_pct"],
        "decline_evidence_label": decline_quality["evidence_label"],
        "scientific_status": decline_quality["scientific_status"],
        "recommendation_ar": recommendation_ar,
        "recommendation_en": recommendation_en,
        "explanation": explanation,
        "forecasts": [
            {
                "date": item["date"].isoformat(),
                "predicted": float(item["predicted"]),
                "lower": float(item["lower"]),
                "upper": float(item["upper"]),
                "decline_probability_pct": float(item["decline_probability"]) * 100.0,
                "decline_percent_pct": float(item["decline_percent"]) * 100.0,
            }
            for item in generated
        ],
    }
