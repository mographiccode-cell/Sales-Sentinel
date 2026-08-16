from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import func, select

from app.models import Alert, DeclineFactor, Forecast, ModelRun, Recommendation, Sale
from app.services.forecasting_engine import forecast, load_artifact as load_point_artifact
from app.services.portable_decline_engine import assess_decline_risk


def _grouped_sales(db, condition):
    return db.execute(
        select(Sale.sale_date, func.sum(Sale.net_sales))
        .where(condition)
        .group_by(Sale.sale_date)
        .order_by(Sale.sale_date)
    ).all()


def daily_sales_rows(db):
    """Prefer real transaction-level imports over daily aggregates and demo seed data."""
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


def _point_quality() -> dict:
    artifact = load_point_artifact()
    metrics = artifact.get("metrics") or {}
    wape = float(metrics.get("wape") or 0.0)
    coverage = float((artifact.get("residual_quantiles") or {}).get("empirical_coverage") or 0.0)
    return {
        "model_name": artifact.get("selected_model", "—"),
        "model_version": artifact.get("version", "—"),
        # This is deliberately labelled as 1-WAPE, not classification accuracy.
        "accuracy_proxy_pct": max(0.0, min(100.0, (1.0 - wape) * 100.0)),
        "error_wape_pct": max(0.0, wape * 100.0),
        "interval_coverage_pct": max(0.0, min(100.0, coverage * 100.0)),
        "mae": float(metrics.get("mae") or 0.0),
        "rmse": float(metrics.get("rmse") or 0.0),
        "smape_pct": float(metrics.get("smape") or 0.0) * 100.0,
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
    """Run forecasting + decline inference immediately after import.

    The result is a fully JSON-serializable snapshot so the UI can keep the last
    analysis in browser localStorage when Vercel is using ephemeral SQLite.
    A normal ModelRun/Forecast/Alert record is still written to SQL so persistent
    deployments retain the standard history and report flow.
    """
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
    point = _point_quality()
    recent = _recent_change(rows)

    if horizon == 7:
        decline_risk = assess_decline_risk(db)
    else:
        decline_risk = {
            "available": False,
            "reason": "V18 is validated for the 7-day early-decline target only.",
        }

    if decline_risk.get("available"):
        risk_score = float(decline_risk.get("score") or 0.0)
        model_name = str(decline_risk.get("model_name") or generated[0]["model_name"])
        model_version = str(decline_risk.get("model_version") or generated[0]["model_version"])
        for item in generated:
            item["decline_probability"] = risk_score
            item["model_name"] = model_name
            item["model_version"] = model_version
    else:
        risk_score = max(float(item.get("decline_probability") or 0.0) for item in generated)
        model_name = str(generated[0]["model_name"])
        model_version = str(generated[0]["model_version"])

    baseline_daily = sum(values[-28:]) / min(28, len(values))
    forecast_total = sum(float(item["predicted"]) for item in generated)
    forecast_average = forecast_total / len(generated)
    predicted_change_pct = (
        ((forecast_average - baseline_daily) / abs(baseline_daily)) * 100.0
        if baseline_daily else 0.0
    )
    predicted_decline_pct = max(0.0, -predicted_change_pct)

    model_alert = bool(decline_risk.get("alert")) if decline_risk.get("available") else risk_score >= 0.55
    observed_alert = recent["change_pct"] <= -5.0
    forecast_alert = predicted_decline_pct >= 5.0
    should_alert = model_alert or observed_alert or forecast_alert

    if risk_score >= 0.70 or recent["decline_pct"] >= 20.0 or predicted_decline_pct >= 20.0:
        severity = "high"
    elif risk_score >= 0.45 or recent["decline_pct"] >= 10.0 or predicted_decline_pct >= 10.0:
        severity = "medium"
    else:
        severity = "low"

    if model_alert and observed_alert:
        alert_source = "model_and_observed"
    elif model_alert:
        alert_source = "model"
    elif observed_alert:
        alert_source = "observed"
    elif forecast_alert:
        alert_source = "forecast"
    else:
        alert_source = "none"

    decline_quality = _decline_quality(decline_risk)
    metrics = {
        "decline_engine": decline_risk,
        "point_forecast_engine": {
            "name": point["model_name"],
            "version": point["model_version"],
            "metrics": generated[0].get("metrics") or {},
        },
        "quality": {"point": point, "decline": decline_quality},
        "observed_recent": recent,
        "predicted_change_pct": predicted_change_pct,
    }

    now = datetime.now(timezone.utc)
    run = ModelRun(
        model_name=model_name,
        model_version=model_version,
        status="completed",
        horizon_days=horizon,
        filters_json={
            "data_mode": data_mode,
            "decline_engine_available": bool(decline_risk.get("available")),
            "auto_after_import": True,
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
    recommendation_ar = "استمر في المراقبة اليومية؛ لا يوجد تنبيه يتجاوز الحدود الحالية."
    recommendation_en = "Continue daily monitoring; no current alert threshold is exceeded."
    if should_alert and forecast_models:
        highest = max(forecast_models, key=lambda item: float(item.decline_probability or 0.0))
        alert_model = Alert(
            forecast_id=highest.id,
            severity=severity,
            title_ar="تنبيه تحليل المبيعات المرفوعة",
            title_en="Uploaded-sales analysis alert",
            message_ar=(
                f"احتمال الانخفاض {risk_score:.1%}، والتغير المرصود لآخر 7 أيام {recent['change_pct']:.1f}%، "
                f"والتغير المتوقع مقابل خط أساس 28 يومًا {predicted_change_pct:.1f}%."
            ),
            message_en=(
                f"Decline probability {risk_score:.1%}; observed last-7-day change {recent['change_pct']:.1f}%; "
                f"forecast change versus the 28-day baseline {predicted_change_pct:.1f}%."
            ),
        )
        db.add(alert_model)
        db.flush()
        impact = max(recent["decline_pct"], predicted_decline_pct) / 100.0
        db.add(DeclineFactor(
            forecast_id=highest.id,
            factor_code="uploaded_recent_vs_baseline",
            factor_name_ar="اتجاه البيانات المرفوعة مقابل خط الأساس",
            factor_name_en="Uploaded-data trend versus baseline",
            impact_value=impact,
            direction="negative" if impact > 0 else "neutral",
            method="v18_plus_observed_window" if decline_risk.get("available") else "point_forecast_plus_observed_window",
        ))
        recommendation_ar = "راجع المنتجات والعملاء والقنوات في آخر 14 يومًا وابدأ بمعالجة العناصر الأكثر مساهمة في الانخفاض."
        recommendation_en = "Review products, customers, and channels in the last 14 days and address the largest decline contributors first."
        db.add(Recommendation(
            alert_id=alert_model.id,
            factor_code="uploaded_recent_vs_baseline",
            text_ar=recommendation_ar,
            text_en=recommendation_en,
            rationale_ar="التوصية مبنية على بيانات الملف المرفوع مع نتيجة النموذج، وليست قرارًا آليًا.",
            rationale_en="The recommendation combines the uploaded data with model output and is decision support, not an automated action.",
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
        "v18_available": bool(decline_risk.get("available")),
        "v18_reason": decline_risk.get("reason"),
        "alert": bool(should_alert),
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
        "forecast_accuracy_pct": point["accuracy_proxy_pct"],
        "forecast_error_pct": point["error_wape_pct"],
        "interval_coverage_pct": point["interval_coverage_pct"],
        "forecast_mae": point["mae"],
        "forecast_rmse": point["rmse"],
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
