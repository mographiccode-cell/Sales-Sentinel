from __future__ import annotations

import csv
import io

from flask import Blueprint, Response, abort, render_template
from sqlalchemy import desc, select
from fpdf import FPDF

from app.database import session_scope
from app.models import Forecast, ModelRun
from app.services.security import login_required, permission_required

reports_bp = Blueprint("reports", __name__, url_prefix="/reports")


@reports_bp.get("/")
@login_required
@permission_required("reports.export")
def index():
    with session_scope() as db:
        runs = db.scalars(select(ModelRun).where(ModelRun.status == "completed").order_by(desc(ModelRun.completed_at)).limit(30)).all()
    return render_template("reports/index.html", runs=runs)


def _pdf_payload(run: ModelRun, forecasts: list[Forecast]) -> bytes:
    metrics = run.metrics_json or {}
    decline = metrics.get("decline_engine", {}) or {}
    point = metrics.get("point_forecast_engine", {}) or {}
    point_metrics = point.get("metrics", {}) or {}
    explanation = metrics.get("explanation", {}) or {}
    observed = metrics.get("observed_recent", {}) or {}
    decline_supported = bool(metrics.get("decline_probability_supported") and decline.get("available"))

    daily_wape = point_metrics.get("wape")
    daily_wape_pct = float(daily_wape) * 100.0 if daily_wape is not None else None
    horizon_wape = point_metrics.get("horizon_total_wape")
    horizon_wape_pct = float(horizon_wape) * 100.0 if horizon_wape is not None else None
    primary_wape_pct = horizon_wape_pct if horizon_wape_pct is not None else daily_wape_pct
    quality_pct = max(0.0, min(100.0, 100.0 - primary_wape_pct)) if primary_wape_pct is not None else None
    risk_pct = float(decline.get("score", 0) or 0) * 100.0 if decline_supported else None
    risk_level = "High" if risk_pct is not None and risk_pct >= 70 else ("Medium" if risk_pct is not None and risk_pct >= 50 else "Low")
    threshold_pct = float(decline.get("decision_threshold", 0) or 0) * 100.0 if decline_supported else None
    forecast_total = float(metrics.get("forecast_total") or sum(float(item.predicted_sales) for item in forecasts))
    predicted_change_pct = float(metrics.get("predicted_change_pct") or 0.0)
    predicted_decline_pct = float(metrics.get("predicted_decline_pct") or max(0.0, -predicted_change_pct))

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Sales Sentinel - Forecast Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(90, 100, 95)
    pdf.cell(0, 6, f"Run #{run.id} | {run.model_name} | {run.model_version}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Data: {run.data_start} to {run.data_end} | History: {run.sample_size} days | Horizon: {run.horizon_days} days", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_text_color(16, 35, 28)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Executive result", new_x="LMARGIN", new_y="NEXT")
    rows = [
        ("Current 7-day sales", f"{float(observed.get('current_sales', 0) or 0):,.2f}"),
        ("Previous 7-day sales", f"{float(observed.get('previous_sales', 0) or 0):,.2f}"),
        ("Observed 7-day change", f"{float(observed.get('change_pct', 0) or 0):+.1f}%"),
        (f"Expected {run.horizon_days}-day sales", f"{forecast_total:,.2f}"),
        ("Expected horizon change", f"{predicted_change_pct:+.1f}%"),
        ("Expected decline magnitude", f"{predicted_decline_pct:.1f}%"),
        ("Validated decline probability", f"{risk_pct:.1f}%" if risk_pct is not None else "N/A - 7-day risk engine unavailable"),
        ("Risk level", risk_level if risk_pct is not None else "N/A"),
        ("Decline decision threshold", f"{threshold_pct:.1f}%" if threshold_pct is not None else "N/A"),
        ("Point forecast model", str(point.get("name") or "N/A")),
        ("Horizon backtest folds", str(point_metrics.get("horizon_backtest_folds") or "N/A")),
        ("Horizon-total WAPE", f"{horizon_wape_pct:.1f}%" if horizon_wape_pct is not None else "N/A"),
        ("Horizon-total quality (1-WAPE)", f"{quality_pct:.1f}%" if quality_pct is not None else "N/A"),
        ("Daily WAPE (technical)", f"{daily_wape_pct:.1f}%" if daily_wape_pct is not None else "N/A"),
    ]
    for label, value in rows:
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(78, 7, label)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")

    if explanation.get("drivers"):
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Why the decline signal may be occurring", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=8)
        for driver in explanation.get("drivers", [])[:4]:
            title = str(driver.get("title_en") or driver.get("code") or "Signal")
            change = float(driver.get("change_pct") or 0.0)
            strength = float(driver.get("strength_pct") or 0.0)
            pdf.cell(0, 6, f"- {title}: {change:+.1f}% | relative signal strength {strength:.1f}%", new_x="LMARGIN", new_y="NEXT")
        action = explanation.get("recommended_action_en")
        if action:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, "Recommended action", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=8)
            pdf.multi_cell(0, 5, str(action))

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"{run.horizon_days}-day forecast", new_x="LMARGIN", new_y="NEXT")
    widths = [30, 40, 40, 40, 35]
    headers = ["Date", "Predicted", "Lower", "Upper", "Decline risk"]
    pdf.set_font("Helvetica", "B", 8)
    for width, header in zip(widths, headers):
        pdf.cell(width, 7, header, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", size=8)
    for item in forecasts:
        risk_value = f"{float(item.decline_probability) * 100:.1f}%" if decline_supported else "N/A"
        values = [item.forecast_date.isoformat(), f"{float(item.predicted_sales):,.2f}", f"{float(item.lower_bound):,.2f}", f"{float(item.upper_bound):,.2f}", risk_value]
        for width, value in zip(widths, values):
            pdf.cell(width, 7, value, border=1)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(90, 100, 95)
    pdf.multi_cell(0, 5, "Decision-support output. Primary point-forecast quality is merchant-specific horizon-total WAPE. Decline alerts are generated only by the dedicated validated 7-day risk runtime; RED/critical classification is not claimed.")
    return bytes(pdf.output())


@reports_bp.get("/<int:run_id>.<fmt>")
@login_required
@permission_required("reports.export")
def download(run_id: int, fmt: str):
    with session_scope() as db:
        run = db.get(ModelRun, run_id)
        if not run:
            abort(404)
        forecasts = db.scalars(select(Forecast).where(Forecast.model_run_id == run_id).order_by(Forecast.forecast_date)).all()

    fmt = fmt.lower()
    if fmt == "pdf":
        response = Response(_pdf_payload(run, forecasts), mimetype="application/pdf")
        response.headers["Content-Disposition"] = f'attachment; filename="sales-sentinel-report-{run_id}.pdf"'
        return response
    if fmt != "csv":
        abort(404)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["date", "predicted_sales", "lower_bound", "upper_bound", "decline_probability"])
    for item in forecasts:
        writer.writerow([item.forecast_date.isoformat(), item.predicted_sales, item.lower_bound, item.upper_bound, item.decline_probability])
    payload = output.getvalue().encode("utf-8-sig")
    response = Response(payload, mimetype="text/csv")
    response.headers["Content-Disposition"] = f'attachment; filename="forecast-{run_id}.csv"'
    return response
