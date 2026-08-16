from __future__ import annotations

import csv
import io

from flask import Blueprint, Response, abort, render_template
from sqlalchemy import desc, select
from fpdf import FPDF

from app.database import session_scope
from app.models import Forecast, ModelRun
from app.services.security import login_required

reports_bp = Blueprint("reports", __name__, url_prefix="/reports")


@reports_bp.get("/")
@login_required
def index():
    with session_scope() as db:
        runs = db.scalars(
            select(ModelRun)
            .where(ModelRun.status == "completed")
            .order_by(desc(ModelRun.completed_at))
            .limit(30)
        ).all()
    return render_template("reports/index.html", runs=runs)


def _pdf_payload(run: ModelRun, forecasts: list[Forecast]) -> bytes:
    """Create a dependency-light, auditable PDF report.

    The UI remains bilingual; the downloadable PDF uses ASCII labels so it can
    be generated consistently in serverless environments without bundling font
    files. Numeric results and model metadata are identical to the stored run.
    """
    metrics = run.metrics_json or {}
    decline = metrics.get("decline_engine", {}) or {}
    quality = (metrics.get("quality", {}) or {}).get("point", {}) or {}
    observed = metrics.get("observed_recent", {}) or {}
    predicted_change = float(metrics.get("predicted_change_pct", 0) or 0)

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
    pdf.set_font("Helvetica", size=10)
    rows = [
        ("Decline probability", f"{float(decline.get('score', 0) or 0) * 100:.1f}%"),
        ("Decision threshold", f"{float(decline.get('decision_threshold', 0) or 0) * 100:.1f}%"),
        ("Observed change", f"{float(observed.get('change_pct', 0) or 0):+.1f}%"),
        ("Forecast decline", f"{max(0.0, -predicted_change):.1f}%"),
        ("Forecast quality (1-WAPE)", f"{float(quality.get('accuracy_proxy_pct', 0) or 0):.1f}%"),
        ("WAPE", f"{float(quality.get('error_wape_pct', 0) or 0):.1f}%"),
        ("Interval coverage", f"{float(quality.get('interval_coverage_pct', 0) or 0):.1f}%"),
    ]
    for label, value in rows:
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(70, 7, label)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "7-day forecast", new_x="LMARGIN", new_y="NEXT")
    widths = [30, 40, 40, 40, 35]
    headers = ["Date", "Predicted", "Lower", "Upper", "Decline risk"]
    pdf.set_font("Helvetica", "B", 8)
    for w, header in zip(widths, headers):
        pdf.cell(w, 7, header, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", size=8)
    for item in forecasts:
        values = [
            item.forecast_date.isoformat(),
            f"{float(item.predicted_sales):,.2f}",
            f"{float(item.lower_bound):,.2f}",
            f"{float(item.upper_bound):,.2f}",
            f"{float(item.decline_probability) * 100:.1f}%",
        ]
        for w, value in zip(widths, values):
            pdf.cell(w, 7, value, border=1)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(90, 100, 95)
    pdf.multi_cell(0, 5, "Decision-support output. Review contributing products, customers, and dates before taking action.")
    return bytes(pdf.output())


@reports_bp.get("/<int:run_id>.<fmt>")
@login_required
def download(run_id: int, fmt: str):
    with session_scope() as db:
        run = db.get(ModelRun, run_id)
        if not run:
            abort(404)
        forecasts = db.scalars(
            select(Forecast)
            .where(Forecast.model_run_id == run_id)
            .order_by(Forecast.forecast_date)
        ).all()

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
        writer.writerow([
            item.forecast_date.isoformat(), item.predicted_sales, item.lower_bound,
            item.upper_bound, item.decline_probability,
        ])
    payload = output.getvalue().encode("utf-8-sig")
    response = Response(payload, mimetype="text/csv")
    response.headers["Content-Disposition"] = f'attachment; filename="forecast-{run_id}.csv"'
    return response
