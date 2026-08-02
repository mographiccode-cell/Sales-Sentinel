from __future__ import annotations

import csv
from pathlib import Path

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from sqlalchemy import desc, select

from app.database import session_scope
from app.models import ImportJob
from app.services.security import current_user, login_required, permission_required, safe_filename, sha256_file

imports_bp = Blueprint("imports", __name__, url_prefix="/imports")
UCI_COLUMNS = {"InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"}
DAILY_COLUMNS = {"date", "net_sales"}


def inspect_csv(path: Path) -> tuple[int, int, list[str]]:
    total = accepted = 0
    errors: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        mode = "uci" if UCI_COLUMNS.issubset(columns) else "daily" if DAILY_COLUMNS.issubset(columns) else "invalid"
        if mode == "invalid":
            return 0, 0, ["Required UCI transaction columns or date/net_sales columns are missing"]
        for line, row in enumerate(reader, start=2):
            total += 1
            try:
                if mode == "uci":
                    float(row["Quantity"]); float(row["UnitPrice"])
                    if not row["InvoiceNo"] or not row["InvoiceDate"]:
                        raise ValueError("missing invoice or date")
                else:
                    __import__("datetime").date.fromisoformat(row["date"]); float(row["net_sales"])
                accepted += 1
            except (ValueError, TypeError, KeyError) as exc:
                if len(errors) < 100:
                    errors.append(f"row {line}: {exc}")
    return total, accepted, errors


@imports_bp.route("/", methods=["GET", "POST"])
@login_required
@permission_required("imports.manage")
def index():
    if request.method == "POST":
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename:
            flash("اختر ملفًا أولًا / Select a file first.", "error")
            return redirect(url_for("imports.index"))
        extension = Path(uploaded.filename).suffix.lower()
        if extension != ".csv":
            flash("استخدم CSV للاستيراد التشغيلي؛ تتم معالجة XLSX عبر Pipeline الرسمي / Use CSV for runtime imports.", "error")
            return redirect(url_for("imports.index"))
        destination = Path(current_app.config["UPLOAD_DIR"]) / safe_filename(uploaded.filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        uploaded.save(destination)
        total, accepted, errors = inspect_csv(destination)
        rejected = total - accepted
        status = "validated" if total and not rejected else "validated_with_errors" if accepted else "failed"
        with session_scope() as db:
            db.add(ImportJob(
                filename=destination.name, file_sha256=sha256_file(destination), status=status,
                total_rows=total, accepted_rows=accepted, rejected_rows=rejected,
                error_details={"errors": errors}, created_by_id=current_user().id if current_user() else None,
            ))
        if accepted:
            flash(f"تم التحقق من {accepted} صف وتسجيل البصمة؛ المرفوض {rejected} / Validation completed.", "success")
        else:
            flash("فشل التحقق من بنية الملف / File validation failed.", "error")
        return redirect(url_for("imports.index"))
    with session_scope() as db:
        jobs = db.scalars(select(ImportJob).order_by(desc(ImportJob.created_at)).limit(30)).all()
    return render_template("imports/index.html", jobs=jobs)
