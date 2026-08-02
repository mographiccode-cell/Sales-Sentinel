from __future__ import annotations

from pathlib import Path
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from sqlalchemy import desc, select

from app.database import session_scope
from app.models import ImportJob
from app.services.security import current_user, login_required, safe_filename, sha256_file

imports_bp = Blueprint("imports", __name__, url_prefix="/imports")


@imports_bp.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename:
            flash("اختر ملفًا أولًا / Select a file first.", "error")
            return redirect(url_for("imports.index"))
        extension = Path(uploaded.filename).suffix.lower()
        if extension not in {".csv", ".xlsx"}:
            flash("Unsupported file type / نوع الملف غير مدعوم", "error")
            return redirect(url_for("imports.index"))
        destination = Path(current_app.config["UPLOAD_DIR"]) / safe_filename(uploaded.filename)
        uploaded.save(destination)
        with session_scope() as db:
            db.add(ImportJob(filename=destination.name, file_sha256=sha256_file(destination), status="validated", total_rows=0, accepted_rows=0, rejected_rows=0, error_details={}, created_by_id=current_user().id if current_user() else None))
        flash("تم التحقق من الملف وتسجيل بصمته / File validated and hash recorded.", "success")
        return redirect(url_for("imports.index"))
    with session_scope() as db:
        jobs = db.scalars(select(ImportJob).order_by(desc(ImportJob.created_at)).limit(30)).all()
    return render_template("imports/index.html", jobs=jobs)
