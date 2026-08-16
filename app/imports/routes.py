from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from sqlalchemy import desc, select

from app.database import session_scope
from app.models import ImportJob
from app.services.sales_importer import ingest_csv, inspect_csv
from app.services.security import current_user, login_required, permission_required, safe_filename, sha256_file
from app.services.tabular_upload import normalize_tabular_upload

imports_bp = Blueprint("imports", __name__, url_prefix="/imports")


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
        if extension not in {".csv", ".xlsx"}:
            flash("الصيغ المدعومة هي CSV وXLSX فقط / Only CSV and XLSX are supported.", "error")
            return redirect(url_for("imports.index"))

        destination = Path(current_app.config["UPLOAD_DIR"]) / safe_filename(uploaded.filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        uploaded.save(destination)
        original_sha256 = sha256_file(destination)
        working_path = destination
        generated_csv = False

        try:
            working_path, generated_csv = normalize_tabular_upload(destination)
            mode, total, accepted, validation_errors = inspect_csv(working_path)
            validation_rejected = total - accepted

            if not accepted:
                with session_scope() as db:
                    db.add(ImportJob(
                        filename=destination.name,
                        file_sha256=original_sha256,
                        status="failed",
                        total_rows=total,
                        accepted_rows=0,
                        rejected_rows=validation_rejected,
                        error_details={
                            "mode": mode,
                            "source_format": extension.lstrip("."),
                            "errors": validation_errors,
                        },
                        created_by_id=current_user().id if current_user() else None,
                    ))
                flash("فشل التحقق من بنية الملف / File validation failed.", "error")
                return redirect(url_for("imports.index"))

            with session_scope() as db:
                job = ImportJob(
                    filename=destination.name,
                    file_sha256=original_sha256,
                    status="importing",
                    total_rows=total,
                    accepted_rows=0,
                    rejected_rows=validation_rejected,
                    error_details={
                        "mode": mode,
                        "source_format": extension.lstrip("."),
                        "validation_errors": validation_errors,
                    },
                    created_by_id=current_user().id if current_user() else None,
                )
                db.add(job)
                db.flush()
                result = ingest_csv(db, working_path, job.id, mode)
                job.status = "imported" if result["inserted_rows"] else "imported_no_new_rows"
                job.accepted_rows = result["inserted_rows"]
                job.rejected_rows = validation_rejected + result["rejected_rows"]
                job.error_details = {
                    "mode": mode,
                    "source_format": extension.lstrip("."),
                    "validated_rows": accepted,
                    "inserted_rows": result["inserted_rows"],
                    "duplicate_rows": result["duplicate_rows"],
                    "validation_errors": validation_errors,
                    "ingestion_errors": result["errors"],
                }
                inserted = result["inserted_rows"]
                duplicates = result["duplicate_rows"]

            if mode == "daily":
                flash(
                    f"تم استيراد {inserted} يوم. هذا وضع مبسط ولا يستخدم V18 إلا عند توفر تفاصيل الفواتير والعملاء والمنتجات. / Imported {inserted} daily rows in minimal mode.",
                    "success",
                )
            else:
                flash(
                    f"تم استيراد {inserted} سجل معاملات فعلي؛ المكرر المتجاهل {duplicates}. / Imported {inserted} transaction rows; ignored {duplicates} duplicates.",
                    "success",
                )
        except Exception as exc:
            try:
                with session_scope() as db:
                    db.add(ImportJob(
                        filename=destination.name,
                        file_sha256=original_sha256,
                        status="failed",
                        total_rows=0,
                        accepted_rows=0,
                        rejected_rows=0,
                        error_details={"source_format": extension.lstrip("."), "error": str(exc)},
                        created_by_id=current_user().id if current_user() else None,
                    ))
            except Exception:
                pass
            flash(f"تعذر إكمال الاستيراد: {exc}", "error")
        finally:
            if generated_csv and working_path != destination:
                working_path.unlink(missing_ok=True)

        return redirect(url_for("imports.index"))

    with session_scope() as db:
        jobs = db.scalars(select(ImportJob).order_by(desc(ImportJob.created_at)).limit(30)).all()
    return render_template("imports/index.html", jobs=jobs)
