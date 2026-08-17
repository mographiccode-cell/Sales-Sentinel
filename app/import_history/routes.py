from __future__ import annotations

from flask import Blueprint, render_template
from sqlalchemy import desc, select

from app.database import session_scope
from app.models import ImportJob
from app.services.security import login_required, permission_required

import_history_bp = Blueprint("import_history", __name__, url_prefix="/imports")


@import_history_bp.get("/history")
@login_required
@permission_required("imports.manage")
def history():
    with session_scope() as db:
        jobs = db.scalars(
            select(ImportJob)
            .order_by(desc(ImportJob.created_at))
            .limit(100)
        ).all()
    return render_template("imports/history.html", jobs=jobs)
