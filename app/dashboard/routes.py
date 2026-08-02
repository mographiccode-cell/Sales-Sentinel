from __future__ import annotations

import json
from flask import Blueprint, render_template

from app.database import session_scope
from app.services.dashboard_service import dashboard_summary
from app.services.security import branch_ids_for_user, current_user, login_required

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
@login_required
def index():
    user = current_user()
    with session_scope() as db:
        summary = dashboard_summary(db, branch_ids_for_user(user) if user else None)
    return render_template(
        "dashboard/index.html",
        summary=summary,
        sales_json=json.dumps(summary["series"], ensure_ascii=False),
        forecasts_json=json.dumps(summary["forecast_series"], ensure_ascii=False),
    )
