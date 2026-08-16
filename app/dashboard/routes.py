from __future__ import annotations

import json
from flask import Blueprint, render_template

from app.database import session_scope
from app.services.dashboard_service import dashboard_summary
from app.services.security import branch_ids_for_user, current_user, login_required

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/dashboard")
@login_required
def index():
    user = current_user()
    with session_scope() as db:
        summary = dashboard_summary(db, branch_ids_for_user(user) if user else None)
    chart_values = [float(item.get('value', 0) or 0) for item in summary['series']] + [float(item.get('median', 0) or 0) for item in summary['forecast_series']]
    chart_max = max(chart_values or [1.0])
    return render_template(
        "dashboard/index.html",
        summary=summary,
        sales_json=json.dumps(summary["series"], ensure_ascii=False),
        forecasts_json=json.dumps(summary["forecast_series"], ensure_ascii=False),
        chart_max=chart_max,
    )
