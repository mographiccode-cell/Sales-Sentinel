from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from sqlalchemy import select

from app.database import session_scope
from app.models import SystemSetting
from app.services.security import login_required, permission_required

settings_bp = Blueprint("settings_pref", __name__, url_prefix="/admin/system-settings")


def _message(en: str, ar: str) -> str:
    return en if session.get("locale", "en") == "en" else ar


def _ensure_setting(db, key: str, default: str, value_type: str, en: str, ar: str) -> SystemSetting:
    setting = db.scalar(select(SystemSetting).where(SystemSetting.key == key))
    if setting is None:
        setting = SystemSetting(
            key=key,
            value=default,
            value_type=value_type,
            description_en=en,
            description_ar=ar,
        )
        db.add(setting)
        db.flush()
    return setting


@settings_bp.route("/", methods=["GET", "POST"])
@login_required
@permission_required("system.manage")
def settings():
    with session_scope() as db:
        horizon_setting = _ensure_setting(
            db,
            "default_forecast_horizon",
            "7",
            "integer",
            "Default horizon used by the manual forecast form.",
            "الأفق الافتراضي المستخدم في نموذج التوقع اليدوي.",
        )

        if request.method == "POST":
            horizon_raw = request.form.get("default_forecast_horizon", "7").strip()
            try:
                horizon = int(horizon_raw)
            except ValueError:
                horizon = 0
            if horizon not in {7, 30}:
                flash(_message("Forecast horizon must be 7 or 30 days.", "يجب أن يكون أفق التوقع 7 أو 30 يومًا."), "error")
                return redirect(url_for("settings_pref.settings"))
            horizon_setting.value = str(horizon)
            flash(_message("System settings saved.", "تم حفظ إعدادات النظام."), "success")
            return redirect(url_for("settings_pref.settings"))

        default_horizon = int(horizon_setting.value) if horizon_setting.value in {"7", "30"} else 7

    return render_template(
        "admin/system_settings.html",
        default_horizon=default_horizon,
    )
