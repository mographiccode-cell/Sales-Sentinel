from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, url_for
from sqlalchemy import select

from app.database import session_scope
from app.models import Role, SystemHealth, SystemSetting, User
from app.services.security import hash_password, login_required, permission_required

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.route("/users", methods=["GET"])
@login_required
@permission_required("users.manage")
def users():
    with session_scope() as db:
        rows = db.scalars(select(User).order_by(User.created_at)).all()
        roles = db.scalars(select(Role).order_by(Role.id)).all()
        for user in rows:
            _ = user.role
    return render_template("admin/users.html", users=rows, roles=roles)


@admin_bp.post("/users")
@login_required
@permission_required("users.manage")
def create_user():
    username = request.form.get("username", "").strip().lower()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    role_id = int(request.form.get("role_id", "0"))
    if not username or not email or len(password) < 10:
        flash("تحقق من البيانات وكلمة المرور / Check the data and password.", "error")
        return redirect(url_for("admin.users"))
    with session_scope() as db:
        role = db.get(Role, role_id)
        if not role:
            flash("Invalid role", "error")
            return redirect(url_for("admin.users"))
        db.add(User(username=username, email=email, full_name_ar=username, full_name_en=username, password_hash=hash_password(password), role_id=role.id, locale="en"))
    flash("تم إنشاء المستخدم / User created.", "success")
    return redirect(url_for("admin.users"))


@admin_bp.get("/health")
@login_required
@permission_required("system.manage")
def health():
    import psutil
    with session_scope() as db:
        checks = db.scalars(select(SystemHealth).order_by(SystemHealth.component)).all()
    disk = psutil.disk_usage("/")
    memory = psutil.virtual_memory()
    resources = {"cpu_percent": psutil.cpu_percent(), "memory_percent": memory.percent, "memory_available_gb": round(memory.available / 1024**3, 2), "disk_free_gb": round(disk.free / 1024**3, 2)}
    return render_template("admin/health.html", checks=checks, resources=resources)


@admin_bp.route("/settings", methods=["GET", "POST"])
@login_required
@permission_required("system.manage")
def settings():
    with session_scope() as db:
        setting = db.scalar(select(SystemSetting).where(SystemSetting.key == "decline_threshold"))
        if request.method == "POST" and setting:
            setting.value = str(float(request.form.get("decline_threshold", "0.08")))
            flash("تم الحفظ / Saved.", "success")
            return redirect(url_for("admin.settings"))
        threshold = float(setting.value) if setting else 0.08
    return render_template("admin/settings.html", threshold=threshold)
