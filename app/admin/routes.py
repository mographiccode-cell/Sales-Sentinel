from __future__ import annotations

import re

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from sqlalchemy import func, select

from app.database import session_scope
from app.models import Permission, Role, SystemHealth, SystemSetting, User
from app.services.security import current_user, hash_password, login_required, permission_required

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def _message(en: str, ar: str) -> str:
    return en if session.get("locale", "en") == "en" else ar


def _active_admin_count(db) -> int:
    return int(
        db.scalar(
            select(func.count(User.id))
            .join(Role, User.role_id == Role.id)
            .where(User.is_active.is_(True), Role.code == "admin")
        )
        or 0
    )


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
    password_confirmation = request.form.get("password_confirmation", "")
    role_id_raw = request.form.get("role_id", "0")

    if not username or not email:
        flash(_message("Username and email are required.", "اسم المستخدم والبريد الإلكتروني مطلوبان."), "error")
        return redirect(url_for("admin.users"))
    if len(password) < 10:
        flash(_message("Password must be at least 10 characters.", "يجب أن تتكون كلمة المرور من 10 أحرف على الأقل."), "error")
        return redirect(url_for("admin.users"))
    if password != password_confirmation:
        flash(_message("Password confirmation does not match.", "تأكيد كلمة المرور غير مطابق."), "error")
        return redirect(url_for("admin.users"))
    try:
        role_id = int(role_id_raw)
    except ValueError:
        flash(_message("Invalid role.", "الدور المحدد غير صالح."), "error")
        return redirect(url_for("admin.users"))

    with session_scope() as db:
        duplicate = db.scalar(select(User).where((User.username == username) | (User.email == email)))
        if duplicate:
            flash(_message("Username or email already exists.", "اسم المستخدم أو البريد الإلكتروني مستخدم مسبقًا."), "error")
            return redirect(url_for("admin.users"))
        role = db.get(Role, role_id)
        if not role:
            flash(_message("Invalid role.", "الدور المحدد غير صالح."), "error")
            return redirect(url_for("admin.users"))
        db.add(
            User(
                username=username,
                email=email,
                full_name_ar=username,
                full_name_en=username,
                password_hash=hash_password(password),
                role_id=role.id,
                locale="en",
                is_active=True,
            )
        )

    flash(_message("User created.", "تم إنشاء المستخدم."), "success")
    return redirect(url_for("admin.users"))


@admin_bp.post("/users/<int:user_id>/edit")
@login_required
@permission_required("users.manage")
def edit_user(user_id: int):
    username = request.form.get("username", "").strip().lower()
    email = request.form.get("email", "").strip().lower()
    full_name_en = request.form.get("full_name_en", "").strip() or username
    full_name_ar = request.form.get("full_name_ar", "").strip() or username
    role_id_raw = request.form.get("role_id", "0")
    password = request.form.get("password", "")

    if not username or not email:
        flash(_message("Username and email are required.", "اسم المستخدم والبريد الإلكتروني مطلوبان."), "error")
        return redirect(url_for("admin.users"))
    try:
        role_id = int(role_id_raw)
    except ValueError:
        flash(_message("Invalid role.", "الدور المحدد غير صالح."), "error")
        return redirect(url_for("admin.users"))
    if password and len(password) < 10:
        flash(_message("New password must be at least 10 characters.", "يجب أن تتكون كلمة المرور الجديدة من 10 أحرف على الأقل."), "error")
        return redirect(url_for("admin.users"))

    actor = current_user()
    with session_scope() as db:
        user = db.get(User, user_id)
        role = db.get(Role, role_id)
        if not user or not role:
            flash(_message("User or role not found.", "المستخدم أو الدور غير موجود."), "error")
            return redirect(url_for("admin.users"))
        duplicate = db.scalar(
            select(User).where(User.id != user.id, (User.username == username) | (User.email == email))
        )
        if duplicate:
            flash(_message("Username or email already exists.", "اسم المستخدم أو البريد الإلكتروني مستخدم مسبقًا."), "error")
            return redirect(url_for("admin.users"))
        if user.role.code == "admin" and role.code != "admin" and _active_admin_count(db) <= 1:
            flash(_message("The last active administrator cannot lose the administrator role.", "لا يمكن إزالة دور المسؤول من آخر مسؤول نشط."), "error")
            return redirect(url_for("admin.users"))
        if actor and actor.id == user.id and role.code != "admin":
            flash(_message("You cannot remove your own administrator role.", "لا يمكنك إزالة صلاحية المسؤول من حسابك الحالي."), "error")
            return redirect(url_for("admin.users"))

        user.username = username
        user.email = email
        user.full_name_en = full_name_en
        user.full_name_ar = full_name_ar
        user.role_id = role.id
        if password:
            user.password_hash = hash_password(password)

    flash(_message("User updated.", "تم تحديث المستخدم."), "success")
    return redirect(url_for("admin.users"))


@admin_bp.post("/users/<int:user_id>/toggle")
@login_required
@permission_required("users.manage")
def toggle_user(user_id: int):
    actor = current_user()
    with session_scope() as db:
        user = db.get(User, user_id)
        if not user:
            flash(_message("User not found.", "المستخدم غير موجود."), "error")
            return redirect(url_for("admin.users"))
        if actor and actor.id == user.id:
            flash(_message("You cannot disable your current account.", "لا يمكنك تعطيل حسابك الحالي."), "error")
            return redirect(url_for("admin.users"))
        if user.is_active and user.role.code == "admin" and _active_admin_count(db) <= 1:
            flash(_message("The last active administrator cannot be disabled.", "لا يمكن تعطيل آخر مسؤول نشط."), "error")
            return redirect(url_for("admin.users"))
        user.is_active = not user.is_active
        active = user.is_active

    flash(_message("User activated." if active else "User disabled.", "تم تفعيل المستخدم." if active else "تم تعطيل المستخدم."), "success")
    return redirect(url_for("admin.users"))


@admin_bp.post("/users/<int:user_id>/remove-access")
@login_required
@permission_required("users.manage")
def remove_user_access(user_id: int):
    """Soft-remove access to retain report/audit foreign-key history."""
    actor = current_user()
    with session_scope() as db:
        user = db.get(User, user_id)
        if not user:
            flash(_message("User not found.", "المستخدم غير موجود."), "error")
            return redirect(url_for("admin.users"))
        if actor and actor.id == user.id:
            flash(_message("You cannot remove access from your current account.", "لا يمكنك إزالة الوصول من حسابك الحالي."), "error")
            return redirect(url_for("admin.users"))
        if user.role.code == "admin" and user.is_active and _active_admin_count(db) <= 1:
            flash(_message("The last active administrator cannot be removed.", "لا يمكن إزالة آخر مسؤول نشط."), "error")
            return redirect(url_for("admin.users"))
        user.is_active = False

    flash(_message("User access removed; historical records were retained.", "تمت إزالة وصول المستخدم مع الاحتفاظ بالسجلات التاريخية."), "success")
    return redirect(url_for("admin.users"))


@admin_bp.route("/roles", methods=["GET", "POST"])
@login_required
@permission_required("users.manage")
def roles():
    if request.method == "POST":
        name_en = request.form.get("name_en", "").strip()
        name_ar = request.form.get("name_ar", "").strip() or name_en
        raw_code = request.form.get("code", "").strip().lower()
        code = re.sub(r"[^a-z0-9-]+", "-", raw_code).strip("-")
        if not code or not name_en:
            flash(_message("Role code and English name are required.", "رمز الدور والاسم الإنجليزي مطلوبان."), "error")
            return redirect(url_for("admin.roles"))
        with session_scope() as db:
            if db.scalar(select(Role).where(Role.code == code)):
                flash(_message("Role code already exists.", "رمز الدور مستخدم مسبقًا."), "error")
                return redirect(url_for("admin.roles"))
            permission_ids = [int(value) for value in request.form.getlist("permission_id") if value.isdigit()]
            permissions = db.scalars(select(Permission).where(Permission.id.in_(permission_ids))).all() if permission_ids else []
            db.add(Role(code=code, name_ar=name_ar, name_en=name_en, permissions=permissions))
        flash(_message("Role created.", "تم إنشاء الدور."), "success")
        return redirect(url_for("admin.roles"))

    with session_scope() as db:
        role_rows = db.scalars(select(Role).order_by(Role.id)).all()
        permissions = db.scalars(select(Permission).order_by(Permission.code)).all()
        for role in role_rows:
            _ = role.permissions
            _ = role.users
    return render_template("admin/roles.html", roles=role_rows, permissions=permissions)


@admin_bp.post("/roles/<int:role_id>/permissions")
@login_required
@permission_required("users.manage")
def update_role_permissions(role_id: int):
    with session_scope() as db:
        role = db.get(Role, role_id)
        if not role:
            flash(_message("Role not found.", "الدور غير موجود."), "error")
            return redirect(url_for("admin.roles"))
        if role.code == "admin":
            role.permissions = db.scalars(select(Permission).order_by(Permission.id)).all()
            flash(_message("Administrator keeps all permissions to prevent lockout.", "يحتفظ المسؤول بجميع الصلاحيات لمنع فقدان الوصول."), "success")
            return redirect(url_for("admin.roles"))
        permission_ids = [int(value) for value in request.form.getlist("permission_id") if value.isdigit()]
        role.permissions = db.scalars(select(Permission).where(Permission.id.in_(permission_ids))).all() if permission_ids else []
    flash(_message("Role permissions updated.", "تم تحديث صلاحيات الدور."), "success")
    return redirect(url_for("admin.roles"))


@admin_bp.post("/roles/<int:role_id>/delete")
@login_required
@permission_required("users.manage")
def delete_role(role_id: int):
    with session_scope() as db:
        role = db.get(Role, role_id)
        if not role:
            flash(_message("Role not found.", "الدور غير موجود."), "error")
            return redirect(url_for("admin.roles"))
        if role.code in {"admin", "analyst"}:
            flash(_message("Built-in roles cannot be deleted.", "لا يمكن حذف الأدوار الأساسية."), "error")
            return redirect(url_for("admin.roles"))
        if int(db.scalar(select(func.count(User.id)).where(User.role_id == role.id)) or 0) > 0:
            flash(_message("Move users to another role before deleting this role.", "انقل المستخدمين إلى دور آخر قبل حذف هذا الدور."), "error")
            return redirect(url_for("admin.roles"))
        db.delete(role)
    flash(_message("Role deleted.", "تم حذف الدور."), "success")
    return redirect(url_for("admin.roles"))


@admin_bp.get("/health")
@login_required
@permission_required("system.manage")
def health():
    import psutil

    with session_scope() as db:
        checks = db.scalars(select(SystemHealth).order_by(SystemHealth.component)).all()
    disk = psutil.disk_usage("/")
    memory = psutil.virtual_memory()
    resources = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": memory.percent,
        "memory_available_gb": round(memory.available / 1024**3, 2),
        "disk_free_gb": round(disk.free / 1024**3, 2),
    }
    return render_template("admin/health.html", checks=checks, resources=resources)


@admin_bp.route("/settings", methods=["GET", "POST"])
@login_required
@permission_required("system.manage")
def settings():
    with session_scope() as db:
        setting = db.scalar(select(SystemSetting).where(SystemSetting.key == "decline_threshold"))
        if request.method == "POST" and setting:
            try:
                threshold = float(request.form.get("decline_threshold", "0.08"))
            except ValueError:
                flash(_message("Decline threshold must be a number.", "يجب أن يكون حد الانخفاض رقمًا."), "error")
                return redirect(url_for("admin.settings"))
            if not 0.01 <= threshold <= 0.50:
                flash(_message("Decline threshold must be between 1% and 50%.", "يجب أن يكون حد الانخفاض بين 1% و50%."), "error")
                return redirect(url_for("admin.settings"))
            setting.value = str(threshold)
            flash(_message("Settings saved.", "تم حفظ الإعدادات."), "success")
            return redirect(url_for("admin.settings"))
        threshold = float(setting.value) if setting else 0.08
    return render_template("admin/settings.html", threshold=threshold)
