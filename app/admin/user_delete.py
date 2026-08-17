from __future__ import annotations

from flask import Blueprint, flash, redirect, session, url_for
from sqlalchemy import func, select

from app.database import session_scope
from app.models import AuditLog, ImportJob, ModelRun, Role, User
from app.services.security import current_user, login_required, permission_required

admin_user_bp = Blueprint("admin_user", __name__, url_prefix="/admin")


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


@admin_user_bp.post("/users/<int:user_id>/delete")
@login_required
@permission_required("users.manage")
def delete_user(user_id: int):
    """Delete unused accounts while preserving historical audit integrity."""
    actor = current_user()
    with session_scope() as db:
        user = db.get(User, user_id)
        if not user:
            flash(_message("User not found.", "المستخدم غير موجود."), "error")
            return redirect(url_for("admin.users"))
        if actor and actor.id == user.id:
            flash(_message("You cannot delete your current account.", "لا يمكنك حذف حسابك الحالي."), "error")
            return redirect(url_for("admin.users"))
        if user.role.code == "admin" and user.is_active and _active_admin_count(db) <= 1:
            flash(_message("The last active administrator cannot be deleted.", "لا يمكن حذف آخر مسؤول نشط."), "error")
            return redirect(url_for("admin.users"))

        historical_references = sum(
            int(count or 0)
            for count in (
                db.scalar(select(func.count(ImportJob.id)).where(ImportJob.created_by_id == user.id)),
                db.scalar(select(func.count(ModelRun.id)).where(ModelRun.created_by_id == user.id)),
                db.scalar(select(func.count(AuditLog.id)).where(AuditLog.user_id == user.id)),
            )
        )
        if historical_references:
            user.is_active = False
            flash(
                _message(
                    "This user has historical records, so the account was disabled instead of deleted.",
                    "لدى المستخدم سجلات تاريخية؛ لذلك تم تعطيل الحساب بدل حذفه للحفاظ على سلامة السجلات.",
                ),
                "success",
            )
            return redirect(url_for("admin.users"))

        db.delete(user)

    flash(_message("User deleted.", "تم حذف المستخدم."), "success")
    return redirect(url_for("admin.users"))
