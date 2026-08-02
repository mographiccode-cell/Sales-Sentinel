from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from sqlalchemy import or_, select

from app.database import session_scope
from app.models import User
from app.services.audit import write_audit
from app.services.i18n import t
from app.services.security import (
    clear_login_attempts,
    current_user,
    login_rate_limited,
    needs_rehash,
    record_login_attempt,
    verify_password,
    hash_password,
)

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("dashboard.index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        identifier = f"{request.remote_addr}:{username}"
        if login_rate_limited(identifier):
            flash(t("rate_limited"), "error")
            return render_template("auth/login.html"), 429
        with session_scope() as db:
            user = db.scalar(select(User).where(or_(User.username == username, User.email == username)))
            if not user or not user.is_active or not verify_password(user.password_hash, password):
                record_login_attempt(identifier)
                write_audit(db, "auth.login_failed", user_id=user.id if user else None, details={"username": username})
                flash(t("invalid_credentials"), "error")
                return render_template("auth/login.html"), 401
            clear_login_attempts(identifier)
            if needs_rehash(user.password_hash):
                user.password_hash = hash_password(password)
            user.last_login_at = datetime.now(timezone.utc)
            session.clear()
            session["user_id"] = user.id
            session["locale"] = user.locale
            session["csrf_token"] = __import__("secrets").token_urlsafe(32)
            write_audit(db, "auth.login_success", user_id=user.id)
        next_url = request.args.get("next")
        if not next_url or not next_url.startswith("/") or next_url.startswith("//"):
            next_url = url_for("dashboard.index")
        return redirect(next_url)
    return render_template("auth/login.html")


@auth_bp.post("/logout")
def logout():
    user = current_user()
    if user:
        with session_scope() as db:
            write_audit(db, "auth.logout", user_id=user.id)
    session.clear()
    return redirect(url_for("auth.login"))
