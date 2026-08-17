from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import defaultdict, deque
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from flask import abort, current_app, redirect, request, session, url_for
from sqlalchemy import select

from app.database import SessionLocal
from app.models import User

P = ParamSpec("P")
R = TypeVar("R")
_password_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)
_login_attempts: dict[str, deque[float]] = defaultdict(deque)


def hash_password(password: str) -> str:
    if len(password) < 10:
        raise ValueError("Password must contain at least 10 characters")
    return _password_hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    try:
        return _password_hasher.check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


def csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def validate_csrf() -> None:
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        supplied = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        expected = session.get("csrf_token")
        if not supplied or not expected or not hmac.compare_digest(str(supplied), str(expected)):
            abort(400, description="Invalid CSRF token")


def current_user() -> User | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    db = SessionLocal()
    try:
        user = db.scalar(select(User).where(User.id == int(user_id), User.is_active.is_(True)))
        if user:
            _ = user.role.permissions
            _ = user.branches
            db.expunge(user)
        return user
    finally:
        db.close()


def login_required(view: Callable[P, R]) -> Callable[P, R]:
    @wraps(view)
    def wrapped(*args: P.args, **kwargs: P.kwargs):
        if current_user() is None:
            return redirect(url_for("auth.login", next=request.full_path))
        return view(*args, **kwargs)
    return wrapped


def permission_required(permission: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(view: Callable[P, R]) -> Callable[P, R]:
        @wraps(view)
        def wrapped(*args: P.args, **kwargs: P.kwargs):
            user = current_user()
            if not user:
                return redirect(url_for("auth.login"))
            if permission not in user.permission_codes:
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorator


def branch_ids_for_user(user: User) -> set[int] | None:
    """Return the user's data scope.

    ``None`` means explicit all-branch access. An empty set means the user has
    no assigned branches and must therefore receive no branch-scoped data.
    Keeping these states distinct prevents an empty assignment from becoming
    an accidental unrestricted query.
    """
    if "branches.view_all" in user.permission_codes:
        return None
    return {branch.id for branch in user.branches}


def record_login_attempt(identifier: str) -> None:
    now = time.monotonic()
    window = int(current_app.config["LOGIN_RATE_WINDOW_SECONDS"])
    queue = _login_attempts[identifier]
    queue.append(now)
    while queue and now - queue[0] > window:
        queue.popleft()


def login_rate_limited(identifier: str) -> bool:
    now = time.monotonic()
    window = int(current_app.config["LOGIN_RATE_WINDOW_SECONDS"])
    limit = int(current_app.config["LOGIN_RATE_LIMIT"])
    queue = _login_attempts[identifier]
    while queue and now - queue[0] > window:
        queue.popleft()
    return len(queue) >= limit


def clear_login_attempts(identifier: str) -> None:
    _login_attempts.pop(identifier, None)


def safe_filename(filename: str) -> str:
    name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    cleaned = "".join(ch if ch in allowed else "_" for ch in name)
    return cleaned[:180] or "upload"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
