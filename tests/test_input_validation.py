from __future__ import annotations

from io import BytesIO
from pathlib import Path

from sqlalchemy import select

from app import create_app
from app.config import Config
from app.database import session_scope
from app.models import Role, User


def _app(tmp_path: Path):
    database_path = tmp_path / "validation.sqlite3"

    class TestConfig(Config):
        SECRET_KEY = "validation-secret"
        DATABASE_URL = f"sqlite:///{database_path}"
        UPLOAD_DIR = tmp_path / "uploads"
        REPORT_DIR = tmp_path / "reports"
        DEPLOYMENT_MODE = "test"
        TESTING = True

    return create_app(TestConfig)


def _csrf(client, token: str = "validation-csrf") -> str:
    with client.session_transaction() as browser_session:
        browser_session["locale"] = "en"
        browser_session["csrf_token"] = token
    return token


def _admin_session(client, admin_id: int, token: str = "validation-csrf") -> str:
    with client.session_transaction() as browser_session:
        browser_session["user_id"] = admin_id
        browser_session["locale"] = "en"
        browser_session["csrf_token"] = token
    return token


def test_invalid_username_or_password_message(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()
    csrf = _csrf(client)

    response = client.post(
        "/auth/login",
        data={
            "csrf_token": csrf,
            "username": "not-a-user",
            "password": "wrong-password",
        },
    )

    assert response.status_code == 401
    assert b"The username or password is incorrect." in response.data
    assert b"flash error" in response.data


def test_account_password_validation_messages(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()

    with session_scope() as db:
        admin = db.scalar(select(User).where(User.username == "admin"))
        role = db.scalar(select(Role).order_by(Role.id))
        assert admin is not None
        assert role is not None
        admin_id = admin.id
        role_id = role.id

    csrf = _admin_session(client, admin_id)
    weak = client.post(
        "/admin/users",
        data={
            "csrf_token": csrf,
            "username": "weakuser",
            "email": "weak@example.com",
            "password": "123",
            "password_confirmation": "123",
            "role_id": str(role_id),
        },
        follow_redirects=True,
    )
    assert weak.status_code == 200
    assert b"Password must be at least 10 characters." in weak.data

    csrf = _admin_session(client, admin_id)
    mismatch = client.post(
        "/admin/users",
        data={
            "csrf_token": csrf,
            "username": "mismatchuser",
            "email": "mismatch@example.com",
            "password": "StrongPass123!",
            "password_confirmation": "DifferentPass123!",
            "role_id": str(role_id),
        },
        follow_redirects=True,
    )
    assert mismatch.status_code == 200
    assert b"Password confirmation does not match." in mismatch.data


def test_in_app_validation_messages(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()

    with session_scope() as db:
        admin = db.scalar(select(User).where(User.username == "admin"))
        assert admin is not None
        admin_id = admin.id

    csrf = _admin_session(client, admin_id)
    missing = client.post(
        "/imports/",
        data={"csrf_token": csrf},
        follow_redirects=True,
    )
    assert missing.status_code == 200
    assert b"Select a file first." in missing.data

    csrf = _admin_session(client, admin_id)
    unsupported = client.post(
        "/imports/",
        data={
            "csrf_token": csrf,
            "file": (BytesIO(b"not supported"), "notes.txt"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert unsupported.status_code == 200
    assert b"Only CSV and XLSX files are supported." in unsupported.data

    _admin_session(client, admin_id)
    invalid_date = client.get("/sales/?start=not-a-date&end=2024-04-01")
    assert invalid_date.status_code == 200
    assert b"Invalid date. Use a valid start and end date." in invalid_date.data
