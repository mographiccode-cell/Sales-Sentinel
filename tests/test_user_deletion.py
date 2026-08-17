from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from app import create_app
from app.config import Config
from app.database import session_scope
from app.models import ModelRun, Role, User
from app.services.security import hash_password


def _app(tmp_path: Path):
    database_path = tmp_path / "user-deletion.sqlite3"

    class TestConfig(Config):
        SECRET_KEY = "user-deletion-secret"
        DATABASE_URL = f"sqlite:///{database_path}"
        UPLOAD_DIR = tmp_path / "uploads"
        REPORT_DIR = tmp_path / "reports"
        DEPLOYMENT_MODE = "test"
        TESTING = True

    return create_app(TestConfig)


def _admin_session(client) -> tuple[int, str]:
    with session_scope() as db:
        admin = db.scalar(select(User).where(User.username == "admin"))
        assert admin is not None
        admin_id = admin.id
    token = "delete-user-csrf"
    with client.session_transaction() as browser_session:
        browser_session["user_id"] = admin_id
        browser_session["locale"] = "en"
        browser_session["csrf_token"] = token
    return admin_id, token


def _new_user(username: str) -> int:
    with session_scope() as db:
        role = db.scalar(select(Role).where(Role.code == "analyst"))
        assert role is not None
        user = User(
            username=username,
            email=f"{username}@example.com",
            full_name_ar=username,
            full_name_en=username,
            password_hash=hash_password("DeletePass123!"),
            role_id=role.id,
            locale="en",
            is_active=True,
        )
        db.add(user)
        db.flush()
        return user.id


def test_admin_can_physically_delete_unused_user(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()
    _, token = _admin_session(client)
    user_id = _new_user("unused-user")

    response = client.post(
        f"/admin/users/{user_id}/delete",
        data={"csrf_token": token},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"User deleted." in response.data
    with session_scope() as db:
        assert db.get(User, user_id) is None


def test_historical_user_is_disabled_instead_of_deleted(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()
    _, token = _admin_session(client)
    user_id = _new_user("historical-user")

    with session_scope() as db:
        db.add(ModelRun(
            model_name="history-test",
            model_version="1",
            status="completed",
            horizon_days=7,
            filters_json={},
            metrics_json={},
            sample_size=0,
            created_by_id=user_id,
        ))

    response = client.post(
        f"/admin/users/{user_id}/delete",
        data={"csrf_token": token},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"disabled instead of deleted" in response.data
    with session_scope() as db:
        user = db.get(User, user_id)
        assert user is not None
        assert user.is_active is False
