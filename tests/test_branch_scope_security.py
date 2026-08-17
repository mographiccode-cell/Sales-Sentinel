from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from app import create_app
from app.config import Config
from app.database import session_scope
from app.models import Permission, Role, User
from app.services.dashboard_service import dashboard_summary
from app.services.security import branch_ids_for_user, hash_password


def _app(tmp_path: Path):
    database_path = tmp_path / "branch-scope-security.sqlite3"

    class TestConfig(Config):
        SECRET_KEY = "branch-scope-security-secret"
        DATABASE_URL = f"sqlite:///{database_path}"
        UPLOAD_DIR = tmp_path / "uploads"
        REPORT_DIR = tmp_path / "reports"
        DEPLOYMENT_MODE = "test"
        TESTING = True

    return create_app(TestConfig)


def test_empty_branch_assignment_is_not_all_branch_access(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()

    with session_scope() as db:
        wanted = {
            "dashboard.view",
            "sales.view",
            "imports.manage",
            "forecasts.run",
            "reports.export",
            "alerts.view",
        }
        permissions = db.scalars(select(Permission).where(Permission.code.in_(wanted))).all()
        role = Role(
            code="branch-limited",
            name_ar="مستخدم محدود بالفروع",
            name_en="Branch limited",
            permissions=permissions,
        )
        db.add(role)
        db.flush()
        user = User(
            username="branch-limited-user",
            email="branch-limited@example.com",
            full_name_ar="مستخدم محدود",
            full_name_en="Branch Limited User",
            password_hash=hash_password("BranchLimited123!"),
            role_id=role.id,
            locale="en",
            is_active=True,
        )
        db.add(user)
        db.flush()
        user_id = user.id

    with client.session_transaction() as browser_session:
        browser_session["user_id"] = user_id
        browser_session["locale"] = "en"
        browser_session["csrf_token"] = "branch-scope-csrf"

    with session_scope() as db:
        user = db.get(User, user_id)
        assert user is not None
        _ = user.role.permissions
        _ = user.branches
        scope = branch_ids_for_user(user)
        assert scope == set()
        summary = dashboard_summary(db, scope)
        assert summary["total_records"] == 0
        assert summary["current_sales"] == 0
        assert summary["forecast_sales"] == 0
        assert summary["decline_probability"] == 0

    sales = client.get("/sales/")
    assert sales.status_code == 200
    assert b"No results" in sales.data

    # Current import, V18/Adaptive forecast, report, and alert artifacts are
    # company-scope. A branch-limited role must never receive those global
    # outputs merely because it has the feature permission.
    assert client.get("/imports/").status_code == 403
    assert client.get("/forecasts/").status_code == 403
    assert client.get("/reports/").status_code == 403
    assert client.get("/alerts/").status_code == 403
