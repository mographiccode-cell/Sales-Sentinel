from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from app import create_app
from app.config import Config
from app.database import session_scope
from app.models import Alert, Permission, Recommendation, Role, User
from app.services.security import hash_password


def _app(tmp_path: Path):
    database_path = tmp_path / "functional-requirements.sqlite3"

    class TestConfig(Config):
        SECRET_KEY = "functional-requirements-secret"
        DATABASE_URL = f"sqlite:///{database_path}"
        UPLOAD_DIR = tmp_path / "uploads"
        REPORT_DIR = tmp_path / "reports"
        DEPLOYMENT_MODE = "test"
        TESTING = True

    return create_app(TestConfig)


def _session(client, user_id: int, token: str = "functional-csrf") -> str:
    with client.session_transaction() as browser_session:
        browser_session["user_id"] = user_id
        browser_session["locale"] = "en"
        browser_session["csrf_token"] = token
    return token


def _users():
    with session_scope() as db:
        admin = db.scalar(select(User).where(User.username == "admin"))
        analyst = db.scalar(select(User).where(User.username == "analyst"))
        assert admin is not None and analyst is not None
        return admin.id, analyst.id


def test_builtin_roles_cover_primary_user_journey(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()
    admin_id, analyst_id = _users()

    with session_scope() as db:
        admin = db.get(User, admin_id)
        analyst = db.get(User, analyst_id)
        assert admin is not None and analyst is not None
        assert {
            "dashboard.view",
            "sales.view",
            "forecasts.run",
            "imports.manage",
            "reports.export",
            "alerts.view",
            "branches.view_all",
        }.issubset(analyst.permission_codes)
        assert {
            "users.manage",
            "system.manage",
            "alerts.view",
            "imports.manage",
        }.issubset(admin.permission_codes)

    _session(client, analyst_id)
    assert client.get("/dashboard").status_code == 200
    assert client.get("/sales/").status_code == 200
    assert client.get("/imports/").status_code == 200
    assert client.get("/forecasts/").status_code == 200
    assert client.get("/reports/").status_code == 200
    assert client.get("/alerts/").status_code == 200


def test_admin_can_create_edit_change_role_disable_and_remove_access(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()
    admin_id, _ = _users()

    with session_scope() as db:
        analyst_role = db.scalar(select(Role).where(Role.code == "analyst"))
        admin_role = db.scalar(select(Role).where(Role.code == "admin"))
        assert analyst_role is not None and admin_role is not None
        analyst_role_id = analyst_role.id
        admin_role_id = admin_role.id

    csrf = _session(client, admin_id)
    created = client.post(
        "/admin/users",
        data={
            "csrf_token": csrf,
            "username": "manager-one",
            "email": "manager-one@example.com",
            "password": "StrongPass123!",
            "password_confirmation": "StrongPass123!",
            "role_id": str(analyst_role_id),
        },
        follow_redirects=True,
    )
    assert created.status_code == 200
    assert b"User created." in created.data

    with session_scope() as db:
        user = db.scalar(select(User).where(User.username == "manager-one"))
        assert user is not None
        user_id = user.id
        assert user.role_id == analyst_role_id
        assert user.is_active is True

    csrf = _session(client, admin_id)
    edited = client.post(
        f"/admin/users/{user_id}/edit",
        data={
            "csrf_token": csrf,
            "username": "manager-one",
            "email": "manager.updated@example.com",
            "full_name_en": "Regional Sales Manager",
            "full_name_ar": "مدير مبيعات إقليمي",
            "role_id": str(admin_role_id),
            "password": "",
        },
        follow_redirects=True,
    )
    assert edited.status_code == 200
    assert b"User updated." in edited.data

    with session_scope() as db:
        user = db.get(User, user_id)
        assert user is not None
        assert user.email == "manager.updated@example.com"
        assert user.full_name_en == "Regional Sales Manager"
        assert user.role_id == admin_role_id

    csrf = _session(client, admin_id)
    disabled = client.post(
        f"/admin/users/{user_id}/toggle",
        data={"csrf_token": csrf},
        follow_redirects=True,
    )
    assert disabled.status_code == 200
    assert b"User disabled." in disabled.data

    csrf = _session(client, admin_id)
    enabled = client.post(
        f"/admin/users/{user_id}/toggle",
        data={"csrf_token": csrf},
        follow_redirects=True,
    )
    assert enabled.status_code == 200
    assert b"User activated." in enabled.data

    csrf = _session(client, admin_id)
    removed = client.post(
        f"/admin/users/{user_id}/remove-access",
        data={"csrf_token": csrf},
        follow_redirects=True,
    )
    assert removed.status_code == 200
    assert b"historical records were retained" in removed.data

    with session_scope() as db:
        user = db.get(User, user_id)
        assert user is not None
        assert user.is_active is False


def test_admin_can_manage_roles_and_permissions(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()
    admin_id, _ = _users()

    with session_scope() as db:
        dashboard_permission = db.scalar(select(Permission).where(Permission.code == "dashboard.view"))
        sales_permission = db.scalar(select(Permission).where(Permission.code == "sales.view"))
        assert dashboard_permission is not None and sales_permission is not None
        dashboard_permission_id = dashboard_permission.id
        sales_permission_id = sales_permission.id

    csrf = _session(client, admin_id)
    response = client.post(
        "/admin/roles",
        data={
            "csrf_token": csrf,
            "code": "regional-manager",
            "name_en": "Regional Manager",
            "name_ar": "مدير إقليمي",
            "permission_id": [str(dashboard_permission_id), str(sales_permission_id)],
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Role created." in response.data

    with session_scope() as db:
        role = db.scalar(select(Role).where(Role.code == "regional-manager"))
        assert role is not None
        role_id = role.id
        assert {permission.code for permission in role.permissions} == {"dashboard.view", "sales.view"}

    csrf = _session(client, admin_id)
    updated = client.post(
        f"/admin/roles/{role_id}/permissions",
        data={"csrf_token": csrf, "permission_id": [str(dashboard_permission_id)]},
        follow_redirects=True,
    )
    assert updated.status_code == 200
    assert b"Role permissions updated." in updated.data

    with session_scope() as db:
        role = db.get(Role, role_id)
        assert role is not None
        assert {permission.code for permission in role.permissions} == {"dashboard.view"}

    csrf = _session(client, admin_id)
    deleted = client.post(
        f"/admin/roles/{role_id}/delete",
        data={"csrf_token": csrf},
        follow_redirects=True,
    )
    assert deleted.status_code == 200
    assert b"Role deleted." in deleted.data


def test_alert_center_supports_read_resolve_and_reopen(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()
    _, analyst_id = _users()

    with session_scope() as db:
        alert = Alert(
            severity="high",
            title_ar="إنذار انخفاض",
            title_en="Sales decline warning",
            message_ar="إشارة اختبار",
            message_en="Functional acceptance signal",
        )
        db.add(alert)
        db.flush()
        db.add(Recommendation(
            alert_id=alert.id,
            factor_code="customer_decline",
            text_ar="راجع العملاء المتوقفين.",
            text_en="Review customers who stopped purchasing.",
            rationale_ar="اختبار قبول وظيفي.",
            rationale_en="Functional acceptance recommendation.",
            priority=1,
        ))
        alert_id = alert.id

    csrf = _session(client, analyst_id)
    page = client.get("/alerts/")
    assert page.status_code == 200
    assert b"Sales decline warning" in page.data
    assert b"Review customers who stopped purchasing." in page.data

    read = client.post(
        f"/alerts/{alert_id}/read",
        data={"csrf_token": csrf, "status": "active"},
        follow_redirects=True,
    )
    assert read.status_code == 200
    with session_scope() as db:
        assert db.get(Alert, alert_id).is_read is True

    csrf = _session(client, analyst_id)
    resolved = client.post(
        f"/alerts/{alert_id}/resolve",
        data={"csrf_token": csrf, "status": "all"},
        follow_redirects=True,
    )
    assert resolved.status_code == 200
    with session_scope() as db:
        assert db.get(Alert, alert_id).is_resolved is True

    csrf = _session(client, analyst_id)
    reopened = client.post(
        f"/alerts/{alert_id}/reopen",
        data={"csrf_token": csrf, "status": "resolved"},
        follow_redirects=True,
    )
    assert reopened.status_code == 200
    with session_scope() as db:
        assert db.get(Alert, alert_id).is_resolved is False


def test_custom_role_cannot_bypass_permission_by_typing_url(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()

    with session_scope() as db:
        dashboard_permission = db.scalar(select(Permission).where(Permission.code == "dashboard.view"))
        assert dashboard_permission is not None
        role = Role(code="dashboard-only", name_ar="لوحة فقط", name_en="Dashboard only", permissions=[dashboard_permission])
        db.add(role)
        db.flush()
        user = User(
            username="limited-user",
            email="limited@example.com",
            full_name_ar="مستخدم محدود",
            full_name_en="Limited User",
            password_hash=hash_password("LimitedPass123!"),
            role_id=role.id,
            locale="en",
            is_active=True,
        )
        db.add(user)
        db.flush()
        user_id = user.id

    _session(client, user_id)
    assert client.get("/dashboard").status_code == 200
    assert client.get("/sales/").status_code == 403
    assert client.get("/imports/").status_code == 403
    assert client.get("/forecasts/").status_code == 403
    assert client.get("/reports/").status_code == 403
    assert client.get("/alerts/").status_code == 403
    assert client.get("/admin/users").status_code == 403
    assert client.get("/admin/health").status_code == 403
