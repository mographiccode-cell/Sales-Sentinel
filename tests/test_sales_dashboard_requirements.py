from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from app import create_app
from app.config import Config
from app.database import session_scope
from app.models import Category, Product, User


def _app(tmp_path: Path):
    database_path = tmp_path / "sales-dashboard-requirements.sqlite3"

    class TestConfig(Config):
        SECRET_KEY = "sales-dashboard-requirements-secret"
        DATABASE_URL = f"sqlite:///{database_path}"
        UPLOAD_DIR = tmp_path / "uploads"
        REPORT_DIR = tmp_path / "reports"
        DEPLOYMENT_MODE = "test"
        TESTING = True

    return create_app(TestConfig)


def _analyst_session(client, analyst_id: int):
    with client.session_transaction() as browser_session:
        browser_session["user_id"] = analyst_id
        browser_session["locale"] = "en"
        browser_session["csrf_token"] = "requirements-csrf"


def test_sales_filters_cover_date_product_category_and_channel(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()

    with session_scope() as db:
        analyst = db.scalar(select(User).where(User.username == "analyst"))
        product = db.scalar(select(Product).where(Product.sku == "DAILY-AGGREGATE"))
        category = db.scalar(select(Category).where(Category.code == "ALL"))
        assert analyst is not None and product is not None and category is not None
        analyst_id = analyst.id
        product_id = product.id
        category_id = category.id

    _analyst_session(client, analyst_id)
    response = client.get(
        f"/sales/?start=2011-12-01&end=2011-12-09&channel=Online&product={product_id}&category={category_id}"
    )

    assert response.status_code == 200
    assert b"Product" in response.data
    assert b"Category" in response.data
    assert b"Online" in response.data
    assert b"Verified daily aggregate" in response.data
    assert b"All products" in response.data
    assert b"No results" not in response.data


def test_dashboard_exposes_required_business_kpis(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()

    with session_scope() as db:
        analyst = db.scalar(select(User).where(User.username == "analyst"))
        assert analyst is not None
        analyst_id = analyst.id

    _analyst_session(client, analyst_id)
    response = client.get("/dashboard")

    assert response.status_code == 200
    for label in (
        b"Current sales",
        b"7-day forecast",
        b"Transactions",
        b"Active customers",
        b"Products",
        b"Sales channels",
        b"Average transaction value",
        b"Returns",
        b"Discounts",
    ):
        assert label in response.data
