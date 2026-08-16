from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Table,
    Column,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column("role_id", ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
    Column("permission_id", ForeignKey("permissions.id", ondelete="CASCADE"), primary_key=True),
)

user_branches = Table(
    "user_branches",
    Base.metadata,
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("branch_id", ForeignKey("branches.id", ondelete="CASCADE"), primary_key=True),
)


class Role(Base):
    __tablename__ = "roles"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name_ar: Mapped[str] = mapped_column(String(100), nullable=False)
    name_en: Mapped[str] = mapped_column(String(100), nullable=False)
    users: Mapped[list["User"]] = relationship(back_populates="role")
    permissions: Mapped[list["Permission"]] = relationship(secondary=role_permissions, back_populates="roles")


class Permission(Base):
    __tablename__ = "permissions"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name_ar: Mapped[str] = mapped_column(String(150), nullable=False)
    name_en: Mapped[str] = mapped_column(String(150), nullable=False)
    roles: Mapped[list[Role]] = relationship(secondary=role_permissions, back_populates="permissions")


class Region(Base):
    __tablename__ = "regions"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    name_ar: Mapped[str] = mapped_column(String(100), nullable=False)
    name_en: Mapped[str] = mapped_column(String(100), nullable=False)
    branches: Mapped[list["Branch"]] = relationship(back_populates="region")


class Branch(Base):
    __tablename__ = "branches"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    name_ar: Mapped[str] = mapped_column(String(120), nullable=False)
    name_en: Mapped[str] = mapped_column(String(120), nullable=False)
    city_ar: Mapped[str] = mapped_column(String(100), nullable=False)
    city_en: Mapped[str] = mapped_column(String(100), nullable=False)
    region_id: Mapped[int] = mapped_column(ForeignKey("regions.id"), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    region: Mapped[Region] = relationship(back_populates="branches")
    users: Mapped[list["User"]] = relationship(secondary=user_branches, back_populates="branches")
    sales: Mapped[list["Sale"]] = relationship(back_populates="branch")


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(180), unique=True, index=True, nullable=False)
    full_name_ar: Mapped[str] = mapped_column(String(150), nullable=False)
    full_name_en: Mapped[str] = mapped_column(String(150), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id"), nullable=False)
    locale: Mapped[str] = mapped_column(String(5), default="en", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    role: Mapped[Role] = relationship(back_populates="users")
    branches: Mapped[list[Branch]] = relationship(secondary=user_branches, back_populates="users")

    @property
    def permission_codes(self) -> set[str]:
        return {permission.code for permission in self.role.permissions}


class Category(Base):
    __tablename__ = "categories"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    name_ar: Mapped[str] = mapped_column(String(120), nullable=False)
    name_en: Mapped[str] = mapped_column(String(120), nullable=False)
    products: Mapped[list["Product"]] = relationship(back_populates="category")


class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(primary_key=True)
    sku: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name_ar: Mapped[str] = mapped_column(String(150), nullable=False)
    name_en: Mapped[str] = mapped_column(String(150), nullable=False)
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"), nullable=False)
    base_price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    category: Mapped[Category] = relationship(back_populates="products")
    sales: Mapped[list["Sale"]] = relationship(back_populates="product")


class CustomerSegment(Base):
    __tablename__ = "customer_segments"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    name_ar: Mapped[str] = mapped_column(String(120), nullable=False)
    name_en: Mapped[str] = mapped_column(String(120), nullable=False)


class Promotion(Base):
    __tablename__ = "promotions"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name_ar: Mapped[str] = mapped_column(String(150), nullable=False)
    name_en: Mapped[str] = mapped_column(String(150), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    discount_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class InventorySnapshot(Base):
    __tablename__ = "inventory_snapshots"
    id: Mapped[int] = mapped_column(primary_key=True)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
    branch_id: Mapped[int] = mapped_column(ForeignKey("branches.id"), nullable=False)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_cost: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class MarketingEvent(Base):
    __tablename__ = "marketing_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    event_date: Mapped[date] = mapped_column(Date, nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    channel: Mapped[str | None] = mapped_column(String(50))
    spend: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Sale(Base):
    __tablename__ = "sales"
    __table_args__ = (
        CheckConstraint("quantity >= 0 OR transaction_type IN ('RETURN','REFUND','CREDIT')", name="ck_sale_quantity"),
        Index("ix_sales_date_branch", "sale_date", "branch_id"),
        Index("ix_sales_date_channel", "sale_date", "channel"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    sale_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    transaction_number: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    transaction_type: Mapped[str] = mapped_column(String(30), default="SALE", nullable=False)
    branch_id: Mapped[int] = mapped_column(ForeignKey("branches.id"), index=True, nullable=False)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True, nullable=False)
    channel: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    family: Mapped[str | None] = mapped_column(String(100))
    class_name: Mapped[str | None] = mapped_column("class", String(100))
    subclass: Mapped[str | None] = mapped_column(String(100))
    franchise: Mapped[str | None] = mapped_column(String(100))
    customer_key: Mapped[str | None] = mapped_column(String(100), index=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    discount_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), default=0, nullable=False)
    discount_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    gross_sales: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    net_sales: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    vat_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), default=0, nullable=False)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    stock_quantity: Mapped[int | None] = mapped_column(Integer)
    inventory_available: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_promotion: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    seasonal_factor: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    source_row_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    branch: Mapped[Branch] = relationship(back_populates="sales")
    product: Mapped[Product] = relationship(back_populates="sales")


class Forecast(Base):
    __tablename__ = "forecasts"
    __table_args__ = (Index("ix_forecasts_target_branch", "target_date", "branch_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    branch_id: Mapped[int | None] = mapped_column(ForeignKey("branches.id"))
    predicted_sales: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    lower_bound: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    upper_bound: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    baseline_sales: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    decline_percent: Mapped[float] = mapped_column(Float, nullable=False)
    decline_probability: Mapped[float] = mapped_column(Float, nullable=False)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class DeclineAlert(Base):
    __tablename__ = "decline_alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    forecast_id: Mapped[int | None] = mapped_column(ForeignKey("forecasts.id"))
    branch_id: Mapped[int | None] = mapped_column(ForeignKey("branches.id"))
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="open", nullable=False)
    title_ar: Mapped[str] = mapped_column(String(200), nullable=False)
    title_en: Mapped[str] = mapped_column(String(200), nullable=False)
    explanation_ar: Mapped[str] = mapped_column(Text, nullable=False)
    explanation_en: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    acknowledged_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ModelRun(Base):
    __tablename__ = "model_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(60), nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    training_start: Mapped[date | None] = mapped_column(Date)
    training_end: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    feature_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)


class Recommendation(Base):
    __tablename__ = "recommendations"
    id: Mapped[int] = mapped_column(primary_key=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("decline_alerts.id"), index=True, nullable=False)
    code: Mapped[str] = mapped_column(String(60), nullable=False)
    text_ar: Mapped[str] = mapped_column(Text, nullable=False)
    text_en: Mapped[str] = mapped_column(Text, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ImportJob(Base):
    __tablename__ = "import_jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    imported_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    accepted_rows: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rejected_rows: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    error_details: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(100))
    entity_id: Mapped[str | None] = mapped_column(String(100))
    ip_address: Mapped[str | None] = mapped_column(String(64))
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class SystemSetting(Base):
    __tablename__ = "system_settings"
    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str] = mapped_column(String(30), default="string", nullable=False)
    description_ar: Mapped[str | None] = mapped_column(Text)
    description_en: Mapped[str | None] = mapped_column(Text)


class SystemHealth(Base):
    __tablename__ = "system_health"
    id: Mapped[int] = mapped_column(primary_key=True)
    component: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    details_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
