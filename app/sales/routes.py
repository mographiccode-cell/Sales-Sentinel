from __future__ import annotations

from datetime import date

from flask import Blueprint, flash, render_template, request, session
from sqlalchemy import case, func, select

from app.database import session_scope
from app.models import Category, Product, Sale
from app.services.data_scope import preferred_sales_condition
from app.services.security import branch_ids_for_user, current_user, login_required, permission_required

sales_bp = Blueprint("sales", __name__, url_prefix="/sales")


def _optional_int(value: str) -> int | None:
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


@sales_bp.get("/")
@login_required
@permission_required("sales.view")
def index():
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    channel = request.args.get("channel", "").strip()
    product_raw = request.args.get("product", "").strip()
    category_raw = request.args.get("category", "").strip()
    product_id = _optional_int(product_raw)
    category_id = _optional_int(category_raw)
    locale = session.get("locale", "en")

    try:
        start_date = date.fromisoformat(start) if start else None
        end_date = date.fromisoformat(end) if end else None
    except ValueError:
        flash(
            "Invalid date. Use a valid start and end date."
            if locale == "en"
            else "التاريخ غير صالح. استخدم تاريخ بداية ونهاية صحيحين.",
            "error",
        )
        start = ""
        end = ""
        start_date = None
        end_date = None

    if start_date and end_date and start_date > end_date:
        flash(
            "Start date cannot be after end date."
            if locale == "en"
            else "لا يمكن أن يكون تاريخ البداية بعد تاريخ النهاية.",
            "error",
        )
        start = ""
        end = ""
        start_date = None
        end_date = None

    if product_raw and product_id is None:
        flash("Invalid product filter." if locale == "en" else "فلتر المنتج غير صالح.", "error")
        product_raw = ""
    if category_raw and category_id is None:
        flash("Invalid category filter." if locale == "en" else "فلتر الفئة غير صالح.", "error")
        category_raw = ""

    user = current_user()
    allowed_branch_ids = branch_ids_for_user(user) if user else set()

    with session_scope() as db:
        sales_condition, data_mode, _ = preferred_sales_condition(db)
        stmt = (
            select(
                Sale.sale_date,
                func.sum(Sale.net_sales).label("net_sales"),
                func.sum(Sale.gross_sales).label("total_amount"),
                func.sum(Sale.quantity).label("quantity"),
                func.count(func.distinct(Sale.transaction_number)).label("transactions"),
                func.sum(
                    case(
                        (Sale.gross_sales > Sale.net_sales, Sale.gross_sales - Sale.net_sales),
                        else_=0,
                    )
                ).label("returns"),
                func.sum(func.abs(Sale.discount_amount)).label("discounts"),
            )
            .join(Product, Product.id == Sale.product_id)
            .where(sales_condition)
        )
        if allowed_branch_ids:
            stmt = stmt.where(Sale.branch_id.in_(allowed_branch_ids))
        if start_date:
            stmt = stmt.where(Sale.sale_date >= start_date)
        if end_date:
            stmt = stmt.where(Sale.sale_date <= end_date)
        if channel:
            stmt = stmt.where(Sale.channel == channel)
        if product_id:
            stmt = stmt.where(Sale.product_id == product_id)
        if category_id:
            stmt = stmt.where(Product.category_id == category_id)
        stmt = stmt.group_by(Sale.sale_date).order_by(Sale.sale_date.desc()).limit(180)
        rows = db.execute(stmt).all()

        channel_stmt = select(Sale.channel).where(sales_condition)
        product_stmt = (
            select(Product.id, Product.name_en, Product.name_ar)
            .join(Sale, Sale.product_id == Product.id)
            .where(sales_condition)
            .distinct()
            .order_by(Product.name_en)
        )
        category_stmt = (
            select(Category.id, Category.name_en, Category.name_ar)
            .join(Product, Product.category_id == Category.id)
            .join(Sale, Sale.product_id == Product.id)
            .where(sales_condition)
            .distinct()
            .order_by(Category.name_en)
        )
        if allowed_branch_ids:
            channel_stmt = channel_stmt.where(Sale.branch_id.in_(allowed_branch_ids))
            product_stmt = product_stmt.where(Sale.branch_id.in_(allowed_branch_ids))
            category_stmt = category_stmt.where(Sale.branch_id.in_(allowed_branch_ids))

        channels = db.scalars(channel_stmt.distinct().order_by(Sale.channel)).all()
        products = [
            {"id": row.id, "name_en": row.name_en, "name_ar": row.name_ar}
            for row in db.execute(product_stmt).all()
        ]
        categories = [
            {"id": row.id, "name_en": row.name_en, "name_ar": row.name_ar}
            for row in db.execute(category_stmt).all()
        ]

    items = [
        {
            "date": row.sale_date,
            "net_sales": float(row.net_sales or 0),
            "total_amount": float(row.total_amount or 0),
            "quantity": int(row.quantity or 0),
            "transactions": int(row.transactions or 0),
            "returns": float(row.returns or 0),
            "discounts": float(row.discounts or 0),
        }
        for row in rows
    ]
    return render_template(
        "sales/index.html",
        items=items,
        channels=channels,
        products=products,
        categories=categories,
        filters={
            "start": start,
            "end": end,
            "channel": channel,
            "product": product_raw,
            "category": category_raw,
        },
        data_mode=data_mode,
    )
