from __future__ import annotations

from datetime import date

from flask import Blueprint, flash, render_template, request, session
from sqlalchemy import case, func, select

from app.database import session_scope
from app.models import Sale
from app.services.data_scope import preferred_sales_condition
from app.services.security import login_required

sales_bp = Blueprint("sales", __name__, url_prefix="/sales")


@sales_bp.get("/")
@login_required
def index():
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    channel = request.args.get("channel", "").strip()
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

    with session_scope() as db:
        sales_condition, data_mode, _ = preferred_sales_condition(db)
        stmt = select(
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
        ).where(sales_condition)
        if start_date:
            stmt = stmt.where(Sale.sale_date >= start_date)
        if end_date:
            stmt = stmt.where(Sale.sale_date <= end_date)
        if channel:
            stmt = stmt.where(Sale.channel == channel)
        stmt = stmt.group_by(Sale.sale_date).order_by(Sale.sale_date.desc()).limit(180)
        rows = db.execute(stmt).all()
        channels = db.scalars(
            select(Sale.channel)
            .where(sales_condition)
            .distinct()
            .order_by(Sale.channel)
        ).all()

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
        filters={"start": start, "end": end, "channel": channel},
        data_mode=data_mode,
    )
