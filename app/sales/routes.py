from __future__ import annotations

from datetime import date

from flask import Blueprint, render_template, request
from sqlalchemy import case, func, select

from app.database import session_scope
from app.models import Sale
from app.services.security import login_required

sales_bp = Blueprint("sales", __name__, url_prefix="/sales")


@sales_bp.get("/")
@login_required
def index():
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    channel = request.args.get("channel", "").strip()
    stmt = select(
        Sale.sale_date,
        func.sum(Sale.net_sales).label("net_sales"),
        func.sum(Sale.gross_sales).label("total_amount"),
        func.sum(Sale.quantity).label("quantity"),
        func.count(func.distinct(Sale.transaction_number)).label("transactions"),
        func.sum(case((Sale.gross_sales > Sale.net_sales, Sale.gross_sales - Sale.net_sales), else_=0)).label("returns"),
        func.sum(func.abs(Sale.discount_amount)).label("discounts"),
    )
    if start:
        stmt = stmt.where(Sale.sale_date >= date.fromisoformat(start))
    if end:
        stmt = stmt.where(Sale.sale_date <= date.fromisoformat(end))
    if channel:
        stmt = stmt.where(Sale.channel == channel)
    stmt = stmt.group_by(Sale.sale_date).order_by(Sale.sale_date.desc()).limit(180)
    with session_scope() as db:
        rows = db.execute(stmt).all()
        channels = db.scalars(select(Sale.channel).distinct().order_by(Sale.channel)).all()
    items = [{"date": row.sale_date, "net_sales": float(row.net_sales or 0), "total_amount": float(row.total_amount or 0), "quantity": int(row.quantity or 0), "transactions": int(row.transactions or 0), "returns": float(row.returns or 0), "discounts": float(row.discounts or 0)} for row in rows]
    return render_template("sales/index.html", items=items, channels=channels, filters={"start": start, "end": end, "channel": channel})
