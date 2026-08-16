from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select, text

from app.models import Product, Sale


_MIN_SIGNAL_PCT = 2.0


def _pct_change(current: float, previous: float) -> float:
    if abs(previous) < 1e-9:
        return 0.0
    return ((current - previous) / abs(previous)) * 100.0


def _period_totals(db, start, end) -> dict:
    row = db.execute(
        select(
            func.coalesce(func.sum(Sale.net_sales), 0),
            func.coalesce(func.sum(Sale.quantity), 0),
            func.count(func.distinct(Sale.transaction_number)),
            func.coalesce(func.sum(func.abs(Sale.discount_amount)), 0),
        )
        .where(
            Sale.source_import_id.is_not(None),
            Sale.transaction_type.notin_(["DAILY_AGGREGATE", "DAILY_IMPORT"]),
            Sale.sale_date.between(start, end),
        )
    ).one()
    sales = float(row[0] or 0.0)
    quantity = float(row[1] or 0.0)
    transactions = int(row[2] or 0)
    discounts = float(row[3] or 0.0)
    return {
        "sales": sales,
        "quantity": quantity,
        "transactions": transactions,
        "discounts": discounts,
        "avg_basket": sales / transactions if transactions else 0.0,
    }


def _active_customers(db, start, end) -> int:
    # customer_key is an additive runtime column retained by the importer for
    # transaction-level Redsea/UCI data. Raw SQL keeps compatibility with older
    # ORM mappings while still allowing a real distinct-customer comparison.
    return int(
        db.execute(
            text(
                """
                SELECT COUNT(DISTINCT customer_key)
                FROM sales
                WHERE source_import_id IS NOT NULL
                  AND transaction_type NOT IN ('DAILY_AGGREGATE','DAILY_IMPORT')
                  AND sale_date BETWEEN :start AND :end
                  AND customer_key IS NOT NULL
                  AND TRIM(customer_key) <> ''
                """
            ),
            {"start": start, "end": end},
        ).scalar_one()
        or 0
    )


def _group_sales(db, start, end, dimension: str) -> dict[str, float]:
    if dimension == "channel":
        stmt = (
            select(Sale.channel, func.sum(Sale.net_sales))
            .where(
                Sale.source_import_id.is_not(None),
                Sale.transaction_type.notin_(["DAILY_AGGREGATE", "DAILY_IMPORT"]),
                Sale.sale_date.between(start, end),
            )
            .group_by(Sale.channel)
        )
    elif dimension == "product":
        stmt = (
            select(Product.name_en, func.sum(Sale.net_sales))
            .join(Product, Product.id == Sale.product_id)
            .where(
                Sale.source_import_id.is_not(None),
                Sale.transaction_type.notin_(["DAILY_AGGREGATE", "DAILY_IMPORT"]),
                Sale.sale_date.between(start, end),
            )
            .group_by(Product.name_en)
        )
    else:
        raise ValueError("unsupported dimension")
    return {str(key or "Unknown"): float(value or 0.0) for key, value in db.execute(stmt).all()}


def _top_decliner(previous: dict[str, float], current: dict[str, float]) -> dict | None:
    candidates = []
    for key in set(previous) | set(current):
        prev = float(previous.get(key, 0.0))
        cur = float(current.get(key, 0.0))
        drop = prev - cur
        if prev > 0 and drop > 0:
            candidates.append(
                {
                    "name": key,
                    "previous": prev,
                    "current": cur,
                    "drop_amount": drop,
                    "change_pct": _pct_change(cur, prev),
                }
            )
    return max(candidates, key=lambda item: item["drop_amount"], default=None)


def _severity(change_pct: float) -> str:
    decline = abs(min(change_pct, 0.0))
    if decline >= 25:
        return "high"
    if decline >= 10:
        return "medium"
    return "low"


def _driver(code: str, title_ar: str, title_en: str, *, current: float, previous: float,
            unit: str, summary_ar: str, summary_en: str, raw_score: float,
            entity: str | None = None) -> dict:
    change = _pct_change(current, previous)
    return {
        "code": code,
        "title_ar": title_ar,
        "title_en": title_en,
        "summary_ar": summary_ar,
        "summary_en": summary_en,
        "current": current,
        "previous": previous,
        "unit": unit,
        "change_pct": change,
        "raw_score": max(0.0, raw_score),
        "strength_pct": 0.0,
        "severity": _severity(change),
        "entity": entity,
    }


def explain_decline_drivers(db, *, window: int = 7) -> dict:
    """Explain *signals associated with* the forecasted decline.

    This deliberately avoids claiming causal proof. It compares the latest
    transaction window with the immediately preceding window and ranks the
    strongest deteriorating commercial signals. Percentages shown to users are
    relative signal strength, not additive causal attribution.
    """
    last_date = db.scalar(
        select(func.max(Sale.sale_date)).where(
            Sale.source_import_id.is_not(None),
            Sale.transaction_type.notin_(["DAILY_AGGREGATE", "DAILY_IMPORT"]),
        )
    )
    if last_date is None:
        return {
            "available": False,
            "reason": "No imported transaction-level data is available for explanation.",
            "drivers": [],
        }

    current_end = last_date
    current_start = current_end - timedelta(days=window - 1)
    previous_end = current_start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=window - 1)

    current = _period_totals(db, current_start, current_end)
    previous = _period_totals(db, previous_start, previous_end)
    current_customers = _active_customers(db, current_start, current_end)
    previous_customers = _active_customers(db, previous_start, previous_end)

    channel_prev = _group_sales(db, previous_start, previous_end, "channel")
    channel_cur = _group_sales(db, current_start, current_end, "channel")
    product_prev = _group_sales(db, previous_start, previous_end, "product")
    product_cur = _group_sales(db, current_start, current_end, "product")
    channel_drop = _top_decliner(channel_prev, channel_cur)
    product_drop = _top_decliner(product_prev, product_cur)

    drivers: list[dict] = []

    quantity_change = _pct_change(current["quantity"], previous["quantity"])
    if quantity_change <= -_MIN_SIGNAL_PCT:
        drivers.append(_driver(
            "quantity_decline",
            "انخفاض الكمية المباعة",
            "Sold quantity declined",
            current=current["quantity"],
            previous=previous["quantity"],
            unit="units",
            summary_ar=f"انخفض عدد الوحدات المباعة من {previous['quantity']:.0f} إلى {current['quantity']:.0f} وحدة.",
            summary_en=f"Sold units fell from {previous['quantity']:.0f} to {current['quantity']:.0f}.",
            raw_score=abs(quantity_change) * 1.15,
        ))

    customer_change = _pct_change(float(current_customers), float(previous_customers))
    if previous_customers > 0 and customer_change <= -_MIN_SIGNAL_PCT:
        drivers.append(_driver(
            "active_customer_decline",
            "تراجع العملاء النشطين",
            "Active customers declined",
            current=float(current_customers),
            previous=float(previous_customers),
            unit="customers",
            summary_ar=f"انخفض عدد العملاء النشطين من {previous_customers} إلى {current_customers}.",
            summary_en=f"Active customers fell from {previous_customers} to {current_customers}.",
            raw_score=abs(customer_change) * 1.10,
        ))

    transaction_change = _pct_change(float(current["transactions"]), float(previous["transactions"]))
    if current["transactions"] and transaction_change <= -_MIN_SIGNAL_PCT:
        drivers.append(_driver(
            "transaction_decline",
            "انخفاض عدد المعاملات",
            "Transactions declined",
            current=float(current["transactions"]),
            previous=float(previous["transactions"]),
            unit="transactions",
            summary_ar=f"انخفض عدد المعاملات من {previous['transactions']} إلى {current['transactions']}.",
            summary_en=f"Transactions fell from {previous['transactions']} to {current['transactions']}.",
            raw_score=abs(transaction_change),
        ))

    basket_change = _pct_change(current["avg_basket"], previous["avg_basket"])
    if previous["avg_basket"] > 0 and basket_change <= -_MIN_SIGNAL_PCT:
        drivers.append(_driver(
            "basket_decline",
            "انخفاض متوسط قيمة المعاملة",
            "Average basket value declined",
            current=current["avg_basket"],
            previous=previous["avg_basket"],
            unit="currency",
            summary_ar=f"انخفض متوسط قيمة المعاملة من {previous['avg_basket']:.2f} إلى {current['avg_basket']:.2f}.",
            summary_en=f"Average transaction value fell from {previous['avg_basket']:.2f} to {current['avg_basket']:.2f}.",
            raw_score=abs(basket_change) * 1.05,
        ))

    if channel_drop and channel_drop["change_pct"] <= -_MIN_SIGNAL_PCT:
        name = channel_drop["name"]
        drivers.append(_driver(
            "channel_decline",
            f"تراجع قناة {name}",
            f"{name} channel declined",
            current=channel_drop["current"],
            previous=channel_drop["previous"],
            unit="currency",
            summary_ar=f"سجلت قناة {name} أكبر تراجع بالقيمة: {channel_drop['previous']:.2f} → {channel_drop['current']:.2f}.",
            summary_en=f"{name} had the largest channel-value drop: {channel_drop['previous']:.2f} → {channel_drop['current']:.2f}.",
            raw_score=min(100.0, abs(channel_drop["change_pct"])) * 0.92,
            entity=name,
        ))

    if product_drop and product_drop["change_pct"] <= -_MIN_SIGNAL_PCT:
        name = product_drop["name"]
        drivers.append(_driver(
            "product_decline",
            f"تراجع المنتج: {name}",
            f"Product declined: {name}",
            current=product_drop["current"],
            previous=product_drop["previous"],
            unit="currency",
            summary_ar=f"هذا المنتج سجّل أكبر انخفاض بالقيمة: {product_drop['previous']:.2f} → {product_drop['current']:.2f}.",
            summary_en=f"This product had the largest value decline: {product_drop['previous']:.2f} → {product_drop['current']:.2f}.",
            raw_score=min(100.0, abs(product_drop["change_pct"])) * 0.88,
            entity=name,
        ))

    drivers.sort(key=lambda item: item["raw_score"], reverse=True)
    drivers = drivers[:4]
    total_score = sum(item["raw_score"] for item in drivers)
    if total_score > 0:
        for item in drivers:
            item["strength_pct"] = round((item["raw_score"] / total_score) * 100.0, 1)
            item.pop("raw_score", None)
    else:
        for item in drivers:
            item.pop("raw_score", None)

    sales_change = _pct_change(current["sales"], previous["sales"])
    primary = drivers[0] if drivers else None
    action_map = {
        "quantity_decline": (
            "راجع توفر المخزون والطلب على المنتجات الأعلى مبيعًا، ثم افحص ما إذا كان التراجع ناتجًا عن نفاد أو ضعف الطلب.",
            "Review inventory availability and demand for top-selling products, then check whether the drop is driven by stock-outs or weaker demand.",
        ),
        "active_customer_decline": (
            "راجع العملاء الذين توقفوا عن الشراء خلال آخر 7 أيام وابدأ بإجراءات استعادة العملاء ذوي القيمة الأعلى.",
            "Review customers who stopped purchasing in the latest 7 days and prioritize reactivation of high-value customers.",
        ),
        "transaction_decline": (
            "راجع حركة الزيارات والتحويل والطلبات حسب القناة لمعرفة سبب انخفاض عدد المعاملات.",
            "Review visits, conversion and orders by channel to identify why transaction count declined.",
        ),
        "basket_decline": (
            "راجع مزيج المنتجات والأسعار والخصومات واقترح حزمًا أو عروضًا ترفع متوسط قيمة المعاملة.",
            "Review product mix, pricing and discounts, then consider bundles or offers that raise average transaction value.",
        ),
        "channel_decline": (
            "افحص القناة المتراجعة أولًا: الطلبات، التحويل، التوفر، والأسعار مقارنة بالفترة السابقة.",
            "Inspect the declining channel first: orders, conversion, availability and pricing versus the previous period.",
        ),
        "product_decline": (
            "ابدأ بالمنتج الأكثر تراجعًا وافحص توفره وسعره وطلبه ومبيعاته حسب القناة قبل اتخاذ إجراء.",
            "Start with the most-declining product and review its availability, price, demand and channel performance before acting.",
        ),
    }
    action_ar, action_en = action_map.get(
        primary["code"] if primary else "",
        (
            "استمر في المراقبة وراجع المنتجات والعملاء والقنوات إذا استمر الاتجاه الهابط.",
            "Continue monitoring and review products, customers and channels if the downward trend persists.",
        ),
    )
    return {
        "available": bool(drivers),
        "method": "recent_7d_vs_previous_7d_driver_ranking_v1",
        "causal_claim": False,
        "current_start": current_start.isoformat(),
        "current_end": current_end.isoformat(),
        "previous_start": previous_start.isoformat(),
        "previous_end": previous_end.isoformat(),
        "current_sales": current["sales"],
        "previous_sales": previous["sales"],
        "sales_change_pct": sales_change,
        "drivers": drivers,
        "primary_driver_code": primary["code"] if primary else None,
        "primary_driver_ar": primary["title_ar"] if primary else "لا توجد إشارة سلبية واحدة مهيمنة في الفترة الحالية.",
        "primary_driver_en": primary["title_en"] if primary else "No single negative signal dominates the current period.",
        "recommended_action_ar": action_ar,
        "recommended_action_en": action_en,
        "note_ar": "هذه إشارات تفسيرية مرتبطة بالتراجع في البيانات وليست إثباتًا سببيًا قاطعًا.",
        "note_en": "These are data-supported decline signals, not proof of causation.",
    }
