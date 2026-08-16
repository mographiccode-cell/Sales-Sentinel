from __future__ import annotations

from sqlalchemy import func, select

from app.models import Sale


def preferred_sales_condition(db):
    """Return the canonical display scope for sales-facing screens.

    Seeded aggregates exist to make a fresh academic/demo installation usable.
    Once at least one real import has inserted sales rows, dashboards and sales
    views must stop mixing those seed records with user data.
    """
    imported = int(
        db.scalar(
            select(func.count(Sale.id)).where(Sale.source_import_id.is_not(None))
        )
        or 0
    )
    if imported > 0:
        return Sale.source_import_id.is_not(None), "imported", imported
    return Sale.source_import_id.is_(None), "seed", 0
