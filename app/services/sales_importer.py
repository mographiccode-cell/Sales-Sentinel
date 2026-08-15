from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from sqlalchemy import select, text

from app.models import Branch, Category, Product, Region

UCI_COLUMNS = {"InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"}
DAILY_COLUMNS = {"date", "net_sales"}
REDSEA_COLUMNS = {"TRX DATE", "TRX NUMBER", "SALES CHANNEL", "CUSTOMER NUMBER", "ITEM CODE", "QUANTITY", "Unit Price", "Net Amount", "TOTAL AMOUNT"}


def _decimal(value, default="0") -> Decimal:
    try:
        return Decimal(str(value if value not in (None, "") else default))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid numeric value: {value!r}") from exc


def _date(value: str, *, uci: bool = False):
    value = str(value or "").strip()
    slash_formats = (
        ("%m/%d/%Y %H:%M", "%m/%d/%Y", "%d/%m/%Y %H:%M", "%d/%m/%Y")
        if uci else
        ("%d/%m/%Y %H:%M", "%d/%m/%Y", "%m/%d/%Y %H:%M", "%m/%d/%Y")
    )
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", *slash_formats):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise ValueError(f"invalid date: {value!r}") from exc


def detect_mode(columns: set[str]) -> str:
    if UCI_COLUMNS.issubset(columns): return "uci"
    if REDSEA_COLUMNS.issubset(columns): return "redsea"
    if DAILY_COLUMNS.issubset(columns): return "daily"
    return "invalid"


def inspect_csv(path: Path) -> tuple[str, int, int, list[str]]:
    total = accepted = 0
    errors: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        mode = detect_mode(set(reader.fieldnames or []))
        if mode == "invalid":
            return mode, 0, 0, ["Required UCI, Redsea, or date/net_sales columns are missing"]
        for line, row in enumerate(reader, start=2):
            total += 1
            try:
                if mode == "uci":
                    _date(row["InvoiceDate"], uci=True); _decimal(row["Quantity"]); _decimal(row["UnitPrice"])
                    if not str(row["InvoiceNo"]).strip() or not str(row["StockCode"]).strip(): raise ValueError("missing invoice or stock code")
                elif mode == "redsea":
                    _date(row["TRX DATE"]); _decimal(row["QUANTITY"]); _decimal(row["Unit Price"]); _decimal(row["Net Amount"])
                    if not str(row["TRX NUMBER"]).strip() or not str(row["ITEM CODE"]).strip(): raise ValueError("missing transaction or item code")
                else:
                    _date(row["date"]); _decimal(row["net_sales"])
                accepted += 1
            except (ValueError, TypeError, KeyError) as exc:
                if len(errors) < 100: errors.append(f"row {line}: {exc}")
    return mode, total, accepted, errors


def _get_or_create_reference_data(db):
    region = db.scalar(select(Region).where(Region.code == "IMPORT"))
    if not region:
        region = Region(code="IMPORT", name_ar="بيانات مستوردة", name_en="Imported data"); db.add(region); db.flush()
    branch = db.scalar(select(Branch).where(Branch.code == "IMPORT-STORE"))
    if not branch:
        branch = Branch(code="IMPORT-STORE", name_ar="المتجر المستورد", name_en="Imported store", city_ar="غير محدد", city_en="Unspecified", region_id=region.id); db.add(branch); db.flush()
    category = db.scalar(select(Category).where(Category.code == "IMPORTED"))
    if not category:
        category = Category(code="IMPORTED", name_ar="منتجات مستوردة", name_en="Imported products"); db.add(category); db.flush()
    aggregate = db.scalar(select(Product).where(Product.sku == "DAILY-IMPORTED-AGGREGATE"))
    if not aggregate:
        aggregate = Product(sku="DAILY-IMPORTED-AGGREGATE", name_ar="إجمالي يومي مستورد", name_en="Imported daily aggregate", category_id=category.id, base_price=Decimal("1")); db.add(aggregate); db.flush()
    return branch, category, aggregate


def _row_hash(mode: str, row: dict) -> str:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{mode}:{payload}".encode()).hexdigest()


def _product(db, cache: dict[str, int], category_id: int, sku: str, description: str, base_price: Decimal) -> int:
    sku = (str(sku).strip() or "UNKNOWN")[:50]
    if sku in cache: return cache[sku]
    product = db.scalar(select(Product).where(Product.sku == sku))
    if not product:
        product = Product(sku=sku, name_ar=(description or sku)[:150], name_en=(description or sku)[:150], category_id=category_id, base_price=max(base_price, Decimal("0"))); db.add(product); db.flush()
    cache[sku] = product.id
    return product.id


_INSERT = text("""
INSERT OR IGNORE INTO sales (
 sale_date,transaction_number,transaction_type,branch_id,product_id,customer_segment_id,promotion_id,
 channel,family,subclass,franchise,quantity,unit_price,discount_amount,discount_percent,gross_sales,net_sales,
 vat_amount,total_amount,stock_quantity,inventory_available,is_promotion,seasonal_factor,is_demo,source_row_hash,
 source_import_id,created_at,customer_key
) VALUES (
 :sale_date,:transaction_number,:transaction_type,:branch_id,:product_id,NULL,NULL,:channel,:family,:subclass,:franchise,
 :quantity,:unit_price,:discount_amount,:discount_percent,:gross_sales,:net_sales,:vat_amount,:total_amount,NULL,0,0,1.0,0,
 :source_row_hash,:source_import_id,CURRENT_TIMESTAMP,:customer_key
)
""")


def ingest_csv(db, path: Path, import_job_id: int, mode: str, chunk_size: int = 2000) -> dict:
    branch, category, aggregate = _get_or_create_reference_data(db)
    product_cache: dict[str, int] = {}
    before = int(db.execute(text("SELECT COUNT(*) FROM sales")).scalar_one())
    params: list[dict] = []
    parsed = rejected = 0
    errors: list[str] = []

    def flush():
        nonlocal params
        if params:
            db.execute(_INSERT, params); params = []

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), start=2):
            try:
                if mode == "daily":
                    day = _date(row["date"]); net = _decimal(row["net_sales"]); gross = _decimal(row.get("gross_sales", net)); quantity = int(_decimal(row.get("quantity", "0")))
                    item = dict(sale_date=day, transaction_number=f"DAILY-{day:%Y%m%d}", transaction_type="DAILY_IMPORT", branch_id=branch.id, product_id=aggregate.id, customer_key=None, channel="Aggregate", family="Daily aggregate", subclass="Daily aggregate", franchise="Imported CSV", quantity=quantity, unit_price=Decimal("1"), discount_amount=Decimal("0"), discount_percent=0.0, gross_sales=gross, net_sales=net, vat_amount=Decimal("0"), total_amount=net)
                elif mode == "uci":
                    day = _date(row["InvoiceDate"], uci=True); qty = int(_decimal(row["Quantity"])); price = _decimal(row["UnitPrice"]); net = Decimal(qty) * price
                    sku = str(row["StockCode"]).strip(); pid = _product(db, product_cache, category.id, sku, str(row.get("Description") or sku).strip(), price)
                    invoice = str(row["InvoiceNo"]).strip()
                    item = dict(sale_date=day, transaction_number=invoice, transaction_type="RETURN" if qty < 0 or invoice.upper().startswith("C") else "INV", branch_id=branch.id, product_id=pid, customer_key=str(row.get("CustomerID") or "").strip() or None, channel="Online", family="Retail", subclass=None, franchise=str(row.get("Country") or "UCI")[:120], quantity=qty, unit_price=price, discount_amount=Decimal("0"), discount_percent=0.0, gross_sales=max(net, Decimal("0")), net_sales=net, vat_amount=Decimal("0"), total_amount=net)
                elif mode == "redsea":
                    day = _date(row["TRX DATE"]); qty = int(_decimal(row["QUANTITY"])); price = _decimal(row["Unit Price"]); net = _decimal(row["Net Amount"]); total = _decimal(row["TOTAL AMOUNT"]); vat = _decimal(row.get("Vat Amount", total-net)); discount = _decimal(row.get("Discount Amount", "0")); discount_pct = float(_decimal(row.get("Discount Amount(%)", "0")))
                    sku = str(row["ITEM CODE"]).strip(); pid = _product(db, product_cache, category.id, sku, str(row.get("ITEM DESC") or sku).strip(), price)
                    item = dict(sale_date=day, transaction_number=str(row["TRX NUMBER"]).strip(), transaction_type=str(row.get("Type") or "INV").strip().upper(), branch_id=branch.id, product_id=pid, customer_key=str(row.get("CUSTOMER NUMBER") or "").strip() or None, channel=str(row.get("SALES CHANNEL") or "Store")[:30], family=str(row.get("FAMILY") or "")[:80] or None, subclass=str(row.get("SUBCLASS") or "")[:120] or None, franchise=str(row.get("FRANCHISE") or "")[:120] or None, quantity=qty, unit_price=price, discount_amount=discount, discount_percent=max(0.0, min(100.0, discount_pct)), gross_sales=max(net, Decimal("0")), net_sales=net, vat_amount=vat, total_amount=total)
                else:
                    raise ValueError("unsupported import mode")
                item["source_row_hash"] = _row_hash(mode, row); item["source_import_id"] = import_job_id
                params.append(item); parsed += 1
                if len(params) >= chunk_size: flush()
            except Exception as exc:
                rejected += 1
                if len(errors) < 100: errors.append(f"row {line}: {exc}")
        flush()
    db.flush()
    after = int(db.execute(text("SELECT COUNT(*) FROM sales")).scalar_one())
    inserted = after - before
    return {"parsed_rows": parsed, "inserted_rows": inserted, "duplicate_rows": max(parsed-inserted, 0), "rejected_rows": rejected, "errors": errors, "mode": mode}
