from __future__ import annotations

import numpy as np
import pandas as pd

import build_train_saudi_panel_v1_4 as base

VERSION = "SA-LOCALIZATION-1.4.1-PANEL-EXACT"
base.VERSION = VERSION


def exact_aggregate_panel():
    if not base.SOURCE.exists():
        raise FileNotFoundError(f"Missing rebuilt v1.3.1 full microdata: {base.SOURCE}")

    used = [
        "TrainingSafeDate", "Region", "SAMASector", "SAMACalibratedNetSalesSAR",
        "BaseNetSalesSAR", "EligibleForSalesTraining", "IsAdministrativeLine",
        "SaudiInvoiceNo", "ObservedSaudiCustomerID", "CustomerIDSource", "StockCode",
        "OriginalQuantity", "PaymentType", "SAMAWeeklyMarketIndex", "SAMACalibrationFactor",
    ]
    d = pd.read_csv(base.SOURCE, compression="gzip", usecols=used)
    input_rows = len(d)
    d["TrainingSafeDate"] = pd.to_datetime(d["TrainingSafeDate"]).dt.normalize()
    eligible = base.truth(d["EligibleForSalesTraining"])
    admin = base.truth(d["IsAdministrativeLine"])
    administrative_rows = int(admin.sum())
    d = d.loc[eligible & ~admin].copy()
    eligible_rows = len(d)
    calibrated_nulls = int(d["SAMACalibratedNetSalesSAR"].isna().sum())

    observed = d["ObservedSaudiCustomerID"].notna() & d["CustomerIDSource"].astype(str).eq("ObservedSourceCustomerID")
    observed_customer_rows = int(observed.sum())
    fallback_customer_rows = int((~observed).sum())
    d["ObservedCustomerForModel"] = d["ObservedSaudiCustomerID"].where(observed, pd.NA)
    d["PositiveSalesSAR"] = d["SAMACalibratedNetSalesSAR"].astype(float).clip(lower=0)
    d["ReturnValueSAR"] = -d["SAMACalibratedNetSalesSAR"].astype(float).clip(upper=0)

    keys = ["TrainingSafeDate", "Region", "SAMASector"]
    g = d.groupby(keys, observed=True, sort=False)
    panel = g.agg(
        calibrated_sales_sar=("SAMACalibratedNetSalesSAR", "sum"),
        base_sales_sar=("BaseNetSalesSAR", "sum"),
        gross_sales_sar=("PositiveSalesSAR", "sum"),
        return_value_sar=("ReturnValueSAR", "sum"),
        transaction_rows=("SaudiInvoiceNo", "size"),
        invoice_count=("SaudiInvoiceNo", "nunique"),
        observed_customer_count=("ObservedCustomerForModel", "nunique"),
        product_count=("StockCode", "nunique"),
        units=("OriginalQuantity", lambda s: float(np.abs(pd.to_numeric(s, errors="coerce")).sum())),
        sama_market_index=("SAMAWeeklyMarketIndex", "median"),
        sama_calibration_factor=("SAMACalibrationFactor", "median"),
    ).reset_index()

    invoice_level = d[keys + ["SaudiInvoiceNo", "PaymentType"]].drop_duplicates(keys + ["SaudiInvoiceNo"])
    invoice_level["ElectronicInvoice"] = invoice_level["PaymentType"].astype(str).eq("Electronic")
    electronic = invoice_level.groupby(keys, observed=True)["ElectronicInvoice"].mean().rename("electronic_invoice_share").reset_index()
    panel = panel.merge(electronic, on=keys, how="left", validate="one_to_one")

    panel["average_invoice_value_sar"] = panel["calibrated_sales_sar"] / panel["invoice_count"].clip(lower=1)
    panel["return_rate_value"] = panel["return_value_sar"] / panel["gross_sales_sar"].clip(lower=1)
    panel["entity"] = panel["Region"].astype(str) + " | " + panel["SAMASector"].astype(str)

    # Do not manufacture absent entity-days as zero. Keep only sufficiently observed real localized series.
    entity_stats = panel.groupby("entity").agg(
        observed_days=("TrainingSafeDate", "nunique"),
        median_invoices=("invoice_count", "median"),
        total_invoices=("invoice_count", "sum"),
    )
    keep = entity_stats[(entity_stats["observed_days"] >= 400) & (entity_stats["median_invoices"] >= 1)].index
    before_sparse = len(panel)
    panel = panel[panel["entity"].isin(keep)].copy()
    panel = panel.sort_values(["entity", "TrainingSafeDate"]).reset_index(drop=True)

    duplicates = int(panel.duplicated(["TrainingSafeDate", "entity"]).sum())
    core = ["TrainingSafeDate", "Region", "SAMASector", "calibrated_sales_sar", "invoice_count", "observed_customer_count"]
    core_nulls = int(panel[core].isna().sum().sum())
    invalid_invoice_counts = int((panel["invoice_count"] <= 0).sum())
    invalid_transaction_counts = int((panel["transaction_rows"] <= 0).sum())
    customer_gt_invoice = int((panel["observed_customer_count"] > panel["invoice_count"]).sum())
    if duplicates or core_nulls or invalid_invoice_counts or invalid_transaction_counts:
        raise RuntimeError(
            f"Exact panel integrity failure dup={duplicates} null={core_nulls} "
            f"invoice={invalid_invoice_counts} transaction={invalid_transaction_counts}"
        )

    panel.to_csv(base.PANEL_CSV, index=False, compression={"method": "gzip", "compresslevel": 5})
    stats = {
        "aggregation_method": "single full-source exact groupby; no summation of chunk-level nunique values",
        "input_microdata_rows": input_rows,
        "eligible_nonadministrative_rows": eligible_rows,
        "administrative_rows_excluded": administrative_rows,
        "observed_customer_rows": observed_customer_rows,
        "fallback_or_missing_customer_rows_not_counted_as_customers": fallback_customer_rows,
        "calibrated_sales_nulls": calibrated_nulls,
        "panel_rows_before_sparse_filter": before_sparse,
        "panel_rows": len(panel),
        "entities": int(panel["entity"].nunique()),
        "regions": int(panel["Region"].nunique()),
        "sectors": int(panel["SAMASector"].nunique()),
        "date_start": str(panel["TrainingSafeDate"].min().date()),
        "date_end": str(panel["TrainingSafeDate"].max().date()),
        "duplicate_entity_dates": duplicates,
        "core_nulls": core_nulls,
        "invalid_invoice_count_rows": invalid_invoice_counts,
        "invalid_transaction_count_rows": invalid_transaction_counts,
        "customer_count_greater_than_invoice_count_rows": customer_gt_invoice,
        "min_entity_observed_days": int(panel.groupby("entity")["TrainingSafeDate"].nunique().min()),
        "median_entity_observed_days": float(panel.groupby("entity")["TrainingSafeDate"].nunique().median()),
        "source_boundary": "Regions and merchant microtransactions remain Saudi-localized synthetic microdata; SAMA sectors and weekly market calibration are official aggregate Saudi signals.",
    }
    return panel, stats


base.aggregate_panel = exact_aggregate_panel

if __name__ == "__main__":
    base.main()
