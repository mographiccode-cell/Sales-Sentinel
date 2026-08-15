from __future__ import annotations

import numpy as np
import pandas as pd

import train_merchant_sama_hybrid_v4 as v4

v4.VERSION = "SALES-SENTINEL-MERCHANT-SAMA-HYBRID-4.0.1"


def make_features(panel: pd.DataFrame):
    d = panel.copy().sort_values(["category", "date"]).reset_index(drop=True)
    g = d.groupby("category", sort=False)
    d["sales_target_base"] = d["net_sales_sar"].clip(lower=0)
    d["baseline28_daily"] = g.sales_target_base.transform(lambda s: s.rolling(28, min_periods=28).mean())
    d["future7_sales"] = sum(g.sales_target_base.shift(-h) for h in range(1, 8))
    d["future_ratio"] = d["future7_sales"] / (7.0 * d["baseline28_daily"].replace(0, np.nan))
    d["target"] = (d["future_ratio"] < (1.0 - v4.DECLINE)).astype(int)

    X = pd.DataFrame(index=d.index)
    dynamic = [
        "net_sales_sar", "gross_sales_sar", "invoice_count", "observed_customer_count",
        "unique_products", "units", "avg_invoice_value_sar", "avg_unit_value_sar",
        "return_rate_value", "electronic_share",
    ]
    for col in dynamic:
        s = d[col].astype(float)
        prefix = col.replace("_sar", "")
        for w in (7, 14, 28, 56):
            mean = g[col].transform(lambda z, w=w: z.rolling(w, min_periods=w).mean())
            ratio = s / mean.replace(0, np.nan)
            X[f"{prefix}_ratio_mean_{w}"] = ratio
        for lag in (1, 7, 14, 28):
            prev = g[col].shift(lag)
            change = (s - prev) / prev.abs().replace(0, np.nan)
            X[f"{prefix}_change_{lag}"] = change

    merchant = d.groupby("date", as_index=False).agg(
        merchant_net_sales=("net_sales_sar", "sum"),
        merchant_gross_sales=("gross_sales_sar", "sum"),
        merchant_invoices=("invoice_count", "sum"),
        merchant_customers=("observed_customer_count", "sum"),
        merchant_units=("units", "sum"),
        merchant_returns=("return_value_sar", "sum"),
    ).sort_values("date")
    for col in ["merchant_net_sales", "merchant_invoices", "merchant_customers", "merchant_units"]:
        for w in (7, 28, 56):
            mean = merchant[col].rolling(w, min_periods=w).mean()
            merchant[f"{col}_ratio_mean_{w}"] = merchant[col] / mean.replace(0, np.nan)
        for lag in (7, 28):
            prev = merchant[col].shift(lag)
            merchant[f"{col}_change_{lag}"] = (merchant[col] - prev) / prev.abs().replace(0, np.nan)
    merchant["merchant_return_rate"] = merchant.merchant_returns / merchant.merchant_gross_sales.clip(lower=1e-9)
    mcols = [c for c in merchant.columns if c != "date"]
    dm = d[["date"]].merge(merchant, on="date", how="left", validate="many_to_one")
    X = pd.concat([X, dm[mcols]], axis=1)

    total_by_day = d.groupby("date").net_sales_sar.transform("sum")
    d["category_share"] = d.net_sales_sar / total_by_day.replace(0, np.nan)
    gs = d.groupby("category", sort=False)
    share_mean = gs.category_share.transform(lambda s: s.rolling(28, min_periods=28).mean())
    X["category_share_ratio_28"] = d.category_share / share_mean.replace(0, np.nan)
    prev_share = gs.category_share.shift(7)
    X["category_share_change_7"] = (d.category_share - prev_share) / prev_share.abs().replace(0, np.nan)

    forecast_cols = [c for c in d.columns if c.startswith("predicted_")]
    for c in forecast_cols:
        X[f"sama_{c}"] = pd.to_numeric(d[c], errors="coerce")

    dates = pd.to_datetime(d.date)
    X["is_ramadan"] = v4.in_ranges(dates, v4.RAMADAN).astype(float)
    X["is_eid_fitr"] = v4.in_ranges(dates, v4.EID_FITR).astype(float)
    X["is_hajj"] = v4.in_ranges(dates, v4.HAJJ).astype(float)
    X["is_eid_adha"] = v4.in_ranges(dates, v4.EID_ADHA).astype(float)
    X["is_national_day_window"] = ((dates.dt.month == 9) & dates.dt.day.between(16, 30)).astype(float)
    X["is_founding_day_window"] = ((dates.dt.month == 2) & dates.dt.day.between(15, 29)).astype(float)
    X["salary_period"] = dates.dt.day.between(24, 31).astype(float)
    dow = dates.dt.dayofweek.astype(float)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    doy = dates.dt.dayofyear.astype(float)
    X["year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X = pd.concat([X, pd.get_dummies(d[["category"]], prefix="category", dtype=float)], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Missingness from a zero operational denominator is itself common merchant behavior.
    # Neutral imputation preserves the day rather than silently dropping sparse categories.
    for c in X.columns:
        if "_ratio_" in c or "index_52median" in c:
            X[c] = X[c].fillna(1.0)
        elif "_change_" in c or c.endswith("_change_7") or c.endswith("_change_28"):
            X[c] = X[c].fillna(0.0)
        elif c.startswith("sama_predicted_"):
            X[c] = X[c].fillna(0.0)
        else:
            X[c] = X[c].fillna(0.0)

    warmup = d.date >= (pd.Timestamp(d.date.min()) + pd.Timedelta(days=56))
    good = warmup & d.future_ratio.notna() & d.baseline28_daily.gt(0)
    meta = d.loc[good, ["date", "category", "sama_sector", "future_ratio", "future7_sales", "baseline28_daily", "target"]].reset_index(drop=True)
    return meta, X.loc[good].reset_index(drop=True)


v4.make_features = make_features

if __name__ == "__main__":
    v4.main()
