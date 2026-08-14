from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "sama_pos" / "sama_pos_2020_2025_normalized.csv"
OUT = ROOT / "data" / "sama_pos" / "sama_sector_walkforward_forecasts_2023_2025.csv"
COMPACT = ROOT / "data" / "sama_pos" / "sama_sector_weekly_value_count_2020_2025.csv"
REPORT_DIR = ROOT / "reports" / "sama_sector_v1_7"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "sama_sector_forecaster_report_v1_7.json"


def make_model():
    return HistGradientBoostingRegressor(
        learning_rate=0.045,
        max_iter=220,
        max_leaf_nodes=18,
        min_samples_leaf=18,
        l2_regularization=2.5,
        random_state=SEED,
    )


def wape(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    return float(np.abs(y-p).sum()/max(np.abs(y).sum(), 1e-9))


def load_panel():
    raw = pd.read_csv(SOURCE, parse_dates=["week_start", "week_end"])
    indicator = raw["indicator"].astype(str).str.lower()
    value_mask = indicator.str.contains("value") & indicator.str.contains("transaction") & ~indicator.str.contains("change")
    count_mask = indicator.str.contains("number") & indicator.str.contains("transaction") & ~indicator.str.contains("change")
    national_sector = raw["city"].astype(str).str.strip().str.lower().eq("total") & ~raw["sector"].astype(str).str.strip().str.lower().eq("total")

    value = raw.loc[value_mask & national_sector, ["week_start", "week_end", "sector", "value"]].rename(columns={"value":"value_thousand_sar"})
    count = raw.loc[count_mask & national_sector, ["week_start", "sector", "value"]].rename(columns={"value":"transaction_count_thousand"})
    d = value.merge(count, on=["week_start", "sector"], how="inner", validate="one_to_one")
    d["value_thousand_sar"] = pd.to_numeric(d["value_thousand_sar"], errors="coerce")
    d["transaction_count_thousand"] = pd.to_numeric(d["transaction_count_thousand"], errors="coerce")
    d = d.dropna().query("value_thousand_sar > 0 and transaction_count_thousand > 0").sort_values(["sector","week_start"]).reset_index(drop=True)
    counts = d.groupby("sector").size()
    keep = counts[counts >= 250].index
    d = d[d.sector.isin(keep)].copy()
    COMPACT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(COMPACT, index=False)
    return d


def features(d):
    x = d.copy().sort_values(["sector","week_start"]).reset_index(drop=True)
    g = x.groupby("sector", sort=False, group_keys=False)
    for col, prefix in [("value_thousand_sar","value"),("transaction_count_thousand","count")]:
        log = np.log1p(x[col].astype(float))
        x[f"log_{prefix}_t0"] = log
        for lag in (1,2,3,4,8,13,26,52):
            x[f"log_{prefix}_lag_{lag}"] = g[col].shift(lag).pipe(np.log1p)
        for w in (4,8,13,26,52):
            x[f"log_{prefix}_mean_{w}"] = g[col].transform(lambda s,w=w: np.log1p(s).rolling(w,min_periods=w).mean())
            x[f"log_{prefix}_std_{w}"] = g[col].transform(lambda s,w=w: np.log1p(s).rolling(w,min_periods=w).std())
        x[f"{prefix}_change_1"] = g[col].pct_change(1)
        x[f"{prefix}_change_4"] = g[col].pct_change(4)
        x[f"{prefix}_change_13"] = g[col].pct_change(13)

    # National context known at origin week.
    raw = pd.read_csv(SOURCE, parse_dates=["week_start"])
    ind = raw.indicator.astype(str).str.lower()
    national = raw.city.astype(str).str.strip().str.lower().eq("total") & raw.sector.astype(str).str.strip().str.lower().eq("total")
    vm = ind.str.contains("value") & ind.str.contains("transaction") & ~ind.str.contains("change")
    nm = ind.str.contains("number") & ind.str.contains("transaction") & ~ind.str.contains("change")
    nv = raw.loc[national & vm, ["week_start","value"]].rename(columns={"value":"national_value"}).drop_duplicates("week_start")
    nc = raw.loc[national & nm, ["week_start","value"]].rename(columns={"value":"national_count"}).drop_duplicates("week_start")
    nat = nv.merge(nc,on="week_start",how="inner").sort_values("week_start")
    nat["log_national_value"] = np.log1p(nat.national_value.astype(float))
    nat["log_national_count"] = np.log1p(nat.national_count.astype(float))
    for c in ["log_national_value","log_national_count"]:
        for lag in (1,4,13,52): nat[f"{c}_lag_{lag}"] = nat[c].shift(lag)
    x = x.merge(nat.drop(columns=["national_value","national_count"]), on="week_start", how="left", validate="many_to_one")

    week = x.week_start.dt.isocalendar().week.astype(float)
    x["week_sin"] = np.sin(2*np.pi*week/52.18)
    x["week_cos"] = np.cos(2*np.pi*week/52.18)
    cats = pd.get_dummies(x[["sector"]], prefix="sector", dtype=float)
    x = pd.concat([x,cats],axis=1)

    # Direct horizon targets in log space.
    gx = x.groupby("sector",sort=False)
    x["value_h1"] = gx.value_thousand_sar.shift(-1).pipe(np.log1p)
    x["value_h2"] = gx.value_thousand_sar.shift(-2).pipe(np.log1p)
    x["count_h1"] = gx.transaction_count_thousand.shift(-1).pipe(np.log1p)
    x["count_h2"] = gx.transaction_count_thousand.shift(-2).pipe(np.log1p)
    targets = ["value_h1","value_h2","count_h1","count_h2"]
    exclude = {"week_start","week_end","sector","value_thousand_sar","transaction_count_thousand",*targets}
    fcols = [c for c in x.columns if c not in exclude]
    return x.replace([np.inf,-np.inf],np.nan), fcols, targets


def main():
    d = load_panel()
    x, fcols, targets = features(d)
    complete = x[fcols].notna().all(axis=1)
    origins = sorted(x.loc[x.week_start >= pd.Timestamp("2022-12-25"), "week_start"].unique())
    rows=[]
    cache={}
    for oi, origin in enumerate(origins):
        origin=pd.Timestamp(origin)
        batch_anchor = origins[(oi//4)*4]
        batch_anchor = pd.Timestamp(batch_anchor)
        current = x[(x.week_start==origin)&complete].copy()
        if current.empty: continue
        preds={}
        for target in targets:
            horizon=2 if target.endswith("h2") else 1
            key=(batch_anchor,target)
            model=cache.get(key)
            if model is None:
                known_by_anchor = x.week_start + pd.to_timedelta(7*horizon,unit="D") <= batch_anchor
                tr=x[complete & x[target].notna() & known_by_anchor & (x.week_start < batch_anchor)]
                if len(tr)<1000: continue
                model=make_model().fit(tr[fcols],tr[target]); cache[key]=model
            preds[target]=np.expm1(model.predict(current[fcols]))
        if len(preds)!=4: continue
        for j, (_,r) in enumerate(current.iterrows()):
            sector=r.sector
            hist=d[(d.sector==sector)&(d.week_start<=origin)].tail(52)
            value_scale=float(hist.value_thousand_sar.median()); count_scale=float(hist.transaction_count_thousand.median())
            last_value=float(r.value_thousand_sar); last_count=float(r.transaction_count_thousand)
            def actual(h,col):
                z=d[(d.sector==sector)&(d.week_start==origin+pd.Timedelta(days=7*h))]
                return float(z.iloc[0][col]) if len(z) else np.nan
            rows.append({
                "origin_week_start":origin,"sector":sector,
                "forecast_h1_week_start":origin+pd.Timedelta(days=7),"forecast_h2_week_start":origin+pd.Timedelta(days=14),
                "predicted_value_h1":float(preds['value_h1'][j]),"predicted_value_h2":float(preds['value_h2'][j]),
                "predicted_count_h1":float(preds['count_h1'][j]),"predicted_count_h2":float(preds['count_h2'][j]),
                "actual_value_h1":actual(1,'value_thousand_sar'),"actual_value_h2":actual(2,'value_thousand_sar'),
                "actual_count_h1":actual(1,'transaction_count_thousand'),"actual_count_h2":actual(2,'transaction_count_thousand'),
                "predicted_value_h1_index_52median":float(preds['value_h1'][j]/value_scale),
                "predicted_value_h2_index_52median":float(preds['value_h2'][j]/value_scale),
                "predicted_count_h1_index_52median":float(preds['count_h1'][j]/count_scale),
                "predicted_count_h2_index_52median":float(preds['count_h2'][j]/count_scale),
                "predicted_value_h1_change_vs_last":float(preds['value_h1'][j]/last_value-1),
                "predicted_value_h2_change_vs_last":float(preds['value_h2'][j]/last_value-1),
                "predicted_count_h1_change_vs_last":float(preds['count_h1'][j]/last_count-1),
                "predicted_count_h2_change_vs_last":float(preds['count_h2'][j]/last_count-1),
            })
    out=pd.DataFrame(rows).sort_values(["origin_week_start","sector"])
    out.to_csv(OUT,index=False)
    metrics={}
    e=out[out.origin_week_start>=pd.Timestamp("2023-01-01")]
    for name,a,p in [
        ('value_h1','actual_value_h1','predicted_value_h1'),('value_h2','actual_value_h2','predicted_value_h2'),
        ('count_h1','actual_count_h1','predicted_count_h1'),('count_h2','actual_count_h2','predicted_count_h2')]:
        q=e[[a,p]].dropna(); metrics[name]={"rows":int(len(q)),"MAE":float(mean_absolute_error(q[a],q[p])),"WAPE":wape(q[a],q[p]),"correlation":float(q[a].corr(q[p]))}
    report={
        "version":"SAMA-SECTOR-FORECASTER-1.7",
        "official_panel_rows":int(len(d)),"sectors":int(d.sector.nunique()),
        "source_start":str(d.week_start.min().date()),"source_end":str(d.week_start.max().date()),
        "walkforward_rows":int(len(out)),"metrics":metrics,
        "leakage_controls":{"four_week_refit_batches_use_only_data_known_at_batch_anchor":True,"future_actual_SAMA_not_features":True,"h1_h2_targets_known_by_training_anchor_only":True}
    }
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
