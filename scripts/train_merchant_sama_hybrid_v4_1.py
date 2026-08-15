from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_merchant_sama_hybrid_v4 as v4

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-MERCHANT-SAMA-HYBRID-4.1-WEEKLY"
FULL_GZ = ROOT / "artifacts" / "saudi_v1_3" / "saudi_localized_transactions_v1_3_sama.csv.gz"
SAMA_FORECAST = ROOT / "data" / "sama_pos" / "sama_sector_walkforward_forecasts_2023_2025.csv"
OUT = ROOT / "reports" / "merchant_sama_hybrid_v4_1"
MOD = ROOT / "models" / "merchant_sama_hybrid_v4_1"
DATA = ROOT / "data" / "merchant_v4_1"
for p in (OUT, MOD, DATA): p.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
MODEL = MOD / "merchant_sama_hybrid_v4_1.joblib"
PANEL_CSV = DATA / "merchant_category_week_panel_v4_1.csv"

TRAIN_END = pd.Timestamp("2023-11-26")
VAL_START = pd.Timestamp("2023-12-10")
VAL_END = pd.Timestamp("2024-02-25")
TEST_START = pd.Timestamp("2024-03-10")
DECLINE = 0.20
SEED = 42


def week_start(series):
    d = pd.to_datetime(series)
    return d - pd.to_timedelta((d.dt.dayofweek + 1) % 7, unit="D")


def aggregate_weekly(path: Path):
    numeric = defaultdict(lambda: defaultdict(float))
    invoices = defaultdict(set); electronic = defaultdict(set); customers = defaultdict(set); products = defaultdict(set)
    observed_days = defaultdict(set)
    source_rows = eligible_rows = 0
    usecols = ["TrainingSafeDate","ProductCategoryCOICOP","SAMASector","SAMACalibratedNetSalesSAR","OriginalQuantity","SaudiInvoiceNo","PaymentType","StockCode","ObservedSaudiCustomerID","EligibleForSalesTraining"]
    for chunk in pd.read_csv(path, compression="gzip", usecols=usecols, chunksize=120_000):
        source_rows += len(chunk)
        ok = chunk.EligibleForSalesTraining.astype(str).str.lower().isin(["true","1"])
        chunk = chunk.loc[ok].copy(); eligible_rows += len(chunk)
        chunk["date"] = pd.to_datetime(chunk.TrainingSafeDate).dt.normalize()
        chunk["week_start"] = week_start(chunk.date)
        chunk["net"] = pd.to_numeric(chunk.SAMACalibratedNetSalesSAR, errors="coerce").fillna(0.0)
        chunk["qty"] = pd.to_numeric(chunk.OriginalQuantity, errors="coerce").fillna(0.0).abs()
        for (ws,cat,sector), z in chunk.groupby(["week_start","ProductCategoryCOICOP","SAMASector"], sort=False):
            k=(pd.Timestamp(ws),str(cat),str(sector)); n=numeric[k]; net=z.net.astype(float)
            n["net_sales_sar"] += float(net.sum()); n["gross_sales_sar"] += float(net.clip(lower=0).sum()); n["return_value_sar"] += float((-net.clip(upper=0)).sum()); n["units"] += float(z.qty.sum()); n["line_rows"] += len(z)
            invoices[k].update(z.SaudiInvoiceNo.dropna().astype(str)); electronic[k].update(z.loc[z.PaymentType.eq("Electronic"),"SaudiInvoiceNo"].dropna().astype(str)); customers[k].update(z.ObservedSaudiCustomerID.dropna().astype(str)); products[k].update(z.StockCode.dropna().astype(str)); observed_days[k].update(z.date.tolist())
    if source_rows != 1_049_042: raise RuntimeError(f"Expected 1,049,042 rows, found {source_rows}")
    rows=[]
    for k in sorted(numeric):
        ws,cat,sector=k; n=numeric[k]; inv=len(invoices[k]); gross=n["gross_sales_sar"]; units=n["units"]
        rows.append({"week_start":ws,"category":cat,"sama_sector":sector,"net_sales_sar":n["net_sales_sar"],"gross_sales_sar":gross,"return_value_sar":n["return_value_sar"],"units":units,"line_rows":int(n["line_rows"]),"invoice_count":inv,"electronic_invoice_count":len(electronic[k]),"observed_customer_count":len(customers[k]),"unique_products":len(products[k]),"avg_invoice_value_sar":n["net_sales_sar"]/max(inv,1),"avg_unit_value_sar":gross/max(units,1.0),"return_rate_value":n["return_value_sar"]/max(gross,1e-9),"electronic_share":len(electronic[k])/max(inv,1),"observed_days":len(observed_days[k])})
    d=pd.DataFrame(rows)
    return d, source_rows, eligible_rows


def complete_full_weeks(raw):
    # Keep only calendar weeks fully covered by the localized merchant date range.
    global_days = pd.date_range(raw.week_start.min(), raw.week_start.max()+pd.Timedelta(days=6), freq="D")
    max_date = pd.Timestamp("2024-08-26")
    full_week_starts = [x for x in sorted(raw.week_start.unique()) if pd.Timestamp(x)+pd.Timedelta(days=6) <= max_date]
    mapping=raw[["category","sama_sector"]].drop_duplicates()
    if mapping.groupby("category").sama_sector.nunique().max()!=1: raise RuntimeError("Unstable category/sector mapping")
    grid=pd.MultiIndex.from_product([full_week_starts,mapping.category.tolist()],names=["week_start","category"]).to_frame(index=False)
    d=grid.merge(raw,on=["week_start","category"],how="left").merge(mapping,on="category",how="left",suffixes=("","_map"))
    d["sama_sector"]=d.sama_sector.fillna(d.sama_sector_map); d=d.drop(columns=["sama_sector_map"])
    nums=["net_sales_sar","gross_sales_sar","return_value_sar","units","line_rows","invoice_count","electronic_invoice_count","observed_customer_count","unique_products","avg_invoice_value_sar","avg_unit_value_sar","return_rate_value","electronic_share","observed_days"]
    d[nums]=d[nums].fillna(0.0)
    return d.sort_values(["category","week_start"]).reset_index(drop=True)


def add_sama(d):
    f=pd.read_csv(SAMA_FORECAST,parse_dates=["origin_week_start"])
    cols=["origin_week_start","sector","predicted_value_h1_index_52median","predicted_value_h2_index_52median","predicted_count_h1_index_52median","predicted_count_h2_index_52median","predicted_value_h1_change_vs_last","predicted_value_h2_change_vs_last","predicted_count_h1_change_vs_last","predicted_count_h2_change_vs_last"]
    f=f[cols].rename(columns={"sector":"sama_sector"})
    q=d.copy(); q["sama_forecast_origin"]=q.week_start-pd.Timedelta(days=7)
    q=q.merge(f,left_on=["sama_forecast_origin","sama_sector"],right_on=["origin_week_start","sama_sector"],how="left",validate="many_to_one").drop(columns=["origin_week_start"])
    q["sama_expected_value_change_h2_vs_h1"]=q.predicted_value_h2_index_52median/q.predicted_value_h1_index_52median.replace(0,np.nan)-1
    q["sama_expected_count_change_h2_vs_h1"]=q.predicted_count_h2_index_52median/q.predicted_count_h1_index_52median.replace(0,np.nan)-1
    return q


def make_dataset(panel):
    d=panel.copy().sort_values(["category","week_start"]).reset_index(drop=True); g=d.groupby("category",sort=False)
    d["sales_pos"]=d.net_sales_sar.clip(lower=0)
    d["baseline4"]=g.sales_pos.transform(lambda s:s.rolling(4,min_periods=4).mean())
    d["next_week_sales"]=g.sales_pos.shift(-1)
    d["future_ratio"]=d.next_week_sales/d.baseline4.replace(0,np.nan)
    d["target"]=(d.future_ratio<1-DECLINE).astype(int)
    X=pd.DataFrame(index=d.index)
    dynamic=["net_sales_sar","gross_sales_sar","invoice_count","observed_customer_count","unique_products","units","avg_invoice_value_sar","avg_unit_value_sar","return_rate_value","electronic_share"]
    for col in dynamic:
        s=d[col].astype(float); pre=col.replace("_sar","")
        for w in (4,8,13):
            m=g[col].transform(lambda z,w=w:z.rolling(w,min_periods=w).mean()); X[f"{pre}_ratio_mean_{w}"]=s/m.replace(0,np.nan)
        for lag in (1,4,13):
            prev=g[col].shift(lag); X[f"{pre}_change_{lag}"]=(s-prev)/prev.abs().replace(0,np.nan)
    merchant=d.groupby("week_start",as_index=False).agg(merchant_sales=("net_sales_sar","sum"),merchant_invoices=("invoice_count","sum"),merchant_customers=("observed_customer_count","sum"),merchant_units=("units","sum"),merchant_returns=("return_value_sar","sum"),merchant_gross=("gross_sales_sar","sum"))
    for col in ["merchant_sales","merchant_invoices","merchant_customers","merchant_units"]:
        for w in (4,13): merchant[f"{col}_ratio_mean_{w}"]=merchant[col]/merchant[col].rolling(w,min_periods=w).mean().replace(0,np.nan)
        for lag in (1,4,13):
            prev=merchant[col].shift(lag); merchant[f"{col}_change_{lag}"]=(merchant[col]-prev)/prev.abs().replace(0,np.nan)
    merchant["merchant_return_rate"]=merchant.merchant_returns/merchant.merchant_gross.clip(lower=1e-9)
    mc=[c for c in merchant.columns if c!="week_start"]; dm=d[["week_start"]].merge(merchant,on="week_start",how="left",validate="many_to_one"); X=pd.concat([X,dm[mc]],axis=1)
    total=d.groupby("week_start").net_sales_sar.transform("sum"); d["category_share"]=d.net_sales_sar/total.replace(0,np.nan); gs=d.groupby("category",sort=False)
    X["category_share_ratio_4"]=d.category_share/gs.category_share.transform(lambda s:s.rolling(4,min_periods=4).mean()).replace(0,np.nan)
    X["category_share_change_1"]=(d.category_share-gs.category_share.shift(1))/gs.category_share.shift(1).abs().replace(0,np.nan)
    for c in [x for x in d.columns if x.startswith("predicted_") or x.startswith("sama_expected_")]: X[f"sama_{c}"]=pd.to_numeric(d[c],errors="coerce")
    ws=pd.to_datetime(d.week_start); next_ws=ws+pd.Timedelta(days=7)
    for name,ranges in [("ramadan",v4.RAMADAN),("eid_fitr",v4.EID_FITR),("hajj",v4.HAJJ),("eid_adha",v4.EID_ADHA)]:
        X[f"current_{name}"]=v4.in_ranges(ws,ranges).astype(float); X[f"next_{name}"]=v4.in_ranges(next_ws,ranges).astype(float)
    for pref,z in [("current",ws),("next",next_ws)]:
        X[f"{pref}_salary_period"]=z.dt.day.between(24,31).astype(float); X[f"{pref}_national_day"]=((z.dt.month==9)&z.dt.day.between(16,30)).astype(float); X[f"{pref}_founding_day"]=((z.dt.month==2)&z.dt.day.between(15,29)).astype(float)
        week=z.dt.isocalendar().week.astype(float); X[f"{pref}_week_sin"]=np.sin(2*np.pi*week/52.18); X[f"{pref}_week_cos"]=np.cos(2*np.pi*week/52.18)
    X=pd.concat([X,pd.get_dummies(d[["category"]],prefix="category",dtype=float)],axis=1).replace([np.inf,-np.inf],np.nan)
    # Ratios/change missing during warm-up or a zero denominator are neutral; explicit zero-sales levels remain in upstream ratios.
    for c in X.columns:
        if "ratio" in c or "index_52median" in c: X[c]=X[c].fillna(1.0)
        else: X[c]=X[c].fillna(0.0)
    warm=d.groupby("category").cumcount()>=13; good=warm & d.future_ratio.notna() & d.baseline4.gt(0)
    meta=d.loc[good,["week_start","category","sama_sector","future_ratio","next_week_sales","baseline4","target"]].reset_index(drop=True)
    return meta,X.loc[good].reset_index(drop=True)


def factories():
    return {
        "logistic":make_pipeline(StandardScaler(),LogisticRegression(C=.12,class_weight="balanced",max_iter=5000,random_state=SEED)),
        "extra_trees":ExtraTreesClassifier(n_estimators=800,max_depth=6,min_samples_leaf=5,max_features=.65,class_weight="balanced",random_state=SEED,n_jobs=-1),
        "hist_gb":HistGradientBoostingClassifier(max_iter=280,learning_rate=.025,max_leaf_nodes=10,min_samples_leaf=16,l2_regularization=10.,random_state=SEED),
    }

def fit_cls(m,X,y):
    if isinstance(m,HistGradientBoostingClassifier):
        pos=max(int(y.sum()),1); neg=max(len(y)-pos,1); return m.fit(X,y,sample_weight=np.where(np.asarray(y)==1,neg/pos,1.))
    return m.fit(X,y)

def met(y,s,t):
    y=np.asarray(y,int); p=np.asarray(s)>=t
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),"precision":float(precision_score(y,p,zero_division=0)),"recall":float(recall_score(y,p,zero_division=0)),"f1":float(f1_score(y,p,zero_division=0)),"roc_auc":float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,"alert_rate":float(p.mean()),"green_npv":float(((y==0)&~p).sum()/max((~p).sum(),1)),"tp":int(((y==1)&p).sum()),"fp":int(((y==0)&p).sum()),"fn":int(((y==1)&~p).sum()),"tn":int(((y==0)&~p).sum())}
def choose(y,s):
    rows=[]
    for t in np.unique(np.r_[np.linspace(.05,.95,181),np.quantile(s,np.linspace(.02,.98,97))]):
        m=met(y,s,float(t)); rows.append((float(t),m))
    feasible=[z for z in rows if z[1]["recall"]>=.65 and z[1]["alert_rate"]<=.50]
    pool=feasible if feasible else rows; pool.sort(key=lambda z:(z[1]["balanced_accuracy"],z[1]["f1"],z[1]["recall"],z[1]["precision"]),reverse=True); return pool[0],len(feasible)
def choose_red(y,s,watch):
    rows=[]
    for t in np.unique(np.r_[np.linspace(max(watch,.35),.99,120),np.quantile(s,np.linspace(.60,.995,70))]):
        m=met(y,s,float(t));
        if m["tp"]+m["fp"]>=3 and m["precision"]>=.70: rows.append((float(t),m))
    if not rows:return .99,met(y,s,.99),0
    rows.sort(key=lambda z:(z[1]["recall"],z[1]["precision"]),reverse=True); return rows[0][0],rows[0][1],len(rows)


def main():
    raw,source_rows,eligible_rows=aggregate_weekly(FULL_GZ); panel=add_sama(complete_full_weeks(raw)); meta,X=make_dataset(panel); PANEL_CSV.write_text(pd.concat([meta,X],axis=1).to_csv(index=False),encoding="utf-8")
    forbidden=[c for c in X.columns if c.startswith("actual_") or "future" in c.lower() or "target" in c.lower()]
    if forbidden: raise RuntimeError(f"Forbidden features {forbidden}")
    tr=meta.week_start<=TRAIN_END; va=meta.week_start.between(VAL_START,VAL_END); te=meta.week_start>=TEST_START
    if min(tr.sum(),va.sum(),te.sum())<70: raise RuntimeError(f"Insufficient weekly split {tr.sum()}/{va.sum()}/{te.sum()}")
    ytr=meta.loc[tr,"target"].astype(int); yv=meta.loc[va,"target"].astype(int); yt=meta.loc[te,"target"].astype(int)
    if any(y.nunique()!=2 for y in [ytr,yv,yt]): raise RuntimeError("One split lacks both target classes")
    models={}; vs={}; ts={}; cand={}
    for name,f in factories().items():
        m=fit_cls(clone(f),X.loc[tr],ytr); pv=m.predict_proba(X.loc[va])[:,1]; pt=m.predict_proba(X.loc[te])[:,1]; models[name]=m; vs[name]=pv; ts[name]=pt; cand[name]=float(roc_auc_score(yv,pv))
    reg=HistGradientBoostingRegressor(max_iter=300,learning_rate=.025,max_leaf_nodes=10,min_samples_leaf=16,l2_regularization=10.,random_state=SEED).fit(X.loc[tr],meta.loc[tr,"future_ratio"].clip(0,2.5)); rv=reg.predict(X.loc[va]); rt=reg.predict(X.loc[te]); rr_v=1/(1+np.exp((rv-.8)/.08)); rr_t=1/(1+np.exp((rt-.8)/.08))
    cls_v=np.column_stack([vs[n] for n in factories()]).mean(1); cls_t=np.column_stack([ts[n] for n in factories()]).mean(1)
    blends=[]
    for w in (0,.15,.30,.45):
        sv=(1-w)*cls_v+w*rr_v; (t,m),nf=choose(yv,sv); blends.append((m["balanced_accuracy"],m["f1"],m["roc_auc"],-w,w,t,m,nf))
    blends.sort(reverse=True); _,_,_,_,w,watch,valm,nfeas=blends[0]; sv=(1-w)*cls_v+w*rr_v; st=(1-w)*cls_t+w*rr_t; red,redvm,nred=choose_red(yv,sv,watch); testm=met(yt,st,watch); redtm=met(yt,st,red)
    fit=meta.week_start<=VAL_END; yf=meta.loc[fit,"target"].astype(int); final_models={n:fit_cls(clone(f),X.loc[fit],yf) for n,f in factories().items()}; final_reg=clone(reg).fit(X.loc[fit],meta.loc[fit,"future_ratio"].clip(0,2.5))
    artifact={"version":VERSION,"feature_columns":list(X.columns),"models":final_models,"regressor":final_reg,"blend_weight_regression":float(w),"watch_threshold":float(watch),"red_threshold":float(red),"target_definition":"next SAMA-aligned merchant category week sales <80% trailing 4 completed category-week mean","sama_signal":"previous official sector origin h2 forecast for target week; actual future SAMA excluded","training_cutoff":str(VAL_END.date())}; joblib.dump(artifact,MODEL)
    contract={"test_roc_auc_min":.75,"test_balanced_accuracy_min":.68,"test_recall_min":.65,"test_green_npv_min":.88,"test_alert_rate_max":.55,"red_precision_min_if_exists":.60}
    redok=(redtm["tp"]+redtm["fp"]==0) or redtm["precision"]>=contract["red_precision_min_if_exists"]
    gates={"source_rows_exact_1049042":source_rows==1049042,"non_overlapping_weekly_origins":True,"future_actual_sama_excluded":True,"one_week_purge_between_splits":TRAIN_END+pd.Timedelta(days=7)<VAL_START and VAL_END+pd.Timedelta(days=7)<TEST_START,"threshold_validation_only":True,"test_untouched_for_selection":True,"test_roc_auc":testm["roc_auc"]>=contract["test_roc_auc_min"],"test_balanced_accuracy":testm["balanced_accuracy"]>=contract["test_balanced_accuracy_min"],"test_recall":testm["recall"]>=contract["test_recall_min"],"test_green_npv":testm["green_npv"]>=contract["test_green_npv_min"],"test_alert_rate":testm["alert_rate"]<=contract["test_alert_rate_max"],"red_precision":bool(redok)}
    rep={"version":VERSION,"scientific_boundary":"UCI-derived Saudi-localized synthetic merchant microdata; official SAMA is aggregate external context, not observed merchant truth.","source_rows":source_rows,"eligible_rows":eligible_rows,"weekly_panel_rows":len(panel),"supervised_rows":len(meta),"categories":meta.category.nunique(),"feature_count":X.shape[1],"split":{"train_rows":int(tr.sum()),"train_positive_rate":float(ytr.mean()),"validation_rows":int(va.sum()),"validation_positive_rate":float(yv.mean()),"test_rows":int(te.sum()),"test_positive_rate":float(yt.mean())},"validation_model_auc":cand,"selected":{"blend_weight_regression":w,"watch_threshold":watch,"red_threshold":red,"validation_metrics":valm,"validation_red_metrics":redvm,"feasible_watch_thresholds":nfeas,"feasible_red_thresholds":nred},"held_out_test":testm,"held_out_test_red":redtm,"contract":contract,"gates":gates,"all_gates_passed":bool(all(gates.values())),"leakage_controls":{"weekly_non_overlapping_target_rows":True,"sama_previous_origin_only":True,"actual_future_sama_not_feature":True,"test_not_used_for_selection":True}}
    REPORT.write_text(json.dumps(rep,indent=2),encoding="utf-8"); SUMMARY.write_text("# Sales Sentinel v4.1 — Weekly Merchant + SAMA Hybrid\n\n"+f"- Full source rows **{source_rows:,}**\n- Weekly supervised rows **{len(meta):,}**\n- Test Accuracy **{testm['accuracy']:.2%}**\n- Test Balanced Accuracy **{testm['balanced_accuracy']:.2%}**\n- Test Precision **{testm['precision']:.2%}**\n- Test Recall **{testm['recall']:.2%}**\n- Test F1 **{testm['f1']:.2%}**\n- Test ROC-AUC **{testm['roc_auc']:.2%}**\n- GREEN NPV **{testm['green_npv']:.2%}**\n- Alert rate **{testm['alert_rate']:.2%}**\n- All gates **{all(gates.values())}**\n",encoding="utf-8"); print(json.dumps(rep,indent=2))

if __name__=="__main__": main()
