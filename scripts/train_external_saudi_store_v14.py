from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("SAUDI_STORE_DIR", "/tmp/saudi_store_sales"))
OUT = ROOT / "reports" / "external_saudi_store_v14"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
OOF = OUT / "development_oof.csv"
BLIND = OUT / "blind_2023_predictions.csv"
PANEL = OUT / "daily_panel_manifest.csv"
SEED = 42
TARGET_H = 7
BLIND_START = pd.Timestamp("2023-01-01")
BLIND_END = pd.Timestamp("2023-12-24")
DEV_TRAIN_END = pd.Timestamp("2022-12-24")  # 7-day target purge before blind period


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw() -> tuple[pd.DataFrame, Path]:
    files = list(RAW_DIR.rglob("*.xlsx")) + list(RAW_DIR.rglob("*.xls")) + list(RAW_DIR.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No tabular dataset found in {RAW_DIR}")
    p = sorted(files)[0]
    d = pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p, low_memory=False)
    required = {"Invoice Date", "Invoice ID", "City", "Product Name", "Product Category", "Channel", "Customer Type", "Total Sales"}
    if not required.issubset(d.columns):
        raise RuntimeError(f"Missing required columns: {sorted(required-set(d.columns))}")
    d["Invoice Date"] = pd.to_datetime(d["Invoice Date"], errors="coerce")
    d["Total Sales"] = pd.to_numeric(d["Total Sales"], errors="coerce")
    d = d.dropna(subset=["Invoice Date", "Total Sales"]).copy()
    d = d[(d["Total Sales"] >= 0) & np.isfinite(d["Total Sales"])].copy()
    return d, p


def satisfaction_score(x: pd.Series) -> pd.Series:
    mp = {"Very Low": 1, "Low": 2, "Ok": 3, "High": 4, "Very High": 5}
    return x.astype(str).map(mp).fillna(3).astype(float)


def daily_panel(tx: pd.DataFrame) -> pd.DataFrame:
    x = tx.copy()
    x["date"] = x["Invoice Date"].dt.normalize()
    x["is_online"] = x["Channel"].astype(str).str.lower().str.contains("online").astype(float)
    x["is_loyal"] = x["Customer Type"].astype(str).str.lower().str.contains("loyal").astype(float)
    x["sat_score"] = satisfaction_score(x.get("Customer Satisfaction", pd.Series(index=x.index, dtype=object)))

    rows = []
    for dt, g in x.groupby("date", sort=True):
        sales = g["Total Sales"].to_numpy(float)
        total = float(sales.sum())
        cat_sales = g.groupby("Product Category")["Total Sales"].sum().to_numpy(float)
        city_sales = g.groupby("City")["Total Sales"].sum().to_numpy(float)
        cat_share = cat_sales / total if total > 0 else np.zeros_like(cat_sales)
        city_share = city_sales / total if total > 0 else np.zeros_like(city_sales)
        rows.append({
            "date": dt,
            "sales": total,
            "invoices": int(g["Invoice ID"].nunique()),
            "customers": int(g["Customer Name"].nunique()) if "Customer Name" in g else 0,
            "products": int(g["Product Name"].nunique()),
            "categories": int(g["Product Category"].nunique()),
            "cities": int(g["City"].nunique()),
            "avg_invoice_sales": float(g.groupby("Invoice ID")["Total Sales"].sum().mean()),
            "median_invoice_sales": float(g.groupby("Invoice ID")["Total Sales"].sum().median()),
            "online_share": float(np.average(g["is_online"], weights=np.clip(g["Total Sales"], 0, None))) if total > 0 else 0.0,
            "loyal_share": float(np.average(g["is_loyal"], weights=np.clip(g["Total Sales"], 0, None))) if total > 0 else 0.0,
            "satisfaction_mean": float(g["sat_score"].mean()),
            "category_hhi": float(np.sum(cat_share**2)) if len(cat_share) else 0.0,
            "category_max_share": float(cat_share.max()) if len(cat_share) else 0.0,
            "city_hhi": float(np.sum(city_share**2)) if len(city_share) else 0.0,
            "city_max_share": float(city_share.max()) if len(city_share) else 0.0,
        })
    d = pd.DataFrame(rows).set_index("date").sort_index()
    full = pd.date_range(d.index.min(), d.index.max(), freq="D")
    d = d.reindex(full)
    count_cols = ["sales", "invoices", "customers", "products", "categories", "cities", "avg_invoice_sales", "median_invoice_sales"]
    d[count_cols] = d[count_cols].fillna(0.0)
    for c in ["online_share", "loyal_share", "satisfaction_mean", "category_hhi", "category_max_share", "city_hhi", "city_max_share"]:
        d[c] = d[c].ffill().fillna(0.0)
    d.index.name = "date"
    return d.reset_index()


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    z = d.copy().sort_values("date").reset_index(drop=True)
    base_cols = ["sales", "invoices", "customers", "products", "categories", "cities", "avg_invoice_sales", "median_invoice_sales", "online_share", "loyal_share", "satisfaction_mean", "category_hhi", "category_max_share", "city_hhi", "city_max_share"]
    for c in base_cols:
        s = pd.to_numeric(z[c], errors="coerce").fillna(0.0)
        for lag in [1, 2, 3, 7, 14, 21, 28, 56]:
            z[f"{c}__lag{lag}"] = s.shift(lag)
        for w in [7, 14, 28, 56]:
            r = s.rolling(w, min_periods=max(3, w//2))
            z[f"{c}__mean{w}"] = r.mean()
            z[f"{c}__std{w}"] = r.std()
        z[f"{c}__r7_28"] = z[f"{c}__mean7"] / z[f"{c}__mean28"].replace(0, np.nan)
        z[f"{c}__r14_28"] = z[f"{c}__mean14"] / z[f"{c}__mean28"].replace(0, np.nan)
        z[f"{c}__chg7"] = s.pct_change(7, fill_method=None).replace([np.inf, -np.inf], np.nan)
        z[f"{c}__chg28"] = s.pct_change(28, fill_method=None).replace([np.inf, -np.inf], np.nan)

    dt = z["date"]
    z["dow_sin"] = np.sin(2*np.pi*dt.dt.dayofweek/7)
    z["dow_cos"] = np.cos(2*np.pi*dt.dt.dayofweek/7)
    z["month_sin"] = np.sin(2*np.pi*(dt.dt.month-1)/12)
    z["month_cos"] = np.cos(2*np.pi*(dt.dt.month-1)/12)
    z["is_weekend_fri_sat"] = dt.dt.dayofweek.isin([4,5]).astype(float)
    z["is_month_start"] = dt.dt.is_month_start.astype(float)
    z["is_month_end"] = dt.dt.is_month_end.astype(float)
    z["salary_window"] = dt.dt.day.isin([25,26,27,28,29,30,31,1,2,3]).astype(float)

    # Frozen target definition identical to the merchant project.
    z["baseline28_daily"] = z["sales"].rolling(28, min_periods=28).mean()
    z["future7_sales"] = sum(z["sales"].shift(-k) for k in range(1,8))
    z["future_ratio"] = z["future7_sales"] / (7*z["baseline28_daily"].replace(0, np.nan))
    z["target"] = np.where(z["future_ratio"].notna(), (z["future_ratio"] < .85).astype(float), np.nan)
    return z


def prepare(Xtr: pd.DataFrame, Xva: pd.DataFrame):
    a=Xtr.copy(); b=Xva.copy(); meta={}
    for c in a.columns:
        q=pd.to_numeric(a[c],errors="coerce").replace([np.inf,-np.inf],np.nan); good=q.dropna()
        if good.empty: lo=hi=med=0.0
        else: lo=float(good.quantile(.01)); hi=float(good.quantile(.99)); med=float(good.median())
        a[c]=pd.to_numeric(a[c],errors="coerce").clip(lo,hi).fillna(med)
        b[c]=pd.to_numeric(b[c],errors="coerce").clip(lo,hi).fillna(med)
        meta[c]={"lo":lo,"hi":hi,"median":med}
    return a.astype(float),b.astype(float),meta


def make_model(name: str, y: np.ndarray):
    y=np.asarray(y,int); pos=max((y==1).sum(),1); neg=max((y==0).sum(),1); spw=float(neg/pos)
    if name=="logistic": return make_pipeline(StandardScaler(),LogisticRegression(C=.25,class_weight="balanced",max_iter=3000,random_state=SEED))
    if name=="extra": return ExtraTreesClassifier(n_estimators=600,max_depth=6,min_samples_leaf=4,max_features=.7,class_weight="balanced",random_state=SEED,n_jobs=2)
    if name=="hist": return HistGradientBoostingClassifier(max_iter=350,learning_rate=.035,max_leaf_nodes=9,l2_regularization=10.0,min_samples_leaf=12,random_state=SEED)
    if name=="xgb": return XGBClassifier(n_estimators=500,max_depth=2,learning_rate=.025,min_child_weight=8,subsample=.86,colsample_bytree=.72,reg_alpha=2.0,reg_lambda=14.0,gamma=.15,objective="binary:logistic",eval_metric="logloss",random_state=SEED,n_jobs=2,scale_pos_weight=spw)
    if name=="catboost": return CatBoostClassifier(iterations=500,depth=4,learning_rate=.025,l2_leaf_reg=14.0,random_seed=SEED,verbose=False,allow_writing_files=False,loss_function="Logloss",auto_class_weights="Balanced")
    raise KeyError(name)


def predict_proba(model,X):
    p=model.predict_proba(X)
    return np.asarray(p)[:,1]


def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,bool)
    tp=int(((y==1)&p).sum()); fp=int(((y==0)&p).sum()); fn=int(((y==1)&(~p)).sum()); tn=int(((y==0)&(~p)).sum())
    return {
        "precision":float(tp/max(tp+fp,1)),"recall":float(tp/max(tp+fn,1)),"f1":float(2*tp/max(2*tp+fp+fn,1)),
        "accuracy":float((tp+tn)/len(y)),"balanced_accuracy":float(.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1))),
        "npv":float(tn/max(tn+fn,1)),"alert_rate":float(p.mean()),"tp":tp,"fp":fp,"fn":fn,"tn":tn,
    }


def select_threshold(y,score):
    best=None
    for t in np.linspace(.05,.95,181):
        m=metrics(y,score>=t)
        feasible=m["recall"]>=.80
        key=(int(feasible),m["f1"],m["precision"],m["npv"],-m["alert_rate"],m["balanced_accuracy"])
        if best is None or key>best[0]: best=(key,float(t),m)
    return best[1],best[2]


def dev_folds():
    return [
        ("2021-01-08","2021-06-30"),
        ("2021-07-08","2021-12-31"),
        ("2022-01-08","2022-06-30"),
        ("2022-07-08","2022-12-24"),
    ]


def main():
    raw, raw_path = load_raw()
    daily = daily_panel(raw)
    data = add_features(daily)
    feature_cols=[c for c in data.columns if c not in {"date","baseline28_daily","future7_sales","future_ratio","target"}]
    # remove rows without 56-day history or label
    valid=data.target.notna() & data["sales__lag56"].notna()
    data=data.loc[valid].reset_index(drop=True)

    blind_mask=data.date.between(BLIND_START,BLIND_END)
    # Blind labels are never consulted during candidate selection below.
    dev=data[data.date < BLIND_START].copy()
    blind=data[blind_mask].copy()
    prevalence={"development":float(dev.target.mean()),"blind_2023":float(blind.target.mean())}

    model_names=["logistic","extra","hist","xgb","catboost"]
    candidates=[]; oof_parts=[]
    for name in model_names:
        parts=[]
        for fid,(a,b) in enumerate(dev_folds()):
            a=pd.Timestamp(a); b=pd.Timestamp(b)
            tr=(dev.date <= a-pd.Timedelta(days=8))
            va=dev.date.between(a,b)
            if tr.sum()<120 or va.sum()<30: continue
            Xtr,Xva,_=prepare(dev.loc[tr,feature_cols],dev.loc[va,feature_cols]); ytr=dev.loc[tr,"target"].astype(int).to_numpy(); yy=dev.loc[va,"target"].astype(int).to_numpy()
            model=make_model(name,ytr); model.fit(Xtr,ytr); sc=predict_proba(model,Xva)
            parts.append(pd.DataFrame({"date":dev.loc[va,"date"].to_numpy(),"y":yy,"score":sc,"fold_id":fid,"model":name}))
        o=pd.concat(parts,ignore_index=True).sort_values(["fold_id","date"]).reset_index(drop=True)
        y=o.y.to_numpy(int); sc=o.score.to_numpy(float); th,tm=select_threshold(y,sc)
        cand={"model":name,"oof_rows":len(o),"roc_auc":float(roc_auc_score(y,sc)),"pr_auc":float(average_precision_score(y,sc)),"threshold":th,"threshold_metrics":tm}
        candidates.append(cand); oof_parts.append(o)

    def cand_key(c):
        m=c["threshold_metrics"]
        return (int(m["recall"]>=.80),m["f1"],c["pr_auc"],c["roc_auc"],m["precision"],m["npv"],-m["alert_rate"])
    selected=max(candidates,key=cand_key); selected_name=selected["model"]; threshold=float(selected["threshold"])
    pd.concat(oof_parts,ignore_index=True).to_csv(OOF,index=False)

    # Freeze model using only labels whose 7-day target ends before 2023-01-01.
    train_final=data[(data.date<=DEV_TRAIN_END)].copy()
    Xtr,Xbl,prep=prepare(train_final[feature_cols],blind[feature_cols]); ytr=train_final.target.astype(int).to_numpy(); ybl=blind.target.astype(int).to_numpy()
    model=make_model(selected_name,ytr); model.fit(Xtr,ytr); blind_score=predict_proba(model,Xbl); blind_pred=blind_score>=threshold
    blind_metrics=metrics(ybl,blind_pred); blind_auc=float(roc_auc_score(ybl,blind_score)); blind_pr=float(average_precision_score(ybl,blind_score))

    pd.DataFrame({"date":blind.date,"y":ybl,"score":blind_score,"pred":blind_pred.astype(int),"future_ratio":blind.future_ratio}).to_csv(BLIND,index=False)
    pd.DataFrame({"metric":["raw_sha256","raw_rows","daily_days","development_rows","blind_rows","feature_count"],"value":[sha256(raw_path),len(raw),len(daily),len(dev),len(blind),len(feature_cols)]}).to_csv(PANEL,index=False)

    report={
        "version":"SALES-SENTINEL-V14-EXTERNAL-DATASET-BLIND-2023",
        "status":"EXTERNAL_DATASET_BLIND_EVALUATED",
        "provenance_boundary":"Public Kaggle dataset labeled as Store Sales in Saudi Arabia; operational/company provenance is not independently verified. Results are an external-dataset generalization experiment, not validation on verified real Saudi merchant records.",
        "source":{"kaggle":"shilton123456/sales-in-saudi-arabia","raw_file":raw_path.name,"sha256":sha256(raw_path),"raw_rows":len(raw),"date_min":str(raw['Invoice Date'].min().date()),"date_max":str(raw['Invoice Date'].max().date())},
        "task":{"target":"next 7-day sales / (7 * trailing 28-day mean including prediction day) < 0.85","development_period":"2020-02/03 through 2022-12-24","blind_period":"2023-01-01 through 2023-12-24","blind_touched_during_selection":False,"purge_days":7},
        "rows":{"daily_days":len(daily),"usable_model_rows":len(data),"development":len(dev),"final_training":len(train_final),"blind_2023":len(blind)},
        "prevalence":prevalence,
        "feature_count":len(feature_cols),
        "development_candidates":candidates,
        "selected_development_model":selected,
        "blind_2023":{"roc_auc":blind_auc,"pr_auc":blind_pr,"metrics":blind_metrics},
    }
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    bm=blind_metrics; dm=selected["threshold_metrics"]
    lines=[
        "# Sales Sentinel V14 — External Dataset Blind 2023 Experiment","",
        "- Status: **EXTERNAL_DATASET_BLIND_EVALUATED**",
        "- Provenance: **Saudi-labeled public Kaggle retail dataset; operational provenance unverified**",
        f"- Raw rows / daily days: **{len(raw):,} / {len(daily):,}**",
        f"- Development / blind rows: **{len(dev):,} / {len(blind):,}**",
        f"- Blind period: **2023-01-01 → 2023-12-24**",
        f"- Selected model from development only: **{selected_name}**",
        f"- Frozen threshold: **{threshold:.3f}**", "",
        "## Development OOF",
        f"- ROC-AUC / PR-AUC: **{selected['roc_auc']:.2%} / {selected['pr_auc']:.2%}**",
        f"- Precision / Recall / F1: **{dm['precision']:.2%} / {dm['recall']:.2%} / {dm['f1']:.2%}**",
        f"- NPV / Alert rate: **{dm['npv']:.2%} / {dm['alert_rate']:.2%}**", "",
        "## Untouched 2023 blind holdout",
        f"- ROC-AUC / PR-AUC: **{blind_auc:.2%} / {blind_pr:.2%}**",
        f"- Accuracy / Balanced Accuracy: **{bm['accuracy']:.2%} / {bm['balanced_accuracy']:.2%}**",
        f"- Precision / Recall / F1: **{bm['precision']:.2%} / {bm['recall']:.2%} / {bm['f1']:.2%}**",
        f"- NPV / Alert rate: **{bm['npv']:.2%} / {bm['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{bm['tp']}/{bm['fp']}/{bm['fn']}/{bm['tn']}**", "",
        "Important: this is independent temporal evidence on a different public Saudi-labeled dataset, but not proof of performance on verified real Saudi merchant operational data.",
    ]
    SUMMARY.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
