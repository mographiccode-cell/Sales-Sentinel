from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import train_merchant_error_corrector_v7_5 as v75
import train_merchant_target_refinement_v8 as v8
import train_merchant_alert_verifier_v8_2 as v82

ROOT=Path(__file__).resolve().parents[1]
VERSION="SALES-SENTINEL-V8.2-FROZEN-DEVELOPMENT-CANDIDATE"
PANEL=ROOT/"data"/"merchant_v7_1"/"merchant_feature_panel_v7_1.csv"
V8_REPORT=ROOT/"reports"/"merchant_target_refinement_v8"/"development_report.json"
V82_REPORT=ROOT/"reports"/"merchant_alert_verifier_v8_2"/"development_report.json"
OUT=ROOT/"reports"/"merchant_v8_2_final"; MOD=ROOT/"models"/"merchant_v8_2_final"
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)
REPORT=OUT/"finalization_report.json"; SUMMARY=OUT/"finalization_summary.md"; MODEL=MOD/"merchant_v8_2_final.joblib"
META={"date","future_ratio","future7_sales","baseline28_daily","target"}


def fit_preprocess(X):
    z=X.copy(); state={}
    for c in z.columns:
        a=pd.to_numeric(z[c],errors="coerce").replace([np.inf,-np.inf],np.nan); f=a.dropna()
        if f.empty: lo=hi=med=0.0
        else: lo=float(f.quantile(.01)); hi=float(f.quantile(.99)); med=float(f.median())
        z[c]=a.clip(lo,hi).fillna(med); state[c]={"lo":lo,"hi":hi,"median":med}
    return z.astype(float),state


def main():
    r8=json.loads(V8_REPORT.read_text(encoding="utf-8")); r82=json.loads(V82_REPORT.read_text(encoding="utf-8"))
    rank_cfg=r8["selected"]["config"]; verifier_cfg=r82["selected"]["config"]
    if rank_cfg!={"model":"xgb2","topk":128,"weight_profile":"baseline"}: raise RuntimeError(f"Unexpected V8 selected config: {rank_cfg}")
    d=pd.read_csv(PANEL,parse_dates=["date"]).sort_values("date").reset_index(drop=True); d=v8.add_hard_negative_features(d)
    feats=[c for c in d.columns if c not in META]; X,state=fit_preprocess(d[feats]); y=d.target.astype(int)
    cols=v75.stable_top(X,y,rank_cfg["topk"]); Xfit=X[cols]
    rank_model=v8.make_model(rank_cfg["model"]); sw=v8.sample_weight_refined(y,d.future_ratio,rank_cfg["weight_profile"]); rank_model.fit(Xfit,y,sample_weight=sw)
    train_raw=rank_model.predict_proba(Xfit)[:,1]; rank_ref=np.sort(train_raw)

    a,b,diag,mx,strong,quiet,market=v82.build_oof_features(); oy=a.y.to_numpy(int); base=a.v7_5_pred.to_numpy(bool); alerts=base
    core=[c for c in mx.columns if not c.startswith("v8__")]
    sc=StandardScaler(); A=sc.fit_transform(mx.loc[alerts,core]); vy=oy[alerts]
    verifier=LogisticRegression(C=verifier_cfg["C"],class_weight="balanced",max_iter=2000,solver="liblinear",random_state=42); verifier.fit(A,vy)
    ptrain=verifier.predict_proba(A)[:,1]; tp_probs=ptrain[vy==1]
    verifier_threshold=float(max(0,np.quantile(tp_probs,verifier_cfg["tp_quantile"])-verifier_cfg["margin"]))

    artifact={
        "version":VERSION,"status":"DEVELOPMENT_FROZEN_PENDING_EXTERNAL_SAUDI_VALIDATION",
        "dependencies":{"base_decision":"V7.5 frozen candidate / V6.1 regime policy","market_context":"SAMA V3"},
        "rank_model":{"config":rank_cfg,"features":cols,"preprocess":{c:state[c] for c in cols},"model":rank_model,"training_score_reference":rank_ref},
        "alert_verifier":{"config":verifier_cfg,"features":core,"scaler":sc,"model":verifier,"true_alert_threshold":verifier_threshold,"training_alerts":int(alerts.sum()),"training_true_alerts":int(vy.sum())},
        "decision_logic":{"start_from":"V7.5 binary alert","veto_scope":verifier_cfg["scope"],"rank_guard":verifier_cfg["rank_guard"],"rescue":None,"note":"Veto only non-strong V7.5 alerts when verifier probability is below threshold AND V8 percentile rank is below rank_guard."},
        "red_supported":False,
    }
    joblib.dump(artifact,MODEL)
    primary=r82["selected"]["metrics"]
    report={"version":VERSION,"status":artifact["status"],"scientific_boundary":"The frozen artifact is fitted on all existing development data after model/configuration selection. Primary reported performance remains the causal prequential V8.2 OOF evidence, not apparent full-development fit. Fresh longitudinal Saudi merchant validation is required before production claims.","rank_config":rank_cfg,"rank_feature_count":len(cols),"verifier_config":verifier_cfg,"verifier_feature_count":len(core),"verifier_threshold":verifier_threshold,"primary_prequential_metrics":primary,"v7_5_reference":r82["v7_5"],"model_artifact":"models/merchant_v8_2_final/merchant_v8_2_final.joblib","external_validation_pending":True,"red_supported":False}
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    m=primary; b0=r82["v7_5"]
    lines=["# Sales Sentinel V8.2 — Frozen Development Candidate","",f"- Status: **{artifact['status']}**",f"- Ranking model: **XGBoost depth 2 / top {len(cols)} features**",f"- Alert verifier: **Logistic Regression / {len(core)} core meta-features**","","## Primary causal prequential evidence",f"- Accuracy: **{m['accuracy']:.2%}**",f"- Balanced accuracy: **{m['balanced_accuracy']:.2%}**",f"- Precision: **{m['precision']:.2%}**",f"- Recall: **{m['recall']:.2%}**",f"- F1: **{m['f1']:.2%}**",f"- GREEN NPV: **{m['green_npv']:.2%}**",f"- Alert rate: **{m['alert_rate']:.2%}**",f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**","",f"- FP improvement vs V7.5: **{b0['fp']} -> {m['fp']}**",f"- External Saudi merchant validation: **Pending**",f"- RED supported: **False**","","Important: do not report full-development fitted performance as validation; use the prequential metrics above."]
    SUMMARY.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
