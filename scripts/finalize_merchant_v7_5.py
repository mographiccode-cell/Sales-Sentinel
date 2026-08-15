from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_merchant_error_corrector_v7_5 as v75

ROOT=Path(__file__).resolve().parents[1]
VERSION="SALES-SENTINEL-V7.5-FROZEN-DEVELOPMENT-CANDIDATE"
PANEL=ROOT/"data"/"merchant_v7_1"/"merchant_feature_panel_v7_1.csv"
V75_REPORT=ROOT/"reports"/"merchant_error_corrector_v7_5"/"development_report.json"
V75_OOF=ROOT/"reports"/"merchant_error_corrector_v7_5"/"oof_candidate_scores.csv"
V61_DIAG=ROOT/"reports"/"merchant_market_fusion_v6_1"/"oof_policy_diagnostics.csv"
OUT=ROOT/"reports"/"merchant_v7_5_final"; MOD=ROOT/"models"/"merchant_v7_5_final"
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)
REPORT=OUT/"finalization_report.json"; SUMMARY=OUT/"finalization_summary.md"; MODEL=MOD/"merchant_v7_5_final.joblib"
META={"date","future_ratio","future7_sales","baseline28_daily","target"}


def strict_contract(m,base):
    return bool(m["recall"]>=.80 and m["green_npv"]>=.95 and m["alert_rate"]<=base["alert_rate"] and m["worst_fold_recall"]>=.60 and m["fp"]<base["fp"] and m["f1"]>base["f1"])


def main():
    rep=json.loads(V75_REPORT.read_text(encoding="utf-8")); sel=rep["selected"]; cfg=sel["config"]; sel_id=int(sel["config_id"])
    d=pd.read_csv(PANEL,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    diag=pd.read_csv(V61_DIAG); y=diag.y.to_numpy(int); folds=diag.fold_id.to_numpy(int); mm=diag.merchant_mean.to_numpy(float); mp=diag.market_v3__risk_p90.to_numpy(float)
    strong,quiet,market,base_pred=v75.base_components(diag); base=v75.metrics(y,base_pred,folds)
    oo=pd.read_csv(V75_OOF); oo=oo[oo.config_id==sel_id].sort_values(["fold_id","date"]).reset_index(drop=True)
    if len(oo)!=381 or not np.array_equal(oo.target.to_numpy(int),y): raise RuntimeError("Selected V7.5 OOF mismatch")
    score=oo.rank_score.to_numpy(float)

    # Select a single development rule on all existing OOF. Its metrics are tuning evidence only;
    # the prequential metrics in V7.5 remain the primary performance evidence.
    best=None
    for rule in v75.rule_grid():
        p=v75.apply_rule(base_pred,strong,quiet,market,score,mm,mp,rule); m=v75.metrics(y,p,folds)
        feasible=strict_contract(m,base)
        key=(int(feasible),m["f1"],m["precision"],-m["fp"],m["recall"],-m["alert_rate"])
        if best is None or key>best[0]: best=(key,rule,m,feasible)
    _,final_rule,final_rule_metrics,rule_feasible=best

    # Threshold-neighborhood sensitivity.
    s,qv,rs,qr=final_rule
    qv_vals=[qv] if s=="none" else sorted(set(float(np.clip(qv+x,0.01,.99)) for x in [-.10,-.05,0,.05,.10]))
    qr_vals=[qr] if rs=="none" else sorted(set(float(np.clip(qr+x,0.01,.99)) for x in [-.05,-.02,0,.02,.05]))
    neighbor=[]
    for a in qv_vals:
        for b in qr_vals:
            r=(s,a,rs,b); p=v75.apply_rule(base_pred,strong,quiet,market,score,mm,mp,r); m=v75.metrics(y,p,folds)
            neighbor.append({"rule":list(r),"metrics":m,"passes_contract":strict_contract(m,base)})
    neighbor_pass=sum(x["passes_contract"] for x in neighbor)

    # Score-noise stress sensitivity; diagnostic, not validation.
    rng=np.random.default_rng(42); stress={}
    for sigma in [.01,.02,.04]:
        rows=[]
        for _ in range(100):
            noisy=np.clip(score+rng.normal(0,sigma,len(score)),0,1)
            p=v75.apply_rule(base_pred,strong,quiet,market,noisy,mm,mp,final_rule); rows.append(v75.metrics(y,p,folds))
        stress[str(sigma)]={
            "contract_pass_rate":float(np.mean([strict_contract(m,base) for m in rows])),
            "median_f1":float(np.median([m["f1"] for m in rows])),
            "median_precision":float(np.median([m["precision"] for m in rows])),
            "median_recall":float(np.median([m["recall"] for m in rows])),
            "worst_recall":float(min(m["recall"] for m in rows)),
            "max_fp":int(max(m["fp"] for m in rows)),
        }

    # Fit the final candidate ranking model on all 541 development rows.
    all_features=[c for c in d.columns if c not in META]
    if cfg["feature_mode"]=="merchant_regime": base_features=[c for c in all_features if c.startswith(("merchant__","market__","calendar__","catregime__"))]
    else: base_features=all_features
    Xfull,_,prep=v75.prepare(d[base_features],d[base_features]); yfull=d.target.astype(int)
    if cfg["feature_mode"]=="top96": features=v75.stable_top(Xfull,yfull,96)
    else: features=list(Xfull.columns)
    Xfit=Xfull[features]
    model=v75.make_model(cfg["model"],yfull)
    sw=v75.sample_weight(yfull,d.future_ratio,cfg["weighted"]); model.fit(Xfit,yfull,sample_weight=sw)
    score_reference=np.sort(model.predict_proba(Xfit)[:,1])

    artifact={
        "version":VERSION,
        "status":"DEVELOPMENT_FROZEN_PENDING_EXTERNAL_SAUDI_VALIDATION",
        "candidate_config":cfg,
        "feature_columns":features,
        "preprocessing":{c:prep[c] if isinstance(prep,dict) and c in prep else None for c in features},
        "model_object":model,
        "training_score_reference":score_reference,
        "decision_layer":{"base":"V6.1 three-branch regime policy","error_correction_rule":list(final_rule),"score_type":"percentile_against_training_score_reference"},
        "required_v6_1_inputs":["merchant_logreg","merchant_extra","merchant_mean","merchant_disagreement","market_v3__risk_p90"],
        "red_supported":False,
    }
    joblib.dump(artifact,MODEL)

    primary=sel["prequential_fusion"]
    report={
        "version":VERSION,
        "status":artifact["status"],
        "scientific_boundary":"Primary V7.5 evidence is the prequential fusion result, where correction rules for each fold are learned only from earlier folds. The final frozen rule is fitted on all development OOF for deployment-candidate packaging and its apparent metrics are not independent validation. External real Saudi merchant longitudinal validation is still required.",
        "candidate_config":cfg,
        "final_feature_count":len(features),
        "primary_prequential_metrics":primary,
        "v6_1_reference":base,
        "final_development_rule":list(final_rule),
        "final_development_rule_metrics_tuning_only":final_rule_metrics,
        "final_rule_passes_contract_on_development":bool(rule_feasible),
        "threshold_neighbor_robustness":{"tested":len(neighbor),"passing":neighbor_pass,"pass_rate":neighbor_pass/max(len(neighbor),1),"neighbors":neighbor},
        "score_noise_stress":stress,
        "model_artifact":"models/merchant_v7_5_final/merchant_v7_5_final.joblib",
        "external_validation_pending":True,
        "red_supported":False,
    }
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    m=primary
    summary=["# Sales Sentinel V7.5 — Frozen Development Candidate","",f"- Status: **{artifact['status']}**",f"- Model: **{cfg['model']} / {cfg['feature_mode']} / weighted={cfg['weighted']}**",f"- Final features: **{len(features)}**","", "## Primary prequential evidence",f"- Precision: **{m['precision']:.2%}**",f"- Recall: **{m['recall']:.2%}**",f"- F1: **{m['f1']:.2%}**",f"- GREEN NPV: **{m['green_npv']:.2%}**",f"- Alert rate: **{m['alert_rate']:.2%}**",f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**","",f"- Final development correction rule: **{list(final_rule)}**",f"- Threshold-neighbor robustness: **{neighbor_pass}/{len(neighbor)} pass**",f"- Score-noise sigma=0.01 contract pass rate: **{stress['0.01']['contract_pass_rate']:.0%}**",f"- Score-noise sigma=0.02 contract pass rate: **{stress['0.02']['contract_pass_rate']:.0%}**",f"- External Saudi merchant validation: **Pending**",f"- RED supported: **False**","","Important: final-rule tuning metrics are not independent evidence; use the prequential metrics above for academic reporting."]
    SUMMARY.write_text("\n".join(summary)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
