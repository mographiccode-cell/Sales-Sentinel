from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

import run_sama_city_risk_v2_1 as source
import train_sama_city_risk_v2_2 as v22

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'
FRESH=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2025_2026_holdout.csv'
HISTORY=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2025.csv'
MODEL=ROOT/'models'/'sama_city_v2_2'/'city_market_risk_v2_2.joblib'
OUT=ROOT/'reports'/'sama_city_v2_2'/'post_diagnosis_stress_test.json'
SUMMARY=ROOT/'reports'/'sama_city_v2_2'/'post_diagnosis_stress_summary.md'


def confusion(y,p,t):
    y=np.asarray(y,int); pred=np.asarray(p)>=t; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn),'precision':float(tp/max(tp+fp,1)),'recall':float(tp/max(tp+fn,1)),'FPR':float(fp/max(fp+tn,1)),'NPV':float(tn/max(tn+fn,1))}

def drift(dev,fresh):
    rows=[]
    for c in dev.columns:
        a=dev[c].astype(float); b=fresh[c].astype(float); sd=float(a.std(ddof=0)); z=float((b.mean()-a.mean())/(sd if sd>1e-12 else 1))
        rows.append({'feature':c,'standardized_mean_shift':z,'abs_shift':abs(z)})
    return sorted(rows,key=lambda x:x['abs_shift'],reverse=True)

def main():
    art=joblib.load(MODEL)
    if art['version']!=v22.VERSION: raise RuntimeError('Unexpected v2.2 artifact')
    dh,Xh=v22.featureize(source.reconciled_load_panel(HISTORY)); kh=dh.week_start<=pd.Timestamp(art['development_end']); dh=dh.loc[kh].reset_index(drop=True); Xh=Xh.loc[kh].reset_index(drop=True)
    de,Xe=v22.featureize(source.reconciled_load_panel(EXT)); fresh=pd.read_csv(FRESH,parse_dates=['week_start']); sets={c:set(q.week_start.dt.normalize()) for c,q in fresh.groupby('city')}
    mask=[]
    for w,c in zip(de.week_start,de.city):
        ww=pd.Timestamp(w).normalize(); weeks=sorted(sets.get(c,set())); mask.append(bool(weeks) and ww in sets[c] and ww!=weeks[-1])
    d=de.loc[mask].reset_index(drop=True); X=Xe.loc[mask].reset_index(drop=True)
    if len(d)<500: raise RuntimeError(f'Too few stress rows: {len(d)}')
    # Exact feature order is frozen in the artifact.
    X=X[art['features']]; raw=art['model'].predict_proba(X)[:,1]; p=art['calibrator'].predict_proba(pd.DataFrame({art['selected']:raw}))[:,1]
    wt=float(art['watch_threshold']); rt=float(art['red_threshold']); red=confusion(d.target,p,rt); alert=confusion(d.target,p,wt)
    rank={'ROC_AUC':float(roc_auc_score(d.target,p)),'PR_AUC':float(average_precision_score(d.target,p))}; brier=float(brier_score_loss(d.target,np.clip(p,1e-6,1-1e-6)))
    gates={'red_precision':red['precision']>=art['contract']['red_precision_min'],'red_fpr':red['FPR']<=art['contract']['red_fpr_max'],'alert_recall':alert['recall']>=art['contract']['alert_recall_min'],'green_npv':alert['NPV']>=art['contract']['green_npv_min'],'roc_auc':rank['ROC_AUC']>=art['contract']['roc_auc_min'],'pr_auc':rank['PR_AUC']>=art['contract']['pr_auc_min']}
    shifts=drift(Xh[art['features']],X)
    report={'version':'SAMA-CITY-RISK-2.2-POST-DIAGNOSIS-STRESS','independence_status':'NOT_A_FRESH_HOLDOUT: v2.2 architecture was designed after diagnosing aggregate v2.1 performance on this same 2025-2026 period. No v2.2 weights or thresholds are tuned on these rows, but this result is a stress backtest, not independent proof.','rows':len(d),'declines':int(d.target.sum()),'decline_rate':float(d.target.mean()),'period':{'start':str(d.week_start.min().date()),'end':str(d.week_start.max().date())},'fixed_thresholds':{'watch':wt,'red':rt},'RED':red,'RED_plus_AMBER':alert,'ranking':rank,'Brier':brier,'contract':art['contract'],'gates':gates,'all_gates_passed':bool(all(gates.values())),'feature_drift':{'max_abs_standardized_mean_shift':float(max(x['abs_shift'] for x in shifts)),'features_over_1sd':int(sum(x['abs_shift']>1 for x in shifts)),'top_20':shifts[:20]},'anti_leakage':{'v2_2_training_used_fresh_rows':False,'v2_2_thresholds_selected_on_fresh_rows':False,'target_prevalence_features_use_only_shifted_completed_labels':True,'future_values_as_features':False}}
    OUT.write_text(json.dumps(report,indent=2),encoding='utf-8'); SUMMARY.write_text(f'''# SAMA City Risk v2.2 — Post-Diagnosis Stress Test\n\n**Not an independent fresh holdout.** No v2.2 weights/thresholds were fitted on these rows, but the architecture was designed after v2.1 diagnosis of this period.\n\n- Rows: **{len(d):,}**\n- Declines: **{d.target.sum()} ({d.target.mean():.2%})**\n- RED precision: **{red['precision']:.2%}** ({red['TP']} TP / {red['FP']} FP)\n- RED FPR: **{red['FPR']:.2%}**\n- RED+AMBER recall: **{alert['recall']:.2%}**\n- GREEN NPV: **{alert['NPV']:.2%}**\n- Missed declines: **{alert['FN']}**\n- PR-AUC: **{rank['PR_AUC']:.2%}**\n- ROC-AUC: **{rank['ROC_AUC']:.2%}**\n- Max stationary-feature mean shift: **{report['feature_drift']['max_abs_standardized_mean_shift']:.2f} SD**\n- Stress gates passed: **{report['all_gates_passed']}**\n''',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
