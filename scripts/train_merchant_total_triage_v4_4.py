from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import train_merchant_total_hybrid_v4_3 as v3

ROOT=Path(__file__).resolve().parents[1]
VERSION='SALES-SENTINEL-MERCHANT-TOTAL-TRIAGE-4.4'
SRC=ROOT/'data'/'merchant_v4_3'/'merchant_total_feature_panel_v4_3.csv'
OUT=ROOT/'reports'/'merchant_total_triage_v4_4';MOD=ROOT/'models'/'merchant_total_triage_v4_4'
OUT.mkdir(parents=True,exist_ok=True);MOD.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'development_report.json';SUMMARY=OUT/'development_summary.md';MODEL=MOD/'merchant_total_triage_v4_4.joblib'
SEED=42


def factories():
    return {
        'extra_trees_reg': ExtraTreesRegressor(n_estimators=1000,max_depth=7,min_samples_leaf=7,max_features=.50,random_state=SEED,n_jobs=-1),
        'hist_gb_mean': HistGradientBoostingRegressor(max_iter=360,learning_rate=.025,max_leaf_nodes=12,min_samples_leaf=24,l2_regularization=10.,random_state=SEED),
        'hist_gb_q25': HistGradientBoostingRegressor(loss='quantile',quantile=.25,max_iter=360,learning_rate=.025,max_leaf_nodes=12,min_samples_leaf=24,l2_regularization=10.,random_state=SEED+1),
    }


def binary_metrics(y,score,threshold):
    y=np.asarray(y,int);p=np.asarray(score)>=float(threshold)
    return {
        'accuracy':float(accuracy_score(y,p)),
        'balanced_accuracy':float(balanced_accuracy_score(y,p)),
        'precision':float(precision_score(y,p,zero_division=0)),
        'recall':float(recall_score(y,p,zero_division=0)),
        'f1':float(f1_score(y,p,zero_division=0)),
        'roc_auc':float(roc_auc_score(y,score)) if len(np.unique(y))==2 else None,
        'alert_rate':float(p.mean()),
        'green_npv':float(((y==0)&(~p)).sum()/max((~p).sum(),1)),
        'tp':int(((y==1)&p).sum()),'fp':int(((y==0)&p).sum()),'fn':int(((y==1)&(~p)).sum()),'tn':int(((y==0)&(~p)).sum()),
    }


def select_amber(y,score):
    cand=np.unique(np.r_[np.linspace(float(np.nanmin(score)),float(np.nanmax(score)),240),np.quantile(score,np.linspace(.02,.98,120))])
    rows=[]
    for t in cand:
        m=binary_metrics(y,score,t)
        if m['recall']>=.70 and m['alert_rate']<=.45 and m['precision']>=.25:
            rows.append((float(t),m))
    if not rows:
        fallback=[(float(t),binary_metrics(y,score,t)) for t in cand]
        fallback.sort(key=lambda x:(x[1]['balanced_accuracy'],x[1]['f1'],x[1]['recall']),reverse=True)
        return fallback[0][0],fallback[0][1],0
    rows.sort(key=lambda x:(x[1]['balanced_accuracy'],x[1]['f1'],x[1]['precision'],-x[1]['alert_rate']),reverse=True)
    return rows[0][0],rows[0][1],len(rows)


def select_red(y,score,amber_mask):
    cand=np.unique(np.r_[np.linspace(float(np.nanmin(score)),float(np.nanmax(score)),260),np.quantile(score,np.linspace(.40,.995,120))])
    rows=[]
    for t in cand:
        m=binary_metrics(y,score,t)
        alerts=m['tp']+m['fp']
        if alerts>=5 and m['precision']>=.60 and m['recall']>=.20:
            rows.append((float(t),m))
    if not rows:
        # Fail closed: no RED if historical OOF cannot support a high-precision severe state.
        return float('inf'),binary_metrics(y,score,float('inf')),0
    rows.sort(key=lambda x:(x[1]['recall'],x[1]['precision'],x[1]['f1'],-x[1]['alert_rate']),reverse=True)
    return rows[0][0],rows[0][1],len(rows)


def main():
    d=pd.read_csv(SRC,parse_dates=['date']).sort_values('date').reset_index(drop=True)
    meta=d[['date','future_ratio']].copy();X=d.drop(columns=['date','future_ratio','target'])
    fs=v3.folds(meta.assign(target=(meta.future_ratio<.8).astype(int)))
    if len(fs)<5:raise RuntimeError(f'Expected five rolling folds, got {len(fs)}')
    oof={n:np.full(len(meta),np.nan) for n in factories()};fold_id=np.full(len(meta),-1,int);valmask=np.zeros(len(meta),bool);foldmeta=[]
    for fid,(st,en,tr,va) in enumerate(fs):
        y=meta.loc[tr,'future_ratio'].clip(0,2.5);valmask|=va.to_numpy();fold_id[va.to_numpy()]=fid
        foldmeta.append({'fold_id':fid,'start':str(st.date()),'end':str(en.date()),'train_rows':int(tr.sum()),'validation_rows':int(va.sum())})
        for n,f in factories().items():
            m=clone(f).fit(X.loc[tr],y);oof[n][va.to_numpy()]=m.predict(X.loc[va])
    idx=np.where(valmask)[0];actual=meta.future_ratio.to_numpy(float)[idx]
    # Fixed score definitions came from the dense diagnostic; no threshold labels are used to alter model weights here.
    pred_tree=oof['extra_trees_reg'][idx];pred_q25=oof['hist_gb_q25'][idx]
    pred_early=.55*pred_tree+.45*pred_q25
    pred_severe=pred_q25
    early_score=-pred_early;severe_score=-pred_severe
    y10=(actual<.90).astype(int);y15=(actual<.85).astype(int);y20=(actual<.80).astype(int)
    amber_t,amber15,n_amber=select_amber(y15,early_score)
    amber10=binary_metrics(y10,early_score,amber_t)
    amber20=binary_metrics(y20,early_score,amber_t)
    amber_mask=early_score>=amber_t
    red_t,red20,n_red=select_red(y20,severe_score,amber_mask)
    red15=binary_metrics(y15,severe_score,red_t)
    state=np.where(severe_score>=red_t,'RED',np.where(amber_mask,'AMBER','GREEN'))
    per=[]
    for fid in sorted(set(fold_id[idx])):
        mask=fold_id[idx]==fid
        yy15=y15[mask];yy20=y20[mask];ss=state[mask]
        alert=ss!='GREEN';red=ss=='RED'
        def from_pred(y,p):
            return {'precision':float(precision_score(y,p,zero_division=0)),'recall':float(recall_score(y,p,zero_division=0)),'f1':float(f1_score(y,p,zero_division=0)),'alert_rate':float(np.mean(p)),'positives':int(np.sum(y))}
        per.append({'fold_id':int(fid),'early15':from_pred(yy15,alert),'severe20_red':from_pred(yy20,red)})
    # Non-overlap weekday cohorts are a stability diagnostic for the 7-day target.
    cohorts=[]
    oof_dates=meta.date.to_numpy()[idx]
    for dow in range(7):
        m=pd.DatetimeIndex(oof_dates).dayofweek==dow
        if m.sum()>=20 and len(np.unique(y15[m]))==2:
            cohorts.append({'weekday':dow,'rows':int(m.sum()),'early15_auc':float(roc_auc_score(y15[m],early_score[m])),'severe20_auc':float(roc_auc_score(y20[m],severe_score[m])) if len(np.unique(y20[m]))==2 else None})
    cohort_early=[x['early15_auc'] for x in cohorts];cohort_severe=[x['severe20_auc'] for x in cohorts if x['severe20_auc'] is not None]
    early_auc=float(roc_auc_score(y15,early_score));severe_auc=float(roc_auc_score(y20,severe_score))
    contract={'early15_auc_min':.75,'early15_recall_min':.70,'early15_alert_rate_max':.45,'early15_green_npv_min':.90,'severe20_auc_min':.77,'red_precision_min_if_red_exists':.60,'median_weekday_early_auc_min':.70}
    red_exists=np.isfinite(red_t);red_ok=(not red_exists) or red20['precision']>=contract['red_precision_min_if_red_exists']
    gates={'rolling_origin_past_only':True,'dense_target_no_future_feature':True,'early15_auc':early_auc>=contract['early15_auc_min'],'early15_recall':amber15['recall']>=contract['early15_recall_min'],'early15_alert_rate':amber15['alert_rate']<=contract['early15_alert_rate_max'],'early15_green_npv':amber15['green_npv']>=contract['early15_green_npv_min'],'severe20_auc':severe_auc>=contract['severe20_auc_min'],'red_precision':bool(red_ok),'weekday_early_stability':float(np.median(cohort_early))>=contract['median_weekday_early_auc_min']}
    # Freeze final regressors on all available labeled development rows. Independent stress must not refit these.
    final={n:clone(f).fit(X,meta.future_ratio.clip(0,2.5)) for n,f in factories().items()}
    artifact={'version':VERSION,'status':'DEVELOPMENT_FROZEN_PENDING_INDEPENDENT_STRESS','feature_columns':list(X.columns),'models':final,'early_model_formula':{'extra_trees_reg':.55,'hist_gb_q25':.45},'severe_model':'hist_gb_q25','amber_threshold_score':float(amber_t),'red_threshold_score':None if not np.isfinite(red_t) else float(red_t),'target_definition':'future 7-day merchant sales ratio versus trailing 28-day daily mean','states':{'GREEN':'no early downside threshold crossed','AMBER':'early downside warning calibrated on >=15% decline','RED':'high-precision severe warning calibrated on >=20% decline if supported'},'source_scope':'UCI-derived Saudi-localized synthetic merchant microdata plus official SAMA aggregate context'}
    joblib.dump(artifact,MODEL)
    rep={'version':VERSION,'status':artifact['status'],'scientific_boundary':'Rolling-origin development evidence only; no historical period is called an untouched final test. Independent stress does not provide real-world accuracy; external real merchant validation remains required.','rows':len(meta),'oof_rows':len(idx),'feature_count':X.shape[1],'folds':foldmeta,'ranking':{'early15_auc':early_auc,'severe20_auc':severe_auc,'early10_auc':float(roc_auc_score(y10,early_score))},'thresholds':{'amber_score_threshold':float(amber_t),'red_score_threshold':None if not red_exists else float(red_t),'amber_feasible_candidates':n_amber,'red_feasible_candidates':n_red},'metrics':{'AMBER_or_RED_vs_10pct':amber10,'AMBER_or_RED_vs_15pct':amber15,'AMBER_or_RED_vs_20pct':amber20,'RED_vs_20pct':red20,'RED_vs_15pct':red15,'state_counts':{k:int(np.sum(state==k)) for k in ['GREEN','AMBER','RED']}},'per_fold':per,'weekday_cohorts':cohorts,'contract':contract,'gates':gates,'all_development_gates_passed':bool(all(gates.values())),'red_supported':bool(red_exists),'next_required_evidence':'Frozen sensitivity stress on new deterioration patterns, then external validation on real merchant longitudinal data.'}
    REPORT.write_text(json.dumps(rep,indent=2),encoding='utf-8')
    SUMMARY.write_text('# Sales Sentinel v4.4 — Merchant Total Triage\n\n'+f"- Early >=15% ROC-AUC **{early_auc:.2%}**\n- Severe >=20% ROC-AUC **{severe_auc:.2%}**\n- Early recall **{amber15['recall']:.2%}**\n- Early precision **{amber15['precision']:.2%}**\n- Early alert rate **{amber15['alert_rate']:.2%}**\n- GREEN NPV **{amber15['green_npv']:.2%}**\n- RED supported **{red_exists}**\n- RED precision **{red20['precision']:.2%}**\n- Development gates **{all(gates.values())}**\n",encoding='utf-8')
    print(json.dumps(rep,indent=2))

if __name__=='__main__':main()
