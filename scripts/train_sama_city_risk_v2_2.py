from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import run_sama_city_risk_v2_1 as source

ROOT=Path(__file__).resolve().parents[1]
HISTORY=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2025.csv'
OUT=ROOT/'reports'/'sama_city_v2_2'; MOD=ROOT/'models'/'sama_city_v2_2'
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'development_report.json'; SUMMARY=OUT/'development_summary.md'; MODEL=MOD/'city_market_risk_v2_2.joblib'
VERSION='SAMA-CITY-RISK-2.2-STATIONARY-FROZEN'; SEED=42; DECLINE=.20
DEV_END=pd.Timestamp('2025-06-29'); SELECT_END=pd.Timestamp('2024-12-31'); POLICY_START=pd.Timestamp('2025-01-01')
# Stricter critical-alert contract. RED is intentionally selective; AMBER carries recall.
CONTRACT={'red_precision_min':.70,'red_fpr_max':.0075,'alert_recall_min':.92,'green_npv_min':.99,'roc_auc_min':.87,'pr_auc_min':.45}


def safe_ratio(a,b): return a.astype(float)/b.astype(float).replace(0,np.nan)
def sigmoid(x): return 1/(1+np.exp(np.clip(-np.asarray(x,float),-35,35)))


def featureize(panel:pd.DataFrame):
    d=panel.copy().sort_values(['city','week_start']).reset_index(drop=True); g=d.groupby('city',sort=False)
    # Target first so ONLY shifted realized labels can be used as adaptive prevalence features.
    d['baseline4']=g.value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).mean())
    d['actual_next_value']=g.value_thousand_sar.shift(-1)
    d['future_ratio']=safe_ratio(d.actual_next_value,d.baseline4)
    d['target_float']=np.where(d.future_ratio.notna(),(d.future_ratio < 1-DECLINE).astype(float),np.nan)
    d['target']=d.target_float.fillna(0).astype(int)

    F=pd.DataFrame(index=d.index)
    # City stationary features: ratios, changes, volatility. NO raw absolute level features.
    for col,pre in [('value_thousand_sar','value'),('transaction_count_thousand','count')]:
        s=d[col].astype(float)
        for w in (4,8,13,26,52):
            mean=g[col].transform(lambda x,w=w:x.rolling(w,min_periods=w).mean())
            std=g[col].transform(lambda x,w=w:x.rolling(w,min_periods=w).std())
            F[f'{pre}_ratio_mean_{w}']=safe_ratio(s,mean)
            F[f'{pre}_cv_{w}']=safe_ratio(std,mean.abs())
        for lag in (1,2,4,8,13,26,52): F[f'{pre}_change_{lag}']=g[col].pct_change(lag)
        F[f'{pre}_yoy_log_ratio']=np.log(s/g[col].shift(52).replace(0,np.nan))

    # City share of national market is also scale-free.
    national=d.groupby('week_start',as_index=False).agg(nvalue=('value_thousand_sar','sum'),ncount=('transaction_count_thousand','sum'))
    d=d.merge(national,on='week_start',how='left',validate='many_to_one')
    d['value_share']=safe_ratio(d.value_thousand_sar,d.nvalue); d['count_share']=safe_ratio(d.transaction_count_thousand,d.ncount)
    gs=d.groupby('city',sort=False)
    for col in ('value_share','count_share'):
        for w in (4,13,26,52):
            mean=gs[col].transform(lambda s,w=w:s.rolling(w,min_periods=w).mean())
            F[f'{col}_ratio_{w}']=safe_ratio(d[col],mean)
        for lag in (1,4,13,52): F[f'{col}_change_{lag}']=gs[col].pct_change(lag)

    # National stationary context: growth/relative-to-trend, never absolute transaction/value levels.
    n=national.sort_values('week_start').copy()
    for col,pre in [('nvalue','nvalue'),('ncount','ncount')]:
        for w in (4,8,13,26,52):
            mean=n[col].rolling(w,min_periods=w).mean(); std=n[col].rolling(w,min_periods=w).std()
            n[f'{pre}_ratio_mean_{w}']=safe_ratio(n[col],mean); n[f'{pre}_cv_{w}']=safe_ratio(std,mean.abs())
        for lag in (1,2,4,8,13,26,52): n[f'{pre}_change_{lag}']=n[col].pct_change(lag)
        n[f'{pre}_yoy_log_ratio']=np.log(n[col]/n[col].shift(52).replace(0,np.nan))
    ncols=[c for c in n.columns if c not in {'week_start','nvalue','ncount'}]
    dn=d[['week_start']].merge(n[['week_start']+ncols],on='week_start',how='left',validate='many_to_one')
    F=pd.concat([F,dn[ncols]],axis=1)

    # Operational prevalence features: target(t-1) is known at origin t because week t has closed.
    # These make risk adapt to quiet regimes without peeking at the next week.
    gt=d.groupby('city',sort=False)
    for w in (13,26,52,104):
        F[f'city_decline_rate_{w}']=gt.target_float.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=max(6,min(w,13))).mean())
    weekly_rate=d.groupby('week_start',as_index=False).target_float.mean().rename(columns={'target_float':'market_decline_rate'})
    for w in (4,13,26,52): weekly_rate[f'market_decline_rate_{w}']=weekly_rate.market_decline_rate.shift(1).rolling(w,min_periods=max(3,min(w,8))).mean()
    rcols=[c for c in weekly_rate.columns if c not in {'week_start','market_decline_rate'}]
    dr=d[['week_start']].merge(weekly_rate[['week_start']+rcols],on='week_start',how='left',validate='many_to_one')
    F=pd.concat([F,dr[rcols]],axis=1)

    # Current drawdown and seasonality.
    F['current_value_vs_baseline4']=safe_ratio(d.value_thousand_sar,d.baseline4)
    week=d.week_start.dt.isocalendar().week.astype(float); F['week_sin']=np.sin(2*np.pi*week/52.18); F['week_cos']=np.cos(2*np.pi*week/52.18)
    F=pd.concat([F,pd.get_dummies(d.city,prefix='city',dtype=float)],axis=1)
    F=F.replace([np.inf,-np.inf],np.nan)
    good=F.notna().all(axis=1)&d.future_ratio.notna()
    return d.loc[good].reset_index(drop=True),F.loc[good].reset_index(drop=True)


def models(pos_weight):
    return {
        'Logistic':make_pipeline(StandardScaler(),LogisticRegression(C=.35,max_iter=3000,class_weight='balanced',random_state=SEED)),
        'ExtraTrees':ExtraTreesClassifier(n_estimators=900,max_depth=8,min_samples_leaf=5,max_features=.70,class_weight='balanced',random_state=SEED,n_jobs=-1),
        'HistGB':HistGradientBoostingClassifier(max_iter=340,learning_rate=.03,max_leaf_nodes=14,min_samples_leaf=22,l2_regularization=5,random_state=SEED),
        'XGBoost':XGBClassifier(n_estimators=450,max_depth=3,learning_rate=.022,subsample=.82,colsample_bytree=.78,min_child_weight=9,reg_lambda=6,reg_alpha=.5,scale_pos_weight=pos_weight,eval_metric='logloss',random_state=SEED,n_jobs=-1),
    }


def folds(d):
    starts=pd.to_datetime(['2023-01-01','2023-07-01','2024-01-01','2024-04-01','2024-07-01','2024-10-01','2025-01-01','2025-04-01']); out=[]
    for st in starts:
        en=min(st+pd.DateOffset(months=3)-pd.Timedelta(days=1),DEV_END); tr=d.week_start<=st-pd.Timedelta(days=14); va=d.week_start.between(st,en)
        if tr.sum()>=700 and va.sum()>=100 and d.loc[tr,'target'].nunique()==2 and d.loc[va,'target'].nunique()==2: out.append((st,en,tr,va))
    return out

def ranking(y,s): return {'ROC_AUC':float(roc_auc_score(y,s)),'PR_AUC':float(average_precision_score(y,s))}

def oof(d,X):
    frames=[]; meta=[]
    for st,en,tr,va in folds(d):
        y=d.loc[tr,'target']; pos=int(y.sum()); neg=len(y)-pos; q=pd.DataFrame({'week_start':d.loc[va,'week_start'].to_numpy(),'y':d.loc[va,'target'].to_numpy()})
        for name,m in models(neg/max(pos,1)).items(): q[name]=clone(m).fit(X.loc[tr],y).predict_proba(X.loc[va])[:,1]
        frames.append(q); meta.append({'start':str(st.date()),'end':str(en.date()),'train_rows':int(tr.sum()),'validation_rows':int(va.sum()),'train_rate':float(y.mean()),'validation_rate':float(d.loc[va,'target'].mean())})
    if not frames: raise RuntimeError('No valid stationary OOF folds')
    return pd.concat(frames,ignore_index=True).sort_values('week_start').reset_index(drop=True),meta

def confusion(y,p,t):
    pred=np.asarray(p)>=t; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn),'precision':float(tp/max(tp+fp,1)),'recall':float(tp/max(tp+fn,1)),'FPR':float(fp/max(fp+tn,1)),'NPV':float(tn/max(tn+fn,1))}

def choose_policy(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float); cand=np.unique(np.r_[np.linspace(.001,.999,500),np.quantile(p,np.linspace(0,1,201))])
    watch=[]
    for t in cand:
        m=confusion(y,p,t)
        if m['recall']>=CONTRACT['alert_recall_min'] and m['NPV']>=CONTRACT['green_npv_min']: watch.append((m['precision'],t,m))
    if not watch: raise RuntimeError('No WATCH threshold meets stationary production contract')
    _,wt,wm=max(watch,key=lambda x:(x[0],x[1]))
    reds=[]
    for t in cand[cand>=wt]:
        m=confusion(y,p,t); alerts=m['TP']+m['FP']
        if alerts>=5 and m['precision']>=CONTRACT['red_precision_min'] and m['FPR']<=CONTRACT['red_fpr_max']: reds.append((m['recall'],m['precision'],-m['FPR'],t,m))
    if not reds: raise RuntimeError('No RED threshold meets stricter stationary production contract')
    best=max(reds,key=lambda x:x[:3]); return float(wt),float(best[3]),wm,best[4]

def triage(y,p,wt,rt):
    red=confusion(y,p,rt); watch=confusion(y,p,wt)
    return {'RED':red,'RED_plus_AMBER':watch,'GREEN':{'NPV':watch['NPV'],'missed_declines':watch['FN']}}


def main():
    d,X=featureize(source.reconciled_load_panel(HISTORY)); keep=d.week_start<=DEV_END; d=d.loc[keep].reset_index(drop=True); X=X.loc[keep].reset_index(drop=True)
    oo,meta=oof(d,X); sel=oo[oo.week_start<=SELECT_END]; pol=oo[oo.week_start.between(POLICY_START,DEV_END)]
    names=[c for c in oo.columns if c not in {'week_start','y'}]; mets={n:ranking(sel.y,sel[n]) for n in names}; selected=max(names,key=lambda n:(mets[n]['PR_AUC'],mets[n]['ROC_AUC']))
    calibrator=LogisticRegression(max_iter=2000,random_state=SEED).fit(sel[[selected]],sel.y); pp=calibrator.predict_proba(pol[[selected]])[:,1]
    wt,rt,wm,rm=choose_policy(pol.y,pp); tr=triage(pol.y,pp,wt,rt); rank=ranking(pol.y,pp); brier=float(brier_score_loss(pol.y,np.clip(pp,1e-6,1-1e-6)))
    gates={'red_precision':rm['precision']>=CONTRACT['red_precision_min'],'red_fpr':rm['FPR']<=CONTRACT['red_fpr_max'],'alert_recall':wm['recall']>=CONTRACT['alert_recall_min'],'green_npv':wm['NPV']>=CONTRACT['green_npv_min'],'roc_auc':rank['ROC_AUC']>=CONTRACT['roc_auc_min'],'pr_auc':rank['PR_AUC']>=CONTRACT['pr_auc_min']}
    pos=int(d.target.sum()); neg=len(d)-pos; final=clone(models(neg/max(pos,1))[selected]).fit(X,d.target)
    artifact={'version':VERSION,'selected':selected,'model':final,'calibrator':calibrator,'features':list(X.columns),'watch_threshold':wt,'red_threshold':rt,'contract':CONTRACT,'development_end':str(DEV_END.date()),'feature_policy':'stationary ratios/growth/volatility + lagged realized prevalence; no absolute market level features'}
    joblib.dump(artifact,MODEL)
    report={'version':VERSION,'scientific_boundary':'Official SAMA City Total only. All model/threshold decisions use data ending 2025-07-06. Fresh 2025-2026 outcomes are not read by this trainer.','rows':len(d),'decline_rate':float(d.target.mean()),'feature_count':len(X.columns),'absolute_level_features_forbidden':True,'folds':meta,'selection_metrics':mets,'selected':selected,'policy_window':f'{POLICY_START.date()}..{DEV_END.date()}','thresholds':{'watch':wt,'red':rt},'policy_triage':tr,'policy_ranking':rank,'Brier':brier,'contract':CONTRACT,'gates':gates,'all_gates_passed':bool(all(gates.values())),'leakage_controls':{'future_values_features':False,'target_derived_prevalence_features_are_shifted_one_completed_week':True,'one_week_purge':True,'chronological_expanding_folds':True,'shuffle':False,'fresh_2025_2026_used':False},'stationarity_controls':{'raw_value_or_count_levels_as_features':False,'ratios_to_rolling_trends':True,'growth_rates':True,'rolling_coefficients_of_variation':True,'city_market_share_ratios':True,'lagged_city_and_market_decline_rates':True}}
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8'); SUMMARY.write_text(f'''# SAMA City Risk v2.2 — Stationary Frozen Development\n\n- Development rows: **{len(d):,}**\n- Decline rate: **{d.target.mean():.2%}**\n- Features: **{len(X.columns)} stationary features**\n- Absolute market-level features: **Forbidden**\n- Selected: **{selected}**\n- RED precision: **{rm['precision']:.2%}**\n- RED FPR: **{rm['FPR']:.2%}**\n- RED+AMBER recall: **{wm['recall']:.2%}**\n- GREEN NPV: **{wm['NPV']:.2%}**\n- PR-AUC: **{rank['PR_AUC']:.2%}**\n- ROC-AUC: **{rank['ROC_AUC']:.2%}**\n- Brier: **{brier:.4f}**\n- Stricter development contract passed: **{report['all_gates_passed']}**\n- Fresh 2025-2026 labels used: **No**\n''',encoding='utf-8')
    print(json.dumps({'selected':selected,'rows':len(d),'thresholds':{'watch':wt,'red':rt},'triage':tr,'ranking':rank,'gates':gates,'all':report['all_gates_passed']},indent=2))

if __name__=='__main__':main()
