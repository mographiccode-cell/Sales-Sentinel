from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier=None

SEED=42
ROOT=Path(__file__).resolve().parents[1]
HIST=ROOT/'data'/'sama_pos'/'sama_sector_weekly_value_count_2020_2025.csv'
FCST=ROOT/'data'/'sama_pos'/'sama_sector_walkforward_forecasts_2023_2025.csv'
OUT=ROOT/'models'/'sama_market_v1_9'; REP=ROOT/'reports'/'sama_market_v1_9'
OUT.mkdir(parents=True,exist_ok=True); REP.mkdir(parents=True,exist_ok=True)
MODEL=OUT/'sama_market_decline_classifier_v1_9.joblib'; REPORT=REP/'sama_market_decline_v1_9.json'; SUMMARY=REP/'sama_market_decline_v1_9.md'
DECLINE=.20


def metr(y,p,t):
    z=(np.asarray(p)>=t).astype(int); y=np.asarray(y).astype(int)
    return {'Accuracy':float(accuracy_score(y,z)),'BalancedAccuracy':float(balanced_accuracy_score(y,z)),'Precision':float(precision_score(y,z,zero_division=0)),'Recall':float(recall_score(y,z,zero_division=0)),'F1':float(f1_score(y,z,zero_division=0)),'ROC_AUC':float(roc_auc_score(y,p)),'ConfusionMatrix':confusion_matrix(y,z,labels=[0,1]).tolist()}

def models(pos_weight):
    d={
      'LogisticRegression':Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.4,max_iter=4000,class_weight='balanced',random_state=SEED))]),
      'RandomForest':RandomForestClassifier(n_estimators=700,max_depth=9,min_samples_leaf=8,max_features=.65,class_weight='balanced_subsample',random_state=SEED,n_jobs=-1),
      'ExtraTrees':ExtraTreesClassifier(n_estimators=900,max_depth=11,min_samples_leaf=6,max_features=.70,class_weight='balanced',random_state=SEED,n_jobs=-1),
      'HistGradientBoosting':HistGradientBoostingClassifier(max_iter=300,learning_rate=.035,max_leaf_nodes=18,min_samples_leaf=25,l2_regularization=2,class_weight='balanced',random_state=SEED),
    }
    if XGBClassifier is not None:
      d['XGBoost']=XGBClassifier(n_estimators=500,max_depth=4,learning_rate=.03,min_child_weight=10,subsample=.85,colsample_bytree=.8,reg_lambda=3,reg_alpha=.2,scale_pos_weight=pos_weight,eval_metric='logloss',random_state=SEED,n_jobs=-1)
    return d

def prepare():
    h=pd.read_csv(HIST,parse_dates=['week_start']).sort_values(['sector','week_start']).reset_index(drop=True)
    g=h.groupby('sector',sort=False)
    h['baseline4']=g.value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).mean())
    h['baseline8']=g.value_thousand_sar.transform(lambda s:s.rolling(8,min_periods=8).mean())
    h['baseline13']=g.value_thousand_sar.transform(lambda s:s.rolling(13,min_periods=13).mean())
    h['value_std4']=g.value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).std())
    h['value_std13']=g.value_thousand_sar.transform(lambda s:s.rolling(13,min_periods=13).std())
    h['count_std4']=g.transaction_count_thousand.transform(lambda s:s.rolling(4,min_periods=4).std())
    h['value_change1']=g.value_thousand_sar.pct_change(1); h['value_change4']=g.value_thousand_sar.pct_change(4)
    h['count_change1']=g.transaction_count_thousand.pct_change(1); h['count_change4']=g.transaction_count_thousand.pct_change(4)
    known=h[['week_start','sector','value_thousand_sar','transaction_count_thousand','baseline4','baseline8','baseline13','value_std4','value_std13','count_std4','value_change1','value_change4','count_change1','count_change4']].rename(columns={'week_start':'origin_week_start','value_thousand_sar':'origin_value','transaction_count_thousand':'origin_count'})
    f=pd.read_csv(FCST,parse_dates=['origin_week_start'])
    d=f.merge(known,on=['origin_week_start','sector'],how='left',validate='many_to_one')
    d=d.dropna(subset=['baseline4','baseline8','baseline13','predicted_value_h1','predicted_value_h2','predicted_count_h1','predicted_count_h2','actual_value_h1']).copy()
    d['target']=(d.actual_value_h1 < (1-DECLINE)*d.baseline4).astype(int)
    # All features are available at origin; actual future values are targets/audit only.
    x=pd.DataFrame(index=d.index)
    x['forecast_value_h1_ratio_baseline4']=d.predicted_value_h1/d.baseline4
    x['forecast_value_h2_ratio_baseline4']=d.predicted_value_h2/d.baseline4
    x['forecast_value_h1_ratio_origin']=d.predicted_value_h1/d.origin_value
    x['forecast_value_h2_ratio_origin']=d.predicted_value_h2/d.origin_value
    x['forecast_count_h1_ratio_origin']=d.predicted_count_h1/d.origin_count
    x['forecast_count_h2_ratio_origin']=d.predicted_count_h2/d.origin_count
    for c in ['predicted_value_h1_index_52median','predicted_value_h2_index_52median','predicted_count_h1_index_52median','predicted_count_h2_index_52median','predicted_value_h1_change_vs_last','predicted_value_h2_change_vs_last','predicted_count_h1_change_vs_last','predicted_count_h2_change_vs_last']:
        x[c]=d[c].to_numpy()
    x['origin_value_ratio_base4']=d.origin_value/d.baseline4
    x['origin_value_ratio_base8']=d.origin_value/d.baseline8
    x['origin_value_ratio_base13']=d.origin_value/d.baseline13
    x['value_cv4']=d.value_std4/d.baseline4
    x['value_cv13']=d.value_std13/d.baseline13
    x['count_cv4']=d.count_std4/d.origin_count
    for c in ['value_change1','value_change4','count_change1','count_change4']: x[c]=d[c].to_numpy()
    week=d.origin_week_start.dt.isocalendar().week.astype(float); x['week_sin']=np.sin(2*np.pi*week/52.18); x['week_cos']=np.cos(2*np.pi*week/52.18)
    cats=pd.get_dummies(d.sector,prefix='sector',dtype=float); x=pd.concat([x,cats],axis=1).replace([np.inf,-np.inf],np.nan)
    good=x.notna().all(axis=1); d=d.loc[good].reset_index(drop=True); x=x.loc[good].reset_index(drop=True)
    return d,x,list(x.columns)

def pick_threshold(y,p):
    candidates=[]
    for t in np.linspace(.03,.90,175):
        m=metr(y,p,float(t))
        # Optimize detection quality while preserving meaningful accuracy on the validation year-half.
        score=.30*m['F1']+.25*m['BalancedAccuracy']+.20*m['Recall']+.15*m['ROC_AUC']+.10*m['Accuracy']
        if m['Accuracy']<.80: score-=2*(.80-m['Accuracy'])
        candidates.append((score,m['F1'],m['BalancedAccuracy'],m['Recall'],m['Accuracy'],-abs(t-.5),float(t),m))
    b=max(candidates); return b[6],b[7],b[0]

def main():
    d,X,F=prepare()
    train_mask=(d.origin_week_start>='2023-01-01')&(d.origin_week_start<'2023-07-01')
    val_mask=(d.origin_week_start>='2023-07-01')&(d.origin_week_start<'2024-01-01')
    test_mask=d.origin_week_start>='2024-01-01'
    tr,va,te=d[train_mask],d[val_mask],d[test_mask]; Xt,Xv,Xe=X[train_mask],X[val_mask],X[test_mask]
    if min(len(tr),len(va),len(te))<300: raise RuntimeError(f'insufficient splits {len(tr)}/{len(va)}/{len(te)}')
    if min(tr.target.sum(),va.target.sum(),te.target.sum())<20: raise RuntimeError('too few positive decline cases')
    pos=int(tr.target.sum()); neg=len(tr)-pos; candidates={}; fitted={}
    for name,model in models(neg/max(pos,1)).items():
        fit=clone(model).fit(Xt,tr.target); p=fit.predict_proba(Xv)[:,1]; t,m,s=pick_threshold(va.target,p); candidates[name]={'threshold':t,'validation_metrics':m,'selection_score':s}; fitted[name]=fit
    selected=max(candidates,key=lambda n:(candidates[n]['selection_score'],candidates[n]['validation_metrics']['ROC_AUC'])); threshold=float(candidates[selected]['threshold'])
    fit_d=pd.concat([tr,va],ignore_index=True); fit_x=pd.concat([Xt,Xv],ignore_index=True); fit_pos=int(fit_d.target.sum()); fit_neg=len(fit_d)-fit_pos
    model=clone(models(fit_neg/max(fit_pos,1))[selected]).fit(fit_x,fit_d.target); prob=model.predict_proba(Xe)[:,1]; tm=metr(te.target,prob,threshold); majority=max(float(te.target.mean()),1-float(te.target.mean()))
    gates={'accuracy_at_least_90pct':tm['Accuracy']>=.90,'balanced_accuracy_at_least_75pct':tm['BalancedAccuracy']>=.75,'recall_at_least_60pct':tm['Recall']>=.60,'f1_at_least_60pct':tm['F1']>=.60,'roc_auc_at_least_85pct':tm['ROC_AUC']>=.85,'beats_majority':tm['Accuracy']>majority}
    out={'version':'SAMA-MARKET-DECLINE-1.9','source':'Official SAMA sector POS history + leakage-safe sector forecasts','target':'next official sector week >=20% below trailing four-week known baseline','features':F,'split':{'train':'2023-01-01..2023-06-30','validation':'2023-07-01..2023-12-31','test':'2024-01-01 onward','train_rows':len(tr),'validation_rows':len(va),'test_rows':len(te),'train_positive_rate':float(tr.target.mean()),'validation_positive_rate':float(va.target.mean()),'test_positive_rate':float(te.target.mean())},'candidate_models':candidates,'selected_model':selected,'selected_probability_threshold':threshold,'test_metrics':tm,'majority_test_accuracy':majority,'acceptance_gates':gates,'all_gates_passed':bool(all(gates.values())),'leakage_controls':{'target_fixed_20pct_before_training':True,'future_actual_SAMA_not_feature':True,'forecast_features_walk_forward_only':True,'model_and_threshold_selected_without_2024_2025':True,'test_2024_2025_opened_after_selection':True}}
    joblib.dump({'model':model,'features':F,'threshold':threshold,'decline_threshold':DECLINE,'version':out['version']},MODEL); REPORT.write_text(json.dumps(out,indent=2),encoding='utf-8')
    SUMMARY.write_text(f'''# SAMA Market Decline v1.9

- Train rows: **{len(tr):,}**
- Validation rows: **{len(va):,}**
- Untouched 2024-2025 test rows: **{len(te):,}**
- Test positive rate: **{te.target.mean():.2%}**
- Selected model: **{selected}**
- Probability threshold: **{threshold:.3f}**
- Accuracy: **{tm['Accuracy']:.2%}**
- Balanced Accuracy: **{tm['BalancedAccuracy']:.2%}**
- Precision: **{tm['Precision']:.2%}**
- Recall: **{tm['Recall']:.2%}**
- F1: **{tm['F1']:.2%}**
- ROC-AUC: **{tm['ROC_AUC']:.2%}**
- Majority baseline: **{majority:.2%}**
- All scientific gates passed: **{out['all_gates_passed']}**
''',encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
