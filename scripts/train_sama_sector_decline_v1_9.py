from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[1]
HIST=ROOT/'data'/'sama_pos'/'sama_sector_weekly_value_count_2020_2025.csv'
FCST=ROOT/'data'/'sama_pos'/'sama_sector_walkforward_forecasts_2023_2025.csv'
OUT=ROOT/'reports'/'sama_sector_decline_v1_9'; MOD=ROOT/'models'/'sama_sector_decline_v1_9'
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'report.json'; SUMMARY=OUT/'summary.md'; MODEL=MOD/'market_decline_classifier_v1_9.joblib'
DECLINE=.20


def met(y,score,t):
    y=np.asarray(y,dtype=int); pred=(np.asarray(score)>=t).astype(int)
    return {'Accuracy':float(accuracy_score(y,pred)),'BalancedAccuracy':float(balanced_accuracy_score(y,pred)),'Precision':float(precision_score(y,pred,zero_division=0)),'Recall':float(recall_score(y,pred,zero_division=0)),'F1':float(f1_score(y,pred,zero_division=0)),'ROC_AUC':float(roc_auc_score(y,score))}


def threshold_accuracy_constrained(y,score,probability=True):
    score=np.asarray(score,float)
    thresholds=np.linspace(.02,.98,193) if probability else np.linspace(np.nanpercentile(score,1),np.nanpercentile(score,99),193)
    eligible=[]; fallback=[]
    for t in thresholds:
        m=met(y,score,float(t)); c=(m['Accuracy'],m['F1'],m['BalancedAccuracy'],m['Precision'],m['ROC_AUC'],-abs(float(t)-.5),float(t),m)
        fallback.append(c)
        if m['Recall']>=.70 and m['BalancedAccuracy']>=.70: eligible.append(c)
    best=max(eligible if eligible else fallback,key=lambda z:z[:6])
    return best[6],best[7],bool(eligible)


def prepare():
    h=pd.read_csv(HIST,parse_dates=['week_start']).sort_values(['sector','week_start']).reset_index(drop=True)
    f=pd.read_csv(FCST,parse_dates=['origin_week_start']).sort_values(['sector','origin_week_start']).reset_index(drop=True)
    g=h.groupby('sector',group_keys=False)
    for w in (4,8,13):
        h[f'value_mean{w}']=g.value_thousand_sar.transform(lambda s,w=w:s.rolling(w,min_periods=w).mean())
        h[f'value_std{w}']=g.value_thousand_sar.transform(lambda s,w=w:s.rolling(w,min_periods=w).std())
        h[f'count_mean{w}']=g.transaction_count_thousand.transform(lambda s,w=w:s.rolling(w,min_periods=w).mean())
    h['value_change1']=g.value_thousand_sar.pct_change(); h['value_change4']=h.value_thousand_sar/g.value_thousand_sar.shift(4)-1
    h['count_change1']=g.transaction_count_thousand.pct_change(); h['count_change4']=h.transaction_count_thousand/g.transaction_count_thousand.shift(4)-1
    cols=['sector','week_start','value_thousand_sar','transaction_count_thousand','value_change1','value_change4','count_change1','count_change4']+[f'{x}_{y}{w}' for w in (4,8,13) for x,y in [('value','mean'),('value','std'),('count','mean')]]
    # correct generated names to actual columns
    cols=['sector','week_start','value_thousand_sar','transaction_count_thousand','value_change1','value_change4','count_change1','count_change4','value_mean4','value_std4','count_mean4','value_mean8','value_std8','count_mean8','value_mean13','value_std13','count_mean13']
    b=h[cols].rename(columns={'week_start':'origin_week_start','value_thousand_sar':'origin_value','transaction_count_thousand':'origin_count'})
    d=f.merge(b,on=['sector','origin_week_start'],how='inner',validate='one_to_one')
    nxt=h[['sector','week_start','value_thousand_sar','transaction_count_thousand']].copy(); nxt['origin_week_start']=nxt.week_start-pd.Timedelta(days=7)
    nxt=nxt.rename(columns={'value_thousand_sar':'actual_next_value','transaction_count_thousand':'actual_next_count'})[['sector','origin_week_start','actual_next_value','actual_next_count']]
    d=d.merge(nxt,on=['sector','origin_week_start'],how='left',validate='one_to_one')
    d['actual_ratio']=d.actual_next_value/d.value_mean4; d['target']=(d.actual_ratio<1-DECLINE).astype(int)
    d['pred_value_ratio4']=d.predicted_value_h1/d.value_mean4; d['pred_value_ratio8']=d.predicted_value_h1/d.value_mean8; d['pred_value_ratio13']=d.predicted_value_h1/d.value_mean13
    d['pred_count_ratio4']=d.predicted_count_h1/d.count_mean4; d['pred_count_ratio8']=d.predicted_count_h1/d.count_mean8; d['pred_count_ratio13']=d.predicted_count_h1/d.count_mean13
    d['origin_value_ratio4']=d.origin_value/d.value_mean4; d['origin_count_ratio4']=d.origin_count/d.count_mean4
    d['value_cv4']=d.value_std4/d.value_mean4; d['value_cv8']=d.value_std8/d.value_mean8; d['value_cv13']=d.value_std13/d.value_mean13
    d['forecast_value_count_gap']=d.predicted_value_h1_change_vs_last-d.predicted_count_h1_change_vs_last
    # Historical forecast-error reliability is shifted, so current/future actual is never used as a feature.
    d['past_value_ape']=np.abs(d.actual_value_h1-d.predicted_value_h1)/d.actual_value_h1.replace(0,np.nan)
    d['past_count_ape']=np.abs(d.actual_count_h1-d.predicted_count_h1)/d.actual_count_h1.replace(0,np.nan)
    fg=d.groupby('sector',group_keys=False)
    d['forecast_value_ape8_known']=fg.past_value_ape.transform(lambda s:s.shift(1).rolling(8,min_periods=4).mean())
    d['forecast_count_ape8_known']=fg.past_count_ape.transform(lambda s:s.shift(1).rolling(8,min_periods=4).mean())
    week=d.origin_week_start.dt.isocalendar().week.astype(float); d['week_sin']=np.sin(2*np.pi*week/52.18); d['week_cos']=np.cos(2*np.pi*week/52.18)
    num=['pred_value_ratio4','pred_value_ratio8','pred_value_ratio13','pred_count_ratio4','pred_count_ratio8','pred_count_ratio13','origin_value_ratio4','origin_count_ratio4','predicted_value_h1_change_vs_last','predicted_count_h1_change_vs_last','predicted_value_h2_change_vs_last','predicted_count_h2_change_vs_last','value_change1','value_change4','count_change1','count_change4','value_cv4','value_cv8','value_cv13','forecast_value_count_gap','forecast_value_ape8_known','forecast_count_ape8_known','week_sin','week_cos']
    cats=pd.get_dummies(d.sector,prefix='sector',dtype=float); X=pd.concat([d[num].reset_index(drop=True),cats.reset_index(drop=True)],axis=1)
    good=d.actual_next_value.notna()&d.value_mean4.gt(0)&X.replace([np.inf,-np.inf],np.nan).notna().all(axis=1)
    return d.loc[good].reset_index(drop=True),X.loc[good].reset_index(drop=True),num


def main():
    d,X,num=prepare()
    tr=d.origin_week_start<=pd.Timestamp('2023-12-31'); va=d.origin_week_start.between(pd.Timestamp('2024-01-01'),pd.Timestamp('2024-04-30')); te=d.origin_week_start>=pd.Timestamp('2024-05-01')
    train,val,test=d[tr],d[va],d[te]; Xtr,Xv,Xt=X[tr],X[va],X[te]
    if min(len(train),len(val),len(test))<150: raise RuntimeError(f'splits {len(train)}/{len(val)}/{len(test)}')
    pos=int(train.target.sum()); neg=len(train)-pos; spw=neg/max(pos,1)
    models={
      'LogisticRegression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',C=.6,random_state=42)),
      'ExtraTrees':ExtraTreesClassifier(n_estimators=1200,max_depth=10,min_samples_leaf=3,max_features=.75,class_weight='balanced',random_state=42,n_jobs=-1),
      'HistGradientBoosting':HistGradientBoostingClassifier(max_iter=350,learning_rate=.035,max_leaf_nodes=18,min_samples_leaf=18,l2_regularization=3.0,random_state=42),
      'XGBoost':XGBClassifier(n_estimators=500,max_depth=3,learning_rate=.03,subsample=.85,colsample_bytree=.8,min_child_weight=6,reg_lambda=4,reg_alpha=.3,scale_pos_weight=spw,eval_metric='logloss',random_state=42,n_jobs=-1),
    }
    vr={}
    for name,m in models.items():
        fit=clone(m).fit(Xtr,train.target); pv=fit.predict_proba(Xv)[:,1]; t,mm,constrained=threshold_accuracy_constrained(val.target,pv,True); vr[name]={'threshold':t,'metrics':mm,'recall_constraint_satisfied':constrained}
    # raw SAMA forecast score is also a candidate and may be better calibrated than a classifier.
    raw_v=1.0-val.pred_value_ratio4.to_numpy(); rt,rm,rcon=threshold_accuracy_constrained(val.target,raw_v,False); vr['ForecastRuleCalibrated']={'threshold':rt,'metrics':rm,'recall_constraint_satisfied':rcon}
    best=max(vr,key=lambda n:(vr[n]['recall_constraint_satisfied'],vr[n]['metrics']['Accuracy'],vr[n]['metrics']['F1'],vr[n]['metrics']['BalancedAccuracy']))
    trainval=tr|va
    if best=='ForecastRuleCalibrated':
        threshold=rt; score=1.0-test.pred_value_ratio4.to_numpy(); final=None
    else:
        threshold=vr[best]['threshold']; final=clone(models[best]).fit(X[trainval],d.loc[trainval,'target']); score=final.predict_proba(Xt)[:,1]
    tm=met(test.target,score,threshold); majority=max(float(test.target.mean()),1-float(test.target.mean()))
    gates={'accuracy_90':tm['Accuracy']>=.90,'beats_majority':tm['Accuracy']>majority,'balanced_accuracy_75':tm['BalancedAccuracy']>=.75,'recall_70':tm['Recall']>=.70,'f1_60':tm['F1']>=.60,'roc_auc_85':tm['ROC_AUC']>=.85}
    out={'version':'SAMA-SECTOR-DECLINE-1.9','target':'next official sector week value >=20% below trailing 4 official weeks','split':{'train':len(train),'validation':len(val),'test':len(test),'shuffle':False},'positive_rates':{'train':float(train.target.mean()),'validation':float(val.target.mean()),'test':float(test.target.mean())},'validation_candidates':vr,'selected':best,'threshold':float(threshold),'test_metrics':tm,'majority_test_accuracy':majority,'gates':gates,'all_gates_passed':bool(all(gates.values())),'leakage_controls':{'future_actual_SAMA_features':False,'forecast_error_features_shifted_before_use':True,'walkforward_SAMA_forecasts_only':True,'test_not_used_for_selection':True}}
    REPORT.write_text(json.dumps(out,indent=2),encoding='utf-8')
    SUMMARY.write_text(f"""# SAMA Sector Decline v1.9\n\n- Selected: **{best}**\n- Train/Validation/Test: **{len(train)}/{len(val)}/{len(test)}**\n- Test decline rate: **{test.target.mean():.2%}**\n- Accuracy: **{tm['Accuracy']:.2%}**\n- Balanced Accuracy: **{tm['BalancedAccuracy']:.2%}**\n- Precision: **{tm['Precision']:.2%}**\n- Recall: **{tm['Recall']:.2%}**\n- F1: **{tm['F1']:.2%}**\n- ROC-AUC: **{tm['ROC_AUC']:.2%}**\n- Majority baseline: **{majority:.2%}**\n- All gates: **{out['all_gates_passed']}**\n""",encoding='utf-8')
    joblib.dump({'model':final,'selected':best,'threshold':threshold,'features':list(X.columns),'version':out['version'],'decline_threshold':DECLINE},MODEL)
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
