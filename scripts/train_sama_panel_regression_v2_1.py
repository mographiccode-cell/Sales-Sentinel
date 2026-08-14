from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_sama_panel_decline_v2_0 import prepare

SEED=42
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'models'/'sama_panel_v2_1'; REP=ROOT/'reports'/'sama_panel_v2_1'; OUT.mkdir(parents=True,exist_ok=True); REP.mkdir(parents=True,exist_ok=True)
DECLINE_RATIO=.90
VAL_WEEKS=13
TEST_WEEKS=13
PURGE_WEEKS=1

def specs():
    return {
        'Ridge':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=20.0))]),
        'RandomForest':RandomForestRegressor(n_estimators=700,max_depth=12,min_samples_leaf=3,max_features=.70,random_state=SEED,n_jobs=-1),
        'ExtraTrees':ExtraTreesRegressor(n_estimators=900,max_depth=14,min_samples_leaf=2,max_features=.75,random_state=SEED,n_jobs=-1),
        'HistGradientBoosting':HistGradientBoostingRegressor(learning_rate=.035,max_iter=400,max_leaf_nodes=24,min_samples_leaf=18,l2_regularization=2.0,random_state=SEED),
        'XGBoost':xgb.XGBRegressor(n_estimators=650,max_depth=5,learning_rate=.025,min_child_weight=8,subsample=.85,colsample_bytree=.80,reg_alpha=.15,reg_lambda=4.0,objective='reg:squarederror',random_state=SEED,n_jobs=-1),
    }

def reg_metrics(y,p):
    return {'MAE_ratio':float(mean_absolute_error(y,p)),'RMSE_ratio':float(mean_squared_error(y,p)**.5),'R2_ratio':float(r2_score(y,p))}

def cls_metrics(y_ratio,p_ratio,cutoff):
    y=(y_ratio<DECLINE_RATIO).astype(int); z=(p_ratio<cutoff).astype(int); risk=-p_ratio
    return {'Accuracy':float(accuracy_score(y,z)),'BalancedAccuracy':float(balanced_accuracy_score(y,z)),'Precision':float(precision_score(y,z,zero_division=0)),'Recall':float(recall_score(y,z,zero_division=0)),'F1':float(f1_score(y,z,zero_division=0)),'ROC_AUC':float(roc_auc_score(y,risk)),'ConfusionMatrix':confusion_matrix(y,z,labels=[0,1]).tolist()}

def score(r,c):
    return -.60*r['MAE_ratio']+.16*c['BalancedAccuracy']+.10*c['F1']+.07*c['Accuracy']+.07*c['ROC_AUC']

def fit_predict(spec,Xtr,ytr,X):
    q=clone(spec); q.fit(Xtr,np.log(np.clip(ytr,.05,5.0))); return q,np.exp(q.predict(X))

def choose(history,F,val_start,val_end):
    train=history[history.week_start<val_start-pd.Timedelta(days=7*PURGE_WEEKS)]
    val=history[(history.week_start>=val_start)&(history.week_start<val_end)]
    if len(train)<1000 or len(val)<150: raise RuntimeError(f'insufficient inner split {len(train)}/{len(val)}')
    out={}
    for name,spec in specs().items():
        q,p=fit_predict(spec,train[F],train.next_ratio.to_numpy(),val[F]); r=reg_metrics(val.next_ratio.to_numpy(),p); best=None
        # Forecast calibration cutoff is selected on validation; true business target remains next_ratio < 0.90.
        for cutoff in np.arange(.75,1.051,.005):
            c=cls_metrics(val.next_ratio.to_numpy(),p,float(cutoff)); penalty=max(0,.70-c['Recall']); candidate=(score(r,c)-.12*penalty,c['BalancedAccuracy'],c['F1'],c['Accuracy'],-r['MAE_ratio'],-abs(cutoff-DECLINE_RATIO),float(cutoff),r,c)
            if best is None or candidate[:6]>best[:6]: best=candidate
        out[name]={'cutoff':best[6],'regression':best[7],'classification':best[8],'score':best[0]}
    selected=max(out,key=lambda n:(out[n]['score'],out[n]['classification']['BalancedAccuracy'],out[n]['classification']['F1'],-out[n]['regression']['MAE_ratio']))
    return selected,float(out[selected]['cutoff']),out

def main():
    d,F=prepare()
    # Target definition is frozen from v2.0 development-only diagnosis: 10% decline vs trailing 4-week mean.
    d=d.copy(); d['target']=(d.next_ratio<DECLINE_RATIO).astype(int)
    starts=[pd.Timestamp('2024-01-07'),pd.Timestamp('2024-04-07'),pd.Timestamp('2024-07-07'),pd.Timestamp('2024-10-06'),pd.Timestamp('2025-01-05'),pd.Timestamp('2025-04-06')]
    all_y=[]; all_p=[]; all_z=[]; folds=[]; selections=[]
    for i,start in enumerate(starts,1):
        end=start+pd.Timedelta(weeks=TEST_WEEKS)
        test=d[(d.week_start>=start)&(d.week_start<end)]
        if len(test)<150: continue
        val_end=start-pd.Timedelta(weeks=PURGE_WEEKS); val_start=val_end-pd.Timedelta(weeks=VAL_WEEKS)
        history=d[d.week_start<start]
        name,cutoff,sel=choose(history,F,val_start,val_end); selections.append(name)
        fit=d[d.week_start<start-pd.Timedelta(weeks=PURGE_WEEKS)]
        q,p=fit_predict(specs()[name],fit[F],fit.next_ratio.to_numpy(),test[F]); y=test.next_ratio.to_numpy(); z=(p<cutoff).astype(int)
        all_y.extend(y.tolist()); all_p.extend(p.tolist()); all_z.extend(z.tolist())
        folds.append({'fold':i,'test_start':str(start.date()),'test_end':str((end-pd.Timedelta(days=1)).date()),'train_rows':int(len(fit)),'test_rows':int(len(test)),'selected_model':name,'forecast_cutoff':cutoff,'regression':reg_metrics(y,p),'classification':cls_metrics(y,p,cutoff),'selection':sel})
    y=np.asarray(all_y); p=np.asarray(all_p); z=np.asarray(all_z); true=(y<DECLINE_RATIO).astype(int); risk=-p
    agg_c={'Accuracy':float(accuracy_score(true,z)),'BalancedAccuracy':float(balanced_accuracy_score(true,z)),'Precision':float(precision_score(true,z,zero_division=0)),'Recall':float(recall_score(true,z,zero_division=0)),'F1':float(f1_score(true,z,zero_division=0)),'ROC_AUC':float(roc_auc_score(true,risk)),'ConfusionMatrix':confusion_matrix(true,z,labels=[0,1]).tolist()}
    agg_r=reg_metrics(y,p); pos=float(true.mean()); majority=max(pos,1-pos)
    gates={'accuracy_at_least_90pct':agg_c['Accuracy']>=.90,'balanced_accuracy_at_least_80pct':agg_c['BalancedAccuracy']>=.80,'recall_at_least_75pct':agg_c['Recall']>=.75,'f1_at_least_70pct':agg_c['F1']>=.70,'roc_auc_at_least_85pct':agg_c['ROC_AUC']>=.85,'beats_majority_accuracy':agg_c['Accuracy']>majority}
    # Deployment selection from latest 13-week validation, fit through last complete target week.
    latest=d.week_start.max()+pd.Timedelta(days=7); val_end=latest-pd.Timedelta(weeks=PURGE_WEEKS); val_start=val_end-pd.Timedelta(weeks=VAL_WEEKS); name,cutoff,selection=choose(d,F,val_start,val_end); fit=d[d.week_start<val_end]; model,_=fit_predict(specs()[name],fit[F],fit.next_ratio.to_numpy(),fit[F].iloc[:1])
    joblib.dump({'model':model,'features':F,'forecast_cutoff':cutoff,'true_decline_ratio':DECLINE_RATIO,'version':'SAMA-PANEL-REGRESSION-2.1'},OUT/'sama_panel_decline_regressor_v2_1.joblib')
    report={'version':'SAMA-PANEL-REGRESSION-2.1','source':'Official SAMA city totals + national sector totals','target':'next-week POS value / trailing 4-week mean; decline if true ratio <0.90','evaluation':'six expanding time-block backtests across 2024-2025; each fold model and cutoff selected from prior validation only','panel_rows':int(len(d)),'entities':int(d.entity_id.nunique()),'feature_count':len(F),'xgboost_version':xgb.__version__,'folds':folds,'aggregate_regression':agg_r,'aggregate_classification':agg_c,'positive_rate':pos,'majority_accuracy':majority,'model_choices':dict(Counter(selections)),'acceptance_gates':gates,'all_acceptance_gates_passed':bool(all(gates.values())),'deployment':{'model':name,'forecast_cutoff':cutoff,'latest_selection':selection}}
    (REP/'sama_panel_regression_v2_1_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); (OUT/'model_metadata_v2_1.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); (REP/'sama_panel_regression_v2_1_summary.md').write_text(f"# SAMA Panel Regression v2.1\n\n- Backtest rows: **{len(y):,}**\n- Positive rate: **{pos:.2%}**\n- Accuracy: **{agg_c['Accuracy']:.2%}**\n- Balanced Accuracy: **{agg_c['BalancedAccuracy']:.2%}**\n- Precision: **{agg_c['Precision']:.2%}**\n- Recall: **{agg_c['Recall']:.2%}**\n- F1: **{agg_c['F1']:.2%}**\n- ROC-AUC: **{agg_c['ROC_AUC']:.2%}**\n- Ratio MAE: **{agg_r['MAE_ratio']:.4f}**\n- Majority baseline: **{majority:.2%}**\n- Model choices: **{dict(Counter(selections))}**\n- Deployment: **{name}**, cutoff **{cutoff:.3f}**\n- All acceptance gates passed: **{all(gates.values())}**\n",encoding='utf-8'); print(json.dumps({'regression':agg_r,'classification':agg_c,'positive_rate':pos,'majority':majority,'choices':dict(Counter(selections)),'gates':gates,'all_passed':all(gates.values()),'deployment':name,'cutoff':cutoff},indent=2))
if __name__=='__main__': main()
