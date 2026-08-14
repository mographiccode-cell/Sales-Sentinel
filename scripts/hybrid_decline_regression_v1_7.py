from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED=42
ROOT=Path(__file__).resolve().parents[1]
DAILY=ROOT/'data'/'saudi_v1_3'/'saudi_daily_sama_calibrated_v1_3.csv'
SAMA_FC=ROOT/'data'/'sama_pos'/'sama_market_walkforward_forecasts_2023_2025.csv'
SAMA_REPORT=ROOT/'reports'/'sama_market_v1_6'/'sama_market_forecaster_report_v1_6.json'
MODEL_DIR=ROOT/'models'/'saudi_v1_7'; REPORT_DIR=ROOT/'reports'/'saudi_v1_7'
MODEL_DIR.mkdir(parents=True,exist_ok=True); REPORT_DIR.mkdir(parents=True,exist_ok=True)
H=7; BASELINE=28; DECLINE_RATIO=.90; PURGE=7; VAL=60; BLOCK=45; MIN_TRAIN=220; MIN_HISTORY=56

RAMADAN=[('2023-03-23','2023-04-20'),('2024-03-11','2024-04-09')]
EID_FITR=[('2023-04-21','2023-04-23'),('2024-04-10','2024-04-12')]
HAJJ=[('2023-06-19','2023-06-30'),('2024-06-07','2024-06-19')]
EID_ADHA=[('2023-06-28','2023-07-01'),('2024-06-16','2024-06-19')]

def ratio(a,b): return a/b.replace(0,np.nan)
def in_ranges(dt,r): return int(any(pd.Timestamp(a)<=dt<=pd.Timestamp(b) for a,b in r))
def wkstart(dt):
    dt=pd.Timestamp(dt); return dt-pd.Timedelta(days=(dt.dayofweek+1)%7)

def load_sama():
    fc=pd.read_csv(SAMA_FC,parse_dates=['origin_week_start','forecast_h1_week_start','forecast_h2_week_start'])
    return fc.set_index('origin_week_start').to_dict('index')

def market_features(dates,lookup):
    rows=[]
    for dt in dates:
        cur=wkstart(dt); last=cur-pd.Timedelta(days=7); r=lookup.get(last)
        if r is None:
            rows.append([np.nan]*10); continue
        v1=float(r['predicted_value_h1_index_52median']); v2=float(r['predicted_value_h2_index_52median'])
        c1=float(r['predicted_count_h1_index_52median']); c2=float(r['predicted_count_h2_index_52median'])
        wv=[]; wc=[]
        for i in range(1,H+1):
            w=wkstart(pd.Timestamp(dt)+pd.Timedelta(days=i))
            if w==pd.Timestamp(r['forecast_h1_week_start']): wv.append(v1); wc.append(c1)
            else: wv.append(v2); wc.append(c2)
        rows.append([v1,v2,c1,c2,float(r['predicted_value_h1_change_vs_last']),float(r['predicted_value_h2_change_vs_last']),float(r['predicted_count_h1_change_vs_last']),float(r['predicted_count_h2_change_vs_last']),float(np.mean(wv)),float(np.mean(wc))])
    return np.asarray(rows)

def build(d,lookup):
    d=d.sort_values('date').reset_index(drop=True).copy(); d['date']=pd.to_datetime(d['date'])
    cal=d['sama_calibrated_net_sales_sar'].astype(float); base=d['base_net_sales_sar_unscaled'].astype(float)
    future_cal=pd.concat([cal.shift(-i) for i in range(1,H+1)],axis=1).mean(axis=1)
    future_base=pd.concat([base.shift(-i) for i in range(1,H+1)],axis=1).mean(axis=1)
    baseline_cal=cal.rolling(BASELINE).mean(); baseline_base=base.rolling(BASELINE).mean()
    x=pd.DataFrame({'date':d['date']})
    series={
        'cal_sales':cal,'base_sales':base,
        'customers':d['unique_observed_customers'].astype(float),'invoices':d['invoice_count'].astype(float),
        'transactions':d['transaction_rows'].astype(float),'avg_invoice':d['average_invoice_value_sar'].astype(float),
        'return_rate':d['return_rate_value'].astype(float),'returning':d['returning_observed_customers'].astype(float),
        'new_customers':d['new_observed_customers'].astype(float)
    }
    for name,s in series.items():
        x[f'{name}_t0']=s
        for lag in (1,2,3,7,14,28,56): x[f'{name}_lag_{lag}']=s.shift(lag)
        for w in (7,14,28,56):
            x[f'{name}_mean_{w}']=s.rolling(w).mean(); x[f'{name}_std_{w}']=s.rolling(w).std()
    x['cal_vs_mean28']=ratio(x['cal_sales_t0'],x['cal_sales_mean_28']); x['base_vs_mean28']=ratio(x['base_sales_t0'],x['base_sales_mean_28'])
    x['cal_mean7_vs_28']=ratio(x['cal_sales_mean_7'],x['cal_sales_mean_28']); x['base_mean7_vs_28']=ratio(x['base_sales_mean_7'],x['base_sales_mean_28'])
    x['calibration_ratio_t0']=ratio(x['cal_sales_t0'],x['base_sales_t0']); x['calibration_ratio_mean7']=ratio(x['cal_sales_mean_7'],x['base_sales_mean_7']); x['calibration_ratio_mean28']=ratio(x['cal_sales_mean_28'],x['base_sales_mean_28'])
    x['customers_vs_mean28']=ratio(x['customers_t0'],x['customers_mean_28']); x['invoices_vs_mean28']=ratio(x['invoices_t0'],x['invoices_mean_28']); x['transactions_vs_mean28']=ratio(x['transactions_t0'],x['transactions_mean_28'])
    mf=market_features(d['date'],lookup)
    names=['sama_pred_value_h1_idx','sama_pred_value_h2_idx','sama_pred_count_h1_idx','sama_pred_count_h2_idx','sama_pred_value_h1_change','sama_pred_value_h2_change','sama_pred_count_h1_change','sama_pred_count_h2_change','sama_pred_next7_value_idx','sama_pred_next7_count_idx']
    for i,n in enumerate(names): x[n]=mf[:,i]
    actual_market=d['sama_weekly_market_index'].astype(float)
    for lag in (7,14,28): x[f'actual_sama_lag_{lag}']=actual_market.shift(lag)
    x['predicted_market_vs_actual_lag7']=ratio(x['sama_pred_next7_value_idx'],x['actual_sama_lag_7'])
    doy=d['date'].dt.dayofyear.astype(float); x['doy_sin']=np.sin(2*np.pi*doy/365.25); x['doy_cos']=np.cos(2*np.pi*doy/365.25)
    future_dates=[[dt+pd.Timedelta(days=i) for i in range(1,H+1)] for dt in d['date']]
    x['next7_ramadan']=[sum(in_ranges(z,RAMADAN) for z in ds) for ds in future_dates]; x['next7_fitr']=[sum(in_ranges(z,EID_FITR) for z in ds) for ds in future_dates]; x['next7_hajj']=[sum(in_ranges(z,HAJJ) for z in ds) for ds in future_dates]; x['next7_adha']=[sum(in_ranges(z,EID_ADHA) for z in ds) for ds in future_dates]
    x['baseline_cal']=baseline_cal; x['baseline_base']=baseline_base
    x['target_cal_ratio']=ratio(future_cal,baseline_cal); x['target_base_ratio']=ratio(future_base,baseline_base)
    x.loc[len(x)-H:,["target_cal_ratio","target_base_ratio"]]=np.nan
    x=x.iloc[MIN_HISTORY:].replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)
    features=[c for c in x if c not in {'date','target_cal_ratio','target_base_ratio'}]
    return x,features

def specs():
    return {
        'Ridge':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=20.0))]),
        'RandomForest':RandomForestRegressor(n_estimators=700,max_depth=8,min_samples_leaf=4,max_features=.7,random_state=SEED,n_jobs=-1),
        'ExtraTrees':ExtraTreesRegressor(n_estimators=900,max_depth=10,min_samples_leaf=3,max_features=.75,random_state=SEED,n_jobs=-1),
        'HistGradientBoosting':HistGradientBoostingRegressor(learning_rate=.035,max_iter=350,max_leaf_nodes=15,min_samples_leaf=12,l2_regularization=2.0,random_state=SEED)
    }
def classmetrics(y_ratio,pred_ratio):
    y=(y_ratio<DECLINE_RATIO).astype(int); p=(pred_ratio<DECLINE_RATIO).astype(int); risk=-pred_ratio
    return {'Accuracy':float(accuracy_score(y,p)),'BalancedAccuracy':float(balanced_accuracy_score(y,p)),'Precision':float(precision_score(y,p,zero_division=0)),'Recall':float(recall_score(y,p,zero_division=0)),'F1':float(f1_score(y,p,zero_division=0)),'ROC_AUC':float(roc_auc_score(y,risk)),'ConfusionMatrix':confusion_matrix(y,p,labels=[0,1]).tolist()}
def regmetrics(y,p): return {'MAE_ratio':float(mean_absolute_error(y,p)),'RMSE_ratio':float(mean_squared_error(y,p)**.5),'R2_ratio':float(r2_score(y,p))}

def select(train,features):
    vs=len(train)-VAL; ie=vs-PURGE
    if ie<MIN_TRAIN: raise RuntimeError('inner history too short')
    tr=train.iloc[:ie]; va=train.iloc[vs:]
    res={}
    for n,s in specs().items():
        m=clone(s).fit(tr[features],tr['target_cal_ratio']); p=m.predict(va[features]); rm=regmetrics(va['target_cal_ratio'],p); cm=classmetrics(va['target_cal_ratio'].to_numpy(),p)
        # Continuous forecast quality primary; classification metrics secondary, fixed business threshold.
        score=-rm['MAE_ratio']+.08*cm['BalancedAccuracy']+.05*cm['F1']
        res[n]={'score':score,'regression':rm,'classification':cm}
    n=max(res,key=lambda k:res[k]['score']); return n,res

def main():
    lookup=load_sama(); sama_report=json.loads(SAMA_REPORT.read_text(encoding='utf-8')); d=pd.read_csv(DAILY,parse_dates=['date']); frame,features=build(d,lookup)
    first=max(MIN_TRAIN+VAL+2*PURGE,260); starts=list(range(first,len(frame),BLOCK))
    ys=[]; ps=[]; folds=[]; choices=[]
    for num,start in enumerate(starts,1):
        end=min(start+BLOCK,len(frame));
        if end-start<20: continue
        hist=frame.iloc[:start-PURGE]; test=frame.iloc[start:end]; name,selection=select(hist,features); choices.append(name)
        model=clone(specs()[name]).fit(hist[features],hist['target_cal_ratio']); pred=model.predict(test[features]); y=test['target_cal_ratio'].to_numpy(); ys.extend(y.tolist()); ps.extend(pred.tolist())
        folds.append({'fold':num,'train_end':str(hist['date'].iloc[-1].date()),'test_start':str(test['date'].iloc[0].date()),'test_end':str(test['date'].iloc[-1].date()),'model':name,'regression':regmetrics(y,pred),'classification':classmetrics(y,pred),'selection':selection})
    y=np.asarray(ys); p=np.asarray(ps); agg_r=regmetrics(y,p); agg_c=classmetrics(y,p); positive=float((y<DECLINE_RATIO).mean()); majority=max(positive,1-positive)
    gates={'accuracy_at_least_90pct':agg_c['Accuracy']>=.90,'balanced_accuracy_at_least_80pct':agg_c['BalancedAccuracy']>=.80,'recall_at_least_75pct':agg_c['Recall']>=.75,'f1_at_least_70pct':agg_c['F1']>=.70,'roc_auc_at_least_85pct':agg_c['ROC_AUC']>=.85,'beats_majority_accuracy':agg_c['Accuracy']>majority}
    dep_hist=frame.iloc[:-PURGE]; dep_name,_=select(dep_hist,features); dep=clone(specs()[dep_name]).fit(dep_hist[features],dep_hist['target_cal_ratio'])
    joblib.dump({'model':dep,'features':features,'decline_ratio':DECLINE_RATIO,'horizon_days':H,'baseline_days':BASELINE,'version':'SALES-DECLINE-1.7-CONTINUOUS-HYBRID'},MODEL_DIR/'sales_decline_hybrid_regressor_v1_7.joblib')
    report={'version':'SALES-DECLINE-1.7-CONTINUOUS-HYBRID','dataset':'SA-LOCALIZATION-1.3.1-SAMA-SAFE','method':'continuous future 7-day calibrated-sales ratio regression; fixed decline boundary 0.90; SAMA future signal comes only from leakage-safe SAMA forecaster','evaluation':'nested walk-forward backtest','evaluation_rows':int(len(y)),'positive_rate':positive,'majority_accuracy':majority,'aggregate_regression':agg_r,'aggregate_classification':agg_c,'folds':folds,'model_choices':dict(Counter(choices)),'deployment_model':dep_name,'acceptance_gates':gates,'all_acceptance_gates_passed':bool(all(gates.values())),'sama_forecaster_metrics':sama_report['metrics']}
    (REPORT_DIR/'hybrid_decline_v1_7_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); (MODEL_DIR/'model_metadata_v1_7.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); (REPORT_DIR/'hybrid_decline_v1_7_summary.md').write_text(f"# Hybrid Decline v1.7\n\n- Evaluation rows: **{len(y)}**\n- Positive rate: **{positive:.2%}**\n- Accuracy: **{agg_c['Accuracy']:.2%}**\n- Balanced Accuracy: **{agg_c['BalancedAccuracy']:.2%}**\n- Precision: **{agg_c['Precision']:.2%}**\n- Recall: **{agg_c['Recall']:.2%}**\n- F1: **{agg_c['F1']:.2%}**\n- ROC-AUC: **{agg_c['ROC_AUC']:.2%}**\n- Ratio MAE: **{agg_r['MAE_ratio']:.4f}**\n- Majority baseline: **{majority:.2%}**\n- Deployment model: **{dep_name}**\n- All gates passed: **{all(gates.values())}**\n",encoding='utf-8')
    print(json.dumps({'regression':agg_r,'classification':agg_c,'majority':majority,'gates':gates,'all_passed':all(gates.values()),'deployment':dep_name},indent=2))
if __name__=='__main__': main()
