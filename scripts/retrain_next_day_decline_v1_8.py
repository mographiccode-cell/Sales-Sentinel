from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED=42; H=1; DECLINE=.20; BASELINE=28; PURGE=1; VAL=60; BLOCK=45; MIN_TRAIN=220; MIN_HISTORY=56
ROOT=Path(__file__).resolve().parents[1]
DAILY=ROOT/'data'/'saudi_v1_3'/'saudi_daily_sama_calibrated_v1_3.csv'
SAMA_FC=ROOT/'data'/'sama_pos'/'sama_market_walkforward_forecasts_2023_2025.csv'
OUT=ROOT/'models'/'saudi_v1_8'; REP=ROOT/'reports'/'saudi_v1_8'; OUT.mkdir(parents=True,exist_ok=True); REP.mkdir(parents=True,exist_ok=True)
RAMADAN=[('2023-03-23','2023-04-20'),('2024-03-11','2024-04-09')]; EID_FITR=[('2023-04-21','2023-04-23'),('2024-04-10','2024-04-12')]; HAJJ=[('2023-06-19','2023-06-30'),('2024-06-07','2024-06-19')]; EID_ADHA=[('2023-06-28','2023-07-01'),('2024-06-16','2024-06-19')]
def ratio(a,b): return a/b.replace(0,np.nan)
def isin(dt,r): return int(any(pd.Timestamp(a)<=dt<=pd.Timestamp(b) for a,b in r))
def wk(dt): dt=pd.Timestamp(dt); return dt-pd.Timedelta(days=(dt.dayofweek+1)%7)
def loadfc(): return pd.read_csv(SAMA_FC,parse_dates=['origin_week_start','forecast_h1_week_start','forecast_h2_week_start']).set_index('origin_week_start').to_dict('index')

def market_row(dt,L):
    current=wk(dt); last=current-pd.Timedelta(days=7); r=L.get(last)
    if r is None: return [np.nan]*8
    tomorrow=pd.Timestamp(dt)+pd.Timedelta(days=1); tw=wk(tomorrow)
    if tw==pd.Timestamp(r['forecast_h1_week_start']): vi=float(r['predicted_value_h1_index_52median']); ci=float(r['predicted_count_h1_index_52median']); vc=float(r['predicted_value_h1_change_vs_last']); cc=float(r['predicted_count_h1_change_vs_last'])
    else: vi=float(r['predicted_value_h2_index_52median']); ci=float(r['predicted_count_h2_index_52median']); vc=float(r['predicted_value_h2_change_vs_last']); cc=float(r['predicted_count_h2_change_vs_last'])
    return [float(r['predicted_value_h1_index_52median']),float(r['predicted_value_h2_index_52median']),float(r['predicted_count_h1_index_52median']),float(r['predicted_count_h2_index_52median']),vi,ci,vc,cc]

def build(d,L):
    d=d.sort_values('date').reset_index(drop=True).copy(); d['date']=pd.to_datetime(d['date']); sales=d['sama_calibrated_net_sales_sar'].astype(float); base=d['base_net_sales_sar_unscaled'].astype(float); baseline=sales.rolling(BASELINE).mean(); tomorrow=sales.shift(-1)
    x=pd.DataFrame({'date':d['date']})
    S={'sales':sales,'base':base,'customers':d['unique_observed_customers'].astype(float),'invoices':d['invoice_count'].astype(float),'transactions':d['transaction_rows'].astype(float),'avg_invoice':d['average_invoice_value_sar'].astype(float),'return_rate':d['return_rate_value'].astype(float),'returning':d['returning_observed_customers'].astype(float),'new_customers':d['new_observed_customers'].astype(float)}
    for n,s in S.items():
        x[f'{n}_t0']=s
        for lag in (1,2,3,7,14,28,56): x[f'{n}_lag_{lag}']=s.shift(lag)
        for w in (7,14,28,56): x[f'{n}_mean_{w}']=s.rolling(w).mean(); x[f'{n}_std_{w}']=s.rolling(w).std()
    for n in ('sales','base','customers','invoices','transactions'): x[f'{n}_vs_mean7']=ratio(x[f'{n}_t0'],x[f'{n}_mean_7']); x[f'{n}_vs_mean28']=ratio(x[f'{n}_t0'],x[f'{n}_mean_28'])
    x['calibration_ratio_t0']=ratio(x['sales_t0'],x['base_t0']); x['calibration_ratio_mean7']=ratio(x['sales_mean_7'],x['base_mean_7'])
    m=np.asarray([market_row(dt,L) for dt in d['date']]); names=['sama_h1_value_idx','sama_h2_value_idx','sama_h1_count_idx','sama_h2_count_idx','tomorrow_pred_sama_value_idx','tomorrow_pred_sama_count_idx','tomorrow_pred_sama_value_change','tomorrow_pred_sama_count_change']
    for i,n in enumerate(names): x[n]=m[:,i]
    actual=d['sama_weekly_market_index'].astype(float)
    for lag in (7,14,28): x[f'actual_sama_lag_{lag}']=actual.shift(lag)
    td=d['date']+pd.Timedelta(days=1); doy=td.dt.dayofyear.astype(float); x['tomorrow_doy_sin']=np.sin(2*np.pi*doy/365.25); x['tomorrow_doy_cos']=np.cos(2*np.pi*doy/365.25); x['tomorrow_ramadan']=[isin(z,RAMADAN) for z in td]; x['tomorrow_fitr']=[isin(z,EID_FITR) for z in td]; x['tomorrow_hajj']=[isin(z,HAJJ) for z in td]; x['tomorrow_adha']=[isin(z,EID_ADHA) for z in td]; x['tomorrow_salary_period']=td.dt.day.between(25,28).astype(int); x['tomorrow_national_day']=((td.dt.month==9)&(td.dt.day==23)).astype(int); x['tomorrow_founding_day']=((td.dt.month==2)&(td.dt.day==22)).astype(int)
    x['baseline28']=baseline; x['tomorrow_ratio']=ratio(tomorrow,baseline); x['target']=(tomorrow<(1-DECLINE)*baseline).astype(float); x.loc[len(x)-1,['tomorrow_ratio','target']]=np.nan
    x=x.iloc[MIN_HISTORY:].replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True); features=[c for c in x if c not in {'date','tomorrow_ratio','target'}]; return x,features

def specs(): return {'LogisticRegression':Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.25,max_iter=5000,class_weight='balanced',random_state=SEED))]),'RandomForest':RandomForestClassifier(n_estimators=900,max_depth=8,min_samples_leaf=4,max_features=.65,class_weight='balanced_subsample',random_state=SEED,n_jobs=-1),'ExtraTrees':ExtraTreesClassifier(n_estimators=1100,max_depth=10,min_samples_leaf=3,max_features=.7,class_weight='balanced',random_state=SEED,n_jobs=-1),'HistGradientBoosting':HistGradientBoostingClassifier(learning_rate=.03,max_iter=400,max_leaf_nodes=15,min_samples_leaf=12,l2_regularization=3,random_state=SEED)}
def M(y,p,t):
    z=(p>=t).astype(int); return {'Accuracy':float(accuracy_score(y,z)),'BalancedAccuracy':float(balanced_accuracy_score(y,z)),'Precision':float(precision_score(y,z,zero_division=0)),'Recall':float(recall_score(y,z,zero_division=0)),'F1':float(f1_score(y,z,zero_division=0)),'ROC_AUC':float(roc_auc_score(y,p)),'ConfusionMatrix':confusion_matrix(y,z,labels=[0,1]).tolist()}
def score(m): return .35*m['BalancedAccuracy']+.30*m['F1']+.20*m['Accuracy']+.15*m['ROC_AUC']
def select(hist,F):
    vs=len(hist)-VAL; ie=vs-PURGE
    if ie<MIN_TRAIN: raise RuntimeError('short history')
    tr=hist.iloc[:ie]; va=hist.iloc[vs:]; y=va.target.astype(int).to_numpy(); out={}
    for n,s in specs().items():
        q=clone(s).fit(tr[F],tr.target.astype(int)); p=q.predict_proba(va[F])[:,1]; best=None
        for t in np.arange(.05,.951,.005):
            m=M(y,p,float(t)); penalty=max(0,.70-m['Recall']); c=(score(m)-.25*penalty,m['BalancedAccuracy'],m['F1'],m['Accuracy'],-abs(t-.5),float(t),m)
            if best is None or c[:5]>best[:5]: best=c
        out[n]={'threshold':best[5],'metrics':best[6],'score':best[0]}
    n=max(out,key=lambda k:(out[k]['score'],out[k]['metrics']['BalancedAccuracy'],out[k]['metrics']['F1'])); return n,float(out[n]['threshold']),out

def main():
    d=pd.read_csv(DAILY,parse_dates=['date']); frame,F=build(d,loadfc()); first=max(MIN_TRAIN+VAL+2*PURGE,260); ys=[]; ps=[]; zs=[]; folds=[]; choices=[]
    for no,start in enumerate(range(first,len(frame),BLOCK),1):
        end=min(start+BLOCK,len(frame));
        if end-start<20: continue
        hist=frame.iloc[:start-PURGE]; te=frame.iloc[start:end]; n,t,sel=select(hist,F); choices.append(n); q=clone(specs()[n]).fit(hist[F],hist.target.astype(int)); p=q.predict_proba(te[F])[:,1]; y=te.target.astype(int).to_numpy(); z=(p>=t).astype(int); ys+=y.tolist(); ps+=p.tolist(); zs+=z.tolist(); folds.append({'fold':no,'train_end':str(hist.date.iloc[-1].date()),'test_start':str(te.date.iloc[0].date()),'test_end':str(te.date.iloc[-1].date()),'model':n,'threshold':t,'metrics':M(y,p,t),'selection':sel})
    y=np.array(ys); p=np.array(ps); z=np.array(zs); cm=confusion_matrix(y,z,labels=[0,1]); agg={'Accuracy':float(accuracy_score(y,z)),'BalancedAccuracy':float(balanced_accuracy_score(y,z)),'Precision':float(precision_score(y,z,zero_division=0)),'Recall':float(recall_score(y,z,zero_division=0)),'F1':float(f1_score(y,z,zero_division=0)),'ROC_AUC':float(roc_auc_score(y,p)),'ConfusionMatrix':cm.tolist(),'Rows':int(len(y)),'PositiveRate':float(y.mean())}; majority=max(float(y.mean()),1-float(y.mean())); gates={'accuracy_at_least_90pct':agg['Accuracy']>=.90,'balanced_accuracy_at_least_80pct':agg['BalancedAccuracy']>=.80,'recall_at_least_75pct':agg['Recall']>=.75,'f1_at_least_70pct':agg['F1']>=.70,'roc_auc_at_least_85pct':agg['ROC_AUC']>=.85,'beats_majority_accuracy':agg['Accuracy']>majority}
    dep_hist=frame.iloc[:-PURGE]; n,t,sel=select(dep_hist,F); q=clone(specs()[n]).fit(dep_hist[F],dep_hist.target.astype(int)); joblib.dump({'model':q,'features':F,'threshold':t,'target':'next-day sales <80% of trailing 28-day mean','version':'SALES-DECLINE-1.8-NEXT-DAY'},OUT/'next_day_sales_decline_classifier_v1_8.joblib')
    R={'version':'SALES-DECLINE-1.8-NEXT-DAY','target':'next-day SAMA-calibrated merchant sales at least 20% below trailing 28-day mean','evaluation':'nested expanding walk-forward backtest','aggregate':agg,'majority_accuracy':majority,'folds':folds,'choices':dict(Counter(choices)),'deployment':{'model':n,'threshold':t,'latest_selection':sel},'gates':gates,'all_gates_passed':bool(all(gates.values())),'leakage':'Tomorrow SAMA actual value is never used; only SAMA forecasts generated from the last completed official week are features.'}; (REP/'next_day_decline_v1_8_report.json').write_text(json.dumps(R,indent=2),encoding='utf-8'); (OUT/'model_metadata_v1_8.json').write_text(json.dumps(R,indent=2),encoding='utf-8'); (REP/'next_day_decline_v1_8_summary.md').write_text(f"# Next-Day Sales Decline v1.8\n\n- Rows: **{len(y)}**\n- Positive rate: **{y.mean():.2%}**\n- Accuracy: **{agg['Accuracy']:.2%}**\n- Balanced Accuracy: **{agg['BalancedAccuracy']:.2%}**\n- Precision: **{agg['Precision']:.2%}**\n- Recall: **{agg['Recall']:.2%}**\n- F1: **{agg['F1']:.2%}**\n- ROC-AUC: **{agg['ROC_AUC']:.2%}**\n- Majority baseline: **{majority:.2%}**\n- Deployment: **{n}**, threshold **{t:.3f}**\n- All gates passed: **{all(gates.values())}**\n",encoding='utf-8'); print(json.dumps({'aggregate':agg,'majority':majority,'gates':gates,'all_passed':all(gates.values()),'deployment':n,'threshold':t},indent=2))
if __name__=='__main__': main()
