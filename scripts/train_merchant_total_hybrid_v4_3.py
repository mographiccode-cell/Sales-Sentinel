from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
VERSION='SALES-SENTINEL-MERCHANT-TOTAL-HYBRID-4.3'
DAILY=ROOT/'data'/'saudi_v1_3'/'saudi_daily_sama_calibrated_v1_3.csv'
CAT=ROOT/'data'/'merchant_v4'/'merchant_sector_daily_features_v4.csv'
RICH=ROOT/'data'/'merchant_v4_2'/'rich_weekly_panel_v4_2.csv'
OUT=ROOT/'reports'/'merchant_total_hybrid_v4_3';MOD=ROOT/'models'/'merchant_total_hybrid_v4_3';DATA=ROOT/'data'/'merchant_v4_3'
for p in (OUT,MOD,DATA):p.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'development_report.json';SUMMARY=OUT/'development_summary.md';MODEL=MOD/'merchant_total_hybrid_v4_3.joblib';PANEL=DATA/'merchant_total_feature_panel_v4_3.csv'
SEED=42


def safe_change(s,lag):
 p=s.shift(lag);return (s-p)/p.abs().replace(0,np.nan)

def week_start(s):
 d=pd.to_datetime(s);return d-pd.to_timedelta((d.dt.dayofweek+1)%7,unit='D')

def merchant_features(d):
 d=d.sort_values('date').reset_index(drop=True);X=pd.DataFrame(index=d.index)
 base=['sama_calibrated_net_sales_sar','gross_sales_sar','invoice_count','unique_observed_customers','new_observed_customers','returning_observed_customers','unique_products','units','average_invoice_value_sar','return_rate_value','transaction_rows']
 for c in base:
  s=d[c].astype(float);X[f'{c}_log']=np.log1p(s.clip(lower=0))
  for w in (7,14,28,56):X[f'{c}_ratio_{w}']=s/s.rolling(w,min_periods=w).mean().replace(0,np.nan)
  for lag in (1,7,14,28):X[f'{c}_change_{lag}']=safe_change(s,lag)
 X['electronic_share']=d.electronic_invoice_count/d.invoice_count.clip(lower=1);X['new_customer_share']=d.new_observed_customers/d.unique_observed_customers.clip(lower=1);X['returning_customer_share']=d.returning_observed_customers/d.unique_observed_customers.clip(lower=1);X['units_per_invoice']=d.units/d.invoice_count.clip(lower=1);X['products_per_invoice_proxy']=d.unique_products/d.invoice_count.clip(lower=1);X['sales_per_customer']=d.sama_calibrated_net_sales_sar/d.unique_observed_customers.clip(lower=1)
 for c in ['electronic_share','new_customer_share','returning_customer_share','units_per_invoice','products_per_invoice_proxy','sales_per_customer']:
  X[f'{c}_delta7']=X[c]-X[c].rolling(7,min_periods=7).mean();X[f'{c}_delta28']=X[c]-X[c].rolling(28,min_periods=28).mean()
 s=d.sama_weekly_market_index.astype(float);X['sama_market_index']=s;X['sama_market_index_change_1']=safe_change(s,1);X['sama_market_index_change_7']=safe_change(s,7)
 dates=pd.to_datetime(d.date);dow=dates.dt.dayofweek.astype(float);doy=dates.dt.dayofyear.astype(float);X['dow_sin']=np.sin(2*np.pi*dow/7);X['dow_cos']=np.cos(2*np.pi*dow/7);X['year_sin']=np.sin(2*np.pi*doy/365.25);X['year_cos']=np.cos(2*np.pi*doy/365.25);X['salary_period']=dates.dt.day.between(24,31).astype(float);X['national_day_window']=((dates.dt.month==9)&dates.dt.day.between(16,30)).astype(float);X['founding_day_window']=((dates.dt.month==2)&dates.dt.day.between(15,29)).astype(float)
 return X

def category_pivot():
 c=pd.read_csv(CAT,parse_dates=['date']); wanted=['net_sales_ratio_mean_7','net_sales_ratio_mean_28','net_sales_change_7','invoice_count_ratio_mean_7','invoice_count_ratio_mean_28','invoice_count_change_7','observed_customer_count_ratio_mean_7','observed_customer_count_change_7','unique_products_ratio_mean_7','unique_products_change_7','return_rate_value_ratio_mean_7','return_rate_value_change_7','category_share_ratio_28','category_share_change_7','sama_predicted_value_h1_change_vs_last','sama_predicted_value_h2_change_vs_last','sama_predicted_count_h1_change_vs_last','sama_predicted_count_h2_change_vs_last']
 wanted=[x for x in wanted if x in c.columns]
 if len(wanted)<12:raise RuntimeError(f'Too few category features found: {wanted}')
 p=c[['date','category']+wanted].pivot_table(index='date',columns='category',values=wanted,aggfunc='first');p.columns=[f'cat__{str(cat).replace(" ","_")}__{feat}' for feat,cat in p.columns];return p.reset_index()

def rich_pivot():
 r=pd.read_csv(RICH,parse_dates=['week_start']); wanted=['sku_dropout_rate','sku_new_rate','customer_dropout_rate','customer_new_rate','sku_sales_hhi','sku_top5_share','customer_sales_hhi','customer_top5_share','overlap_sku_price_change','price_increase_share_gt5','price_decrease_share_gt5','unique_products_ratio_4','observed_customer_count_ratio_4','avg_skus_per_invoice_ratio_4','avg_lines_per_invoice_ratio_4']
 wanted=[x for x in wanted if x in r.columns]
 if len(wanted)<10:raise RuntimeError(f'Too few rich features found: {wanted}')
 p=r[['week_start','category']+wanted].pivot_table(index='week_start',columns='category',values=wanted,aggfunc='first');p.columns=[f'rich__{str(cat).replace(" ","_")}__{feat}' for feat,cat in p.columns];p=p.reset_index();p['available_week_start']=p.week_start+pd.Timedelta(days=7);return p.drop(columns=['week_start'])

def build():
 d=pd.read_csv(DAILY,parse_dates=['date']).sort_values('date').reset_index(drop=True);X=merchant_features(d);cp=category_pivot();rp=rich_pivot();q=d[['date']].merge(cp,on='date',how='left',validate='one_to_one');q['current_week_start']=week_start(q.date);q=q.merge(rp,on='available_week_start',how='left',left_on='current_week_start',right_on='available_week_start',validate='many_to_one').drop(columns=['current_week_start','available_week_start']);X=pd.concat([X,q.drop(columns=['date'])],axis=1)
 # Merchant-total target: next 7 days vs trailing 28-day average. This is the original Sales Sentinel store-level objective.
 sales=d.sama_calibrated_net_sales_sar.clip(lower=0).astype(float);base=sales.rolling(28,min_periods=28).mean();future=sum(sales.shift(-h) for h in range(1,8));ratio=future/(7*base.replace(0,np.nan));target=(ratio<.8).astype(int)
 X=X.replace([np.inf,-np.inf],np.nan)
 for c in X:
  if 'ratio' in c or 'index' in c:X[c]=X[c].fillna(1.)
  else:X[c]=X[c].fillna(0.)
 good=(d.date>=d.date.min()+pd.Timedelta(days=56))&ratio.notna()&base.gt(0);meta=pd.DataFrame({'date':d.date,'future_ratio':ratio,'target':target}).loc[good].reset_index(drop=True);return meta,X.loc[good].reset_index(drop=True)

def factories():
 return {'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=.08,class_weight='balanced',max_iter=5000,random_state=SEED)),'extra_trees':ExtraTreesClassifier(n_estimators=900,max_depth=7,min_samples_leaf=7,max_features=.50,class_weight='balanced',random_state=SEED,n_jobs=-1),'hist_gb':HistGradientBoostingClassifier(max_iter=300,learning_rate=.022,max_leaf_nodes=12,min_samples_leaf=24,l2_regularization=12.,random_state=SEED)}
def fitc(m,X,y):
 if isinstance(m,HistGradientBoostingClassifier):
  pos=max(int(y.sum()),1);neg=max(len(y)-pos,1);return m.fit(X,y,sample_weight=np.where(np.asarray(y)==1,neg/pos,1.))
 return m.fit(X,y)
def met(y,s,t):
 y=np.asarray(y,int);p=np.asarray(s)>=t;return {'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'precision':float(precision_score(y,p,zero_division=0)),'recall':float(recall_score(y,p,zero_division=0)),'f1':float(f1_score(y,p,zero_division=0)),'roc_auc':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,'alert_rate':float(p.mean()),'green_npv':float(((y==0)&~p).sum()/max((~p).sum(),1)),'tp':int(((y==1)&p).sum()),'fp':int(((y==0)&p).sum()),'fn':int(((y==1)&~p).sum()),'tn':int(((y==0)&~p).sum())}
def folds(meta):
 windows=[('2023-07-08','2023-09-30'),('2023-10-08','2023-12-31'),('2024-01-08','2024-03-31'),('2024-04-08','2024-06-30'),('2024-07-08','2024-08-19')];out=[]
 for st,en in windows:
  st=pd.Timestamp(st);en=pd.Timestamp(en);tr=meta.date<=st-pd.Timedelta(days=8);va=meta.date.between(st,en)
  if tr.sum()>=100 and va.sum()>=35 and meta.loc[tr,'target'].nunique()==2 and meta.loc[va,'target'].nunique()==2:out.append((st,en,tr,va))
 return out
def choose(y,s):
 rows=[]
 for t in np.unique(np.r_[np.linspace(.05,.95,181),np.quantile(s,np.linspace(.02,.98,97))]):
  m=met(y,s,float(t));rows.append((float(t),m))
 feas=[z for z in rows if z[1]['recall']>=.70 and z[1]['alert_rate']<=.45];pool=feas if feas else rows;pool.sort(key=lambda z:(z[1]['roc_auc'],z[1]['balanced_accuracy'],z[1]['f1'],z[1]['recall']),reverse=True);return pool[0],len(feas)
def choose_red(y,s,watch):
 rows=[]
 for t in np.unique(np.r_[np.linspace(max(watch,.35),.99,130),np.quantile(s,np.linspace(.60,.995,70))]):
  m=met(y,s,float(t));
  if m['tp']+m['fp']>=5 and m['precision']>=.70:rows.append((float(t),m))
 if not rows:return .99,met(y,s,.99),0
 rows.sort(key=lambda z:(z[1]['recall'],z[1]['precision']),reverse=True);return rows[0][0],rows[0][1],len(rows)

def main():
 meta,X=build();PANEL.write_text(pd.concat([meta,X],axis=1).to_csv(index=False),encoding='utf-8');fs=folds(meta)
 if len(fs)<5:raise RuntimeError(f'Need 5 OOF folds, got {len(fs)}')
 rows=[];fm=[]
 for fid,(st,en,tr,va) in enumerate(fs):
  y=meta.loc[tr,'target'].astype(int);q=meta.loc[va,['date','target']].copy();q['fold_id']=fid;pr=[]
  for n,f in factories().items():m=fitc(clone(f),X.loc[tr],y);p=m.predict_proba(X.loc[va])[:,1];q[n]=p;pr.append(p)
  reg=HistGradientBoostingRegressor(max_iter=300,learning_rate=.025,max_leaf_nodes=12,min_samples_leaf=24,l2_regularization=10.,random_state=SEED).fit(X.loc[tr],meta.loc[tr,'future_ratio'].clip(0,2.5));rr=reg.predict(X.loc[va]);q['reg_risk']=1/(1+np.exp((rr-.8)/.08));q['ensemble']=np.column_stack(pr).mean(1);rows.append(q);fm.append({'fold_id':fid,'start':str(st.date()),'end':str(en.date()),'train_rows':int(tr.sum()),'validation_rows':int(va.sum()),'positives':int(q.target.sum())})
 q=pd.concat(rows,ignore_index=True).reset_index(drop=True);y=q.target.to_numpy(int);defs={'logistic':q.logistic.to_numpy(float),'extra_trees':q.extra_trees.to_numpy(float),'hist_gb':q.hist_gb.to_numpy(float),'ensemble':q.ensemble.to_numpy(float),'ensemble_reg20':.8*q.ensemble.to_numpy(float)+.2*q.reg_risk.to_numpy(float),'ensemble_reg40':.6*q.ensemble.to_numpy(float)+.4*q.reg_risk.to_numpy(float)};cand={}
 for n,s in defs.items():(t,m),nf=choose(y,s);cand[n]={'threshold':t,'metrics':m,'feasible_thresholds':nf}
 bestn=max(cand,key=lambda n:(cand[n]['metrics']['roc_auc'],cand[n]['metrics']['balanced_accuracy'],cand[n]['metrics']['f1']));best=cand[bestn];score=defs[bestn];red,redm,nred=choose_red(y,score,best['threshold']);per=[]
 for fid,z in q.groupby('fold_id'):
  idx=z.index.to_numpy();per.append({'fold_id':int(fid),**met(z.target.to_numpy(int),score[idx],best['threshold'])})
 # Seven weekday cohorts: within each cohort, 7-day targets do not overlap.
 cohort=[]
 for dow,z in q.assign(dow=q.date.dt.dayofweek).groupby('dow'):
  if z.target.nunique()==2 and len(z)>=20:
   idx=z.index.to_numpy();cohort.append({'weekday':int(dow),'rows':len(z),'positives':int(z.target.sum()),'roc_auc':float(roc_auc_score(z.target,score[idx]))})
 med_auc=float(np.median([x['roc_auc'] for x in cohort]));min_auc=float(np.min([x['roc_auc'] for x in cohort]))
 # External-context ablation removes category-SAMA forecast fields and market index, preserving merchant/category operational signals.
 ext=[c for c in X if 'sama_predicted_' in c or c.startswith('sama_market_index')];Xm=X.drop(columns=ext);abl=[]
 for fid,(st,en,tr,va) in enumerate(fs):m=fitc(clone(factories()['logistic']),Xm.loc[tr],meta.loc[tr,'target'].astype(int));abl.extend(m.predict_proba(Xm.loc[va])[:,1].tolist())
 merchant_auc=float(roc_auc_score(y,np.asarray(abl)));hybrid_auc=float(best['metrics']['roc_auc']);worst=min([x['recall'] for x in per if x['tp']+x['fn']>=5],default=1.)
 contract={'oof_auc_min':.80,'oof_balanced_accuracy_min':.72,'oof_recall_min':.70,'oof_green_npv_min':.90,'oof_alert_rate_max':.45,'median_nonoverlap_weekday_auc_min':.75,'min_nonoverlap_weekday_auc_min':.60,'worst_fold_recall_min':.50}
 gates={'rolling_origin_past_only':True,'target_purge_7days':True,'previous_completed_week_rich_features_only':True,'oof_auc':hybrid_auc>=contract['oof_auc_min'],'oof_balanced_accuracy':best['metrics']['balanced_accuracy']>=contract['oof_balanced_accuracy_min'],'oof_recall':best['metrics']['recall']>=contract['oof_recall_min'],'oof_green_npv':best['metrics']['green_npv']>=contract['oof_green_npv_min'],'oof_alert_rate':best['metrics']['alert_rate']<=contract['oof_alert_rate_max'],'weekday_median_auc':med_auc>=contract['median_nonoverlap_weekday_auc_min'],'weekday_min_auc':min_auc>=contract['min_nonoverlap_weekday_auc_min'],'fold_stability':worst>=contract['worst_fold_recall_min']}
 yf=meta.target.astype(int);models={n:fitc(clone(f),X,yf) for n,f in factories().items()};reg=HistGradientBoostingRegressor(max_iter=300,learning_rate=.025,max_leaf_nodes=12,min_samples_leaf=24,l2_regularization=10.,random_state=SEED).fit(X,meta.future_ratio.clip(0,2.5));joblib.dump({'version':VERSION,'status':'DEVELOPMENT_FROZEN_PENDING_INDEPENDENT_STRESS','feature_columns':list(X.columns),'models':models,'regressor':reg,'score_variant':bestn,'watch_threshold':best['threshold'],'red_threshold':red,'target_definition':'merchant total next7 sales <80% trailing28 daily mean x7'},MODEL)
 rep={'version':VERSION,'status':'DEVELOPMENT_FROZEN_PENDING_INDEPENDENT_STRESS','scientific_boundary':'Merchant-total target on UCI-derived Saudi-localized synthetic microdata. Historical periods are development rolling-origin evidence, not a new untouched final test.','source_rows_proven_by_upstream_v1_3_1':1049042,'supervised_days':len(meta),'feature_count':X.shape[1],'folds':fm,'candidate_oof':cand,'selected':{'score_variant':bestn,'watch_threshold':best['threshold'],'red_threshold':red,'metrics':best['metrics'],'red_metrics':redm,'red_candidates':nred,'worst_fold_recall':worst,'per_fold':per},'non_overlapping_weekday_cohorts':cohort,'median_nonoverlap_weekday_auc':med_auc,'min_nonoverlap_weekday_auc':min_auc,'ablation':{'merchant_operational_logistic_without_external_sama_auc':merchant_auc,'selected_hybrid_auc':hybrid_auc,'delta_auc':hybrid_auc-merchant_auc},'contract':contract,'gates':gates,'all_development_gates_passed':bool(all(gates.values())),'next_required_evidence':'Independent frozen merchant-total stress test using patterns not used in this development.'};REPORT.write_text(json.dumps(rep,indent=2),encoding='utf-8');SUMMARY.write_text('# Sales Sentinel v4.3 — Merchant Total Hybrid\n\n'+f"- Rolling OOF ROC-AUC **{hybrid_auc:.2%}**\n- Balanced Accuracy **{best['metrics']['balanced_accuracy']:.2%}**\n- Precision **{best['metrics']['precision']:.2%}**\n- Recall **{best['metrics']['recall']:.2%}**\n- F1 **{best['metrics']['f1']:.2%}**\n- GREEN NPV **{best['metrics']['green_npv']:.2%}**\n- Alert rate **{best['metrics']['alert_rate']:.2%}**\n- Median non-overlap weekday AUC **{med_auc:.2%}**\n- Minimum non-overlap weekday AUC **{min_auc:.2%}**\n- Development gates **{all(gates.values())}**\n",encoding='utf-8');print(json.dumps(rep,indent=2))

if __name__=='__main__':main()
