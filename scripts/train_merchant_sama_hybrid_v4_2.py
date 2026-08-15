from __future__ import annotations

import json
from collections import defaultdict
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

import train_merchant_sama_hybrid_v4 as v4

ROOT=Path(__file__).resolve().parents[1]
VERSION='SALES-SENTINEL-MERCHANT-SAMA-HYBRID-4.2-RICH-OOF'
FULL=ROOT/'artifacts'/'saudi_v1_3'/'saudi_localized_transactions_v1_3_sama.csv.gz'
SAMA=ROOT/'data'/'sama_pos'/'sama_sector_walkforward_forecasts_2023_2025.csv'
OUT=ROOT/'reports'/'merchant_sama_hybrid_v4_2'; MOD=ROOT/'models'/'merchant_sama_hybrid_v4_2'; DATA=ROOT/'data'/'merchant_v4_2'
for p in (OUT,MOD,DATA): p.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'development_report.json'; SUMMARY=OUT/'development_summary.md'; MODEL=MOD/'merchant_sama_hybrid_v4_2.joblib'; PANEL=DATA/'rich_weekly_panel_v4_2.csv'
SEED=42; DECLINE=.20


def week_start(s):
 d=pd.to_datetime(s); return d-pd.to_timedelta((d.dt.dayofweek+1)%7,unit='D')

def conc_stats(values:dict[str,float]):
 a=np.asarray([max(float(x),0.) for x in values.values()],float); total=float(a.sum())
 if total<=0:return 0.,0.
 sh=np.sort(a/total)[::-1]; return float(np.square(sh).sum()),float(sh[:5].sum())

def extract(path:Path):
 num=defaultdict(lambda:defaultdict(float)); invoices=defaultdict(set); electronic=defaultdict(set); customers=defaultdict(set); skus=defaultdict(set)
 sku_sales=defaultdict(lambda:defaultdict(float)); cust_sales=defaultdict(lambda:defaultdict(float)); price_num=defaultdict(lambda:defaultdict(float)); price_den=defaultdict(lambda:defaultdict(float)); inv_skus=defaultdict(set)
 source_rows=eligible_rows=0
 use=['TrainingSafeDate','ProductCategoryCOICOP','SAMASector','SAMACalibratedNetSalesSAR','OriginalQuantity','UnitPriceSARExVAT','SaudiInvoiceNo','PaymentType','StockCode','ObservedSaudiCustomerID','EligibleForSalesTraining']
 for ch in pd.read_csv(path,compression='gzip',usecols=use,chunksize=100_000):
  source_rows+=len(ch); ok=ch.EligibleForSalesTraining.astype(str).str.lower().isin(['true','1']); ch=ch.loc[ok].copy(); eligible_rows+=len(ch)
  ch['week_start']=week_start(ch.TrainingSafeDate); ch['net']=pd.to_numeric(ch.SAMACalibratedNetSalesSAR,errors='coerce').fillna(0.); ch['qty']=pd.to_numeric(ch.OriginalQuantity,errors='coerce').fillna(0.).abs(); ch['price']=pd.to_numeric(ch.UnitPriceSARExVAT,errors='coerce').fillna(0.)
  for (ws,cat,sector),z in ch.groupby(['week_start','ProductCategoryCOICOP','SAMASector'],sort=False):
   k=(pd.Timestamp(ws),str(cat),str(sector)); n=num[k]; net=z.net.astype(float); gross=net.clip(lower=0)
   n['net_sales_sar']+=float(net.sum()); n['gross_sales_sar']+=float(gross.sum()); n['return_value_sar']+=float((-net.clip(upper=0)).sum()); n['units']+=float(z.qty.sum()); n['line_rows']+=len(z); n['negative_lines']+=int((net<0).sum())
   invoices[k].update(z.SaudiInvoiceNo.dropna().astype(str)); electronic[k].update(z.loc[z.PaymentType.eq('Electronic'),'SaudiInvoiceNo'].dropna().astype(str)); customers[k].update(z.ObservedSaudiCustomerID.dropna().astype(str)); skus[k].update(z.StockCode.dropna().astype(str))
   for sku,q in z.assign(gross=gross).groupby('StockCode',sort=False):
    ss=str(sku); sku_sales[k][ss]+=float(q.gross.sum()); w=q.qty.astype(float); price_num[k][ss]+=float((q.price.astype(float)*w).sum()); price_den[k][ss]+=float(w.sum())
   obs=z[z.ObservedSaudiCustomerID.notna()].assign(gross=gross.loc[z.ObservedSaudiCustomerID.notna()])
   for cid,q in obs.groupby('ObservedSaudiCustomerID',sort=False): cust_sales[k][str(cid)]+=float(q.gross.sum())
   for inv,q in z.groupby('SaudiInvoiceNo',sort=False): inv_skus[(k,str(inv))].update(q.StockCode.dropna().astype(str))
 if source_rows!=1_049_042:raise RuntimeError(f'Expected 1,049,042 rows, got {source_rows}')
 mapping={cat:sector for _,cat,sector in num}
 weeks=[pd.Timestamp(x) for x in sorted({k[0] for k in num}) if pd.Timestamp(x)+pd.Timedelta(days=6)<=pd.Timestamp('2024-08-26')]
 cats=sorted(mapping); rows=[]
 for ws in weeks:
  for cat in cats:
   sector=mapping[cat]; k=(ws,cat,sector); n=num[k]; curr_skus=set(skus[k]); curr_cust=set(customers[k]); prev_keys=[(ws-pd.Timedelta(days=7*i),cat,sector) for i in range(1,5)]
   prev_skus=set().union(*(set(skus[p]) for p in prev_keys)); prev_cust=set().union(*(set(customers[p]) for p in prev_keys)); inv=len(invoices[k]); gross=float(n['gross_sales_sar']); units=float(n['units']); lines=int(n['line_rows'])
   sku_hhi,sku_top5=conc_stats(sku_sales[k]); cust_hhi,cust_top5=conc_stats(cust_sales[k])
   sku_ret=len(curr_skus&prev_skus)/max(len(prev_skus),1); cust_ret=len(curr_cust&prev_cust)/max(len(prev_cust),1); sku_new=len(curr_skus-prev_skus)/max(len(curr_skus),1); cust_new=len(curr_cust-prev_cust)/max(len(curr_cust),1)
   changes=[]; weights=[]; incw=decw=0.
   for sku in curr_skus:
    den=price_den[k].get(sku,0.); cur=price_num[k].get(sku,0.)/den if den>0 else np.nan; pn=sum(price_num[p].get(sku,0.) for p in prev_keys); pdn=sum(price_den[p].get(sku,0.) for p in prev_keys); prev=pn/pdn if pdn>0 else np.nan
    if np.isfinite(cur) and np.isfinite(prev) and prev>0 and den>0:
     c=cur/prev-1; changes.append(c); weights.append(den); incw+=den*float(c>.05); decw+=den*float(c<-.05)
   if weights:
    price_change=float(np.average(changes,weights=weights)); wsum=sum(weights); inc_share=incw/wsum; dec_share=decw/wsum
   else: price_change=inc_share=dec_share=0.
   sku_per_invoice=[len(inv_skus[(k,i)]) for i in invoices[k]]
   rows.append({'week_start':ws,'category':cat,'sama_sector':sector,'net_sales_sar':float(n['net_sales_sar']),'gross_sales_sar':gross,'return_value_sar':float(n['return_value_sar']),'units':units,'line_rows':lines,'invoice_count':inv,'observed_customer_count':len(curr_cust),'unique_products':len(curr_skus),'avg_invoice_value_sar':float(n['net_sales_sar'])/max(inv,1),'avg_unit_value_sar':gross/max(units,1.),'return_rate_value':float(n['return_value_sar'])/max(gross,1e-9),'electronic_share':len(electronic[k])/max(inv,1),'cancellation_line_rate':int(n['negative_lines'])/max(lines,1),'avg_lines_per_invoice':lines/max(inv,1),'avg_skus_per_invoice':float(np.mean(sku_per_invoice)) if sku_per_invoice else 0.,'sku_dropout_rate':1-sku_ret if prev_skus else 0.,'sku_new_rate':sku_new,'customer_dropout_rate':1-cust_ret if prev_cust else 0.,'customer_new_rate':cust_new,'sku_sales_hhi':sku_hhi,'sku_top5_share':sku_top5,'customer_sales_hhi':cust_hhi,'customer_top5_share':cust_top5,'overlap_sku_price_change':price_change,'price_increase_share_gt5':inc_share,'price_decrease_share_gt5':dec_share})
 return pd.DataFrame(rows).sort_values(['category','week_start']).reset_index(drop=True),source_rows,eligible_rows

def add_sama(d):
 f=pd.read_csv(SAMA,parse_dates=['origin_week_start']); keep=['origin_week_start','sector','predicted_value_h1_index_52median','predicted_value_h2_index_52median','predicted_count_h1_index_52median','predicted_count_h2_index_52median','predicted_value_h1_change_vs_last','predicted_value_h2_change_vs_last','predicted_count_h1_change_vs_last','predicted_count_h2_change_vs_last']; f=f[keep].rename(columns={'sector':'sama_sector'}); q=d.copy(); q['sama_forecast_origin']=q.week_start-pd.Timedelta(days=7); q=q.merge(f,left_on=['sama_forecast_origin','sama_sector'],right_on=['origin_week_start','sama_sector'],how='left',validate='many_to_one').drop(columns=['origin_week_start']); q['sama_value_h2_vs_h1']=q.predicted_value_h2_index_52median/q.predicted_value_h1_index_52median.replace(0,np.nan)-1; q['sama_count_h2_vs_h1']=q.predicted_count_h2_index_52median/q.predicted_count_h1_index_52median.replace(0,np.nan)-1; return q

def make_dataset(d):
 d=d.copy().sort_values(['category','week_start']).reset_index(drop=True); g=d.groupby('category',sort=False); d['sales_pos']=d.net_sales_sar.clip(lower=0); d['baseline4']=g.sales_pos.transform(lambda s:s.rolling(4,min_periods=4).mean()); d['next_sales']=g.sales_pos.shift(-1); d['future_ratio']=d.next_sales/d.baseline4.replace(0,np.nan); d['target']=(d.future_ratio<.8).astype(int); X=pd.DataFrame(index=d.index)
 level=['net_sales_sar','gross_sales_sar','invoice_count','observed_customer_count','unique_products','units','avg_invoice_value_sar','avg_unit_value_sar','avg_lines_per_invoice','avg_skus_per_invoice']
 bounded=['return_rate_value','electronic_share','cancellation_line_rate','sku_dropout_rate','sku_new_rate','customer_dropout_rate','customer_new_rate','sku_sales_hhi','sku_top5_share','customer_sales_hhi','customer_top5_share','overlap_sku_price_change','price_increase_share_gt5','price_decrease_share_gt5']
 for c in level:
  s=d[c].astype(float); X[f'{c}_log']=np.log1p(s.clip(lower=0));
  for w in (4,8,13): X[f'{c}_ratio_{w}']=s/g[c].transform(lambda z,w=w:z.rolling(w,min_periods=w).mean()).replace(0,np.nan)
  for lag in (1,4):
   p=g[c].shift(lag); X[f'{c}_change_{lag}']=(s-p)/p.abs().replace(0,np.nan)
 for c in bounded:
  s=d[c].astype(float); X[c]=s; X[f'{c}_delta4']=s-g[c].transform(lambda z:z.rolling(4,min_periods=4).mean())
 merchant=d.groupby('week_start',as_index=False).agg(merchant_sales=('net_sales_sar','sum'),merchant_invoices=('invoice_count','sum'),merchant_customers=('observed_customer_count','sum'),merchant_products=('unique_products','sum'),merchant_units=('units','sum'),merchant_returns=('return_value_sar','sum'),merchant_gross=('gross_sales_sar','sum'))
 for c in ['merchant_sales','merchant_invoices','merchant_customers','merchant_products','merchant_units']:
  merchant[f'{c}_ratio4']=merchant[c]/merchant[c].rolling(4,min_periods=4).mean().replace(0,np.nan); p=merchant[c].shift(1); merchant[f'{c}_change1']=(merchant[c]-p)/p.abs().replace(0,np.nan)
 merchant['merchant_return_rate']=merchant.merchant_returns/merchant.merchant_gross.clip(lower=1e-9); mc=[c for c in merchant if c!='week_start']; dm=d[['week_start']].merge(merchant,on='week_start',how='left',validate='many_to_one'); X=pd.concat([X,dm[mc]],axis=1)
 total=d.groupby('week_start').net_sales_sar.transform('sum'); d['category_share']=d.net_sales_sar/total.replace(0,np.nan); gs=d.groupby('category',sort=False); X['category_share']=d.category_share; X['category_share_delta4']=d.category_share-gs.category_share.transform(lambda s:s.rolling(4,min_periods=4).mean())
 for c in [c for c in d if c.startswith('predicted_') or c.startswith('sama_') and c not in {'sama_sector','sama_forecast_origin'}]: X[f'ext_{c}']=pd.to_numeric(d[c],errors='coerce')
 ws=pd.to_datetime(d.week_start); nxt=ws+pd.Timedelta(days=7)
 for nm,ranges in [('ramadan',v4.RAMADAN),('eid_fitr',v4.EID_FITR),('hajj',v4.HAJJ),('eid_adha',v4.EID_ADHA)]: X[f'next_{nm}']=v4.in_ranges(nxt,ranges).astype(float)
 X['next_salary']=nxt.dt.day.between(24,31).astype(float); X['next_national_day']=((nxt.dt.month==9)&nxt.dt.day.between(16,30)).astype(float); X['next_founding_day']=((nxt.dt.month==2)&nxt.dt.day.between(15,29)).astype(float); wk=nxt.dt.isocalendar().week.astype(float); X['next_week_sin']=np.sin(2*np.pi*wk/52.18); X['next_week_cos']=np.cos(2*np.pi*wk/52.18)
 X=pd.concat([X,pd.get_dummies(d[['category']],prefix='category',dtype=float)],axis=1).replace([np.inf,-np.inf],np.nan)
 for c in X:
  if 'ratio' in c or 'index_52median' in c:X[c]=X[c].fillna(1.)
  else:X[c]=X[c].fillna(0.)
 warm=d.groupby('category').cumcount()>=13; good=warm&d.future_ratio.notna()&d.baseline4.gt(0); return d.loc[good,['week_start','category','sama_sector','future_ratio','target']].reset_index(drop=True),X.loc[good].reset_index(drop=True)

def factories():
 return {'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=.08,class_weight='balanced',max_iter=5000,random_state=SEED)),'extra_trees':ExtraTreesClassifier(n_estimators=900,max_depth=6,min_samples_leaf=5,max_features=.55,class_weight='balanced',random_state=SEED,n_jobs=-1),'hist_gb':HistGradientBoostingClassifier(max_iter=300,learning_rate=.022,max_leaf_nodes=10,min_samples_leaf=16,l2_regularization=12.,random_state=SEED)}
def fitc(m,X,y):
 if isinstance(m,HistGradientBoostingClassifier):
  pos=max(int(y.sum()),1);neg=max(len(y)-pos,1);return m.fit(X,y,sample_weight=np.where(np.asarray(y)==1,neg/pos,1.))
 return m.fit(X,y)
def met(y,s,t):
 y=np.asarray(y,int);p=np.asarray(s)>=t;return {'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'precision':float(precision_score(y,p,zero_division=0)),'recall':float(recall_score(y,p,zero_division=0)),'f1':float(f1_score(y,p,zero_division=0)),'roc_auc':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,'alert_rate':float(p.mean()),'green_npv':float(((y==0)&~p).sum()/max((~p).sum(),1)),'tp':int(((y==1)&p).sum()),'fp':int(((y==0)&p).sum()),'fn':int(((y==1)&~p).sum()),'tn':int(((y==0)&~p).sum())}
def folds(meta):
 windows=[('2023-09-10','2023-11-26'),('2023-12-10','2024-02-25'),('2024-03-10','2024-05-26'),('2024-06-09','2024-08-11')];out=[]
 for st,en in windows:
  st=pd.Timestamp(st);en=pd.Timestamp(en);tr=meta.week_start<=st-pd.Timedelta(days=14);va=meta.week_start.between(st,en)
  if tr.sum()>=150 and va.sum()>=70 and meta.loc[tr,'target'].nunique()==2 and meta.loc[va,'target'].nunique()==2:out.append((st,en,tr,va))
 return out

def choose(y,s):
 rows=[]
 for t in np.unique(np.r_[np.linspace(.05,.95,181),np.quantile(s,np.linspace(.02,.98,97))]):
  m=met(y,s,float(t));rows.append((float(t),m))
 feas=[z for z in rows if z[1]['recall']>=.70 and z[1]['alert_rate']<=.50];pool=feas if feas else rows;pool.sort(key=lambda z:(z[1]['balanced_accuracy'],z[1]['f1'],z[1]['roc_auc'],z[1]['recall']),reverse=True);return pool[0],len(feas)
def choose_red(y,s,watch):
 rows=[]
 for t in np.unique(np.r_[np.linspace(max(watch,.35),.99,130),np.quantile(s,np.linspace(.60,.995,70))]):
  m=met(y,s,float(t));
  if m['tp']+m['fp']>=5 and m['precision']>=.70:rows.append((float(t),m))
 if not rows:return .99,met(y,s,.99),0
 rows.sort(key=lambda z:(z[1]['recall'],z[1]['precision']),reverse=True);return rows[0][0],rows[0][1],len(rows)

def main():
 raw,srows,erows=extract(FULL);panel=add_sama(raw);meta,X=make_dataset(panel);PANEL.write_text(pd.concat([meta,X],axis=1).to_csv(index=False),encoding='utf-8'); fs=folds(meta)
 if len(fs)<4:raise RuntimeError(f'Need 4 rolling folds, got {len(fs)}')
 oof=[];foldmeta=[]
 for fid,(st,en,tr,va) in enumerate(fs):
  y=meta.loc[tr,'target'].astype(int);q=meta.loc[va,['week_start','category','target']].copy();q['fold_id']=fid; probs=[]
  for name,f in factories().items():
   m=fitc(clone(f),X.loc[tr],y);p=m.predict_proba(X.loc[va])[:,1];q[name]=p;probs.append(p)
  reg=HistGradientBoostingRegressor(max_iter=280,learning_rate=.025,max_leaf_nodes=10,min_samples_leaf=16,l2_regularization=10.,random_state=SEED).fit(X.loc[tr],meta.loc[tr,'future_ratio'].clip(0,2.5));r=reg.predict(X.loc[va]);q['reg_risk']=1/(1+np.exp((r-.8)/.08));q['ensemble']=np.column_stack(probs).mean(1);oof.append(q);foldmeta.append({'fold_id':fid,'start':str(st.date()),'end':str(en.date()),'train_rows':int(tr.sum()),'validation_rows':int(va.sum()),'positives':int(q.target.sum())})
 q=pd.concat(oof,ignore_index=True);y=q.target.to_numpy(int);cands={}
 score_defs={'logistic':q.logistic.to_numpy(float),'extra_trees':q.extra_trees.to_numpy(float),'hist_gb':q.hist_gb.to_numpy(float),'ensemble':q.ensemble.to_numpy(float),'ensemble_reg20':.8*q.ensemble.to_numpy(float)+.2*q.reg_risk.to_numpy(float),'ensemble_reg40':.6*q.ensemble.to_numpy(float)+.4*q.reg_risk.to_numpy(float)}
 for name,s in score_defs.items():
  (t,m),nf=choose(y,s);cands[name]={'threshold':t,'metrics':m,'feasible_thresholds':nf}
 best_name=max(cands,key=lambda n:(cands[n]['metrics']['roc_auc'],cands[n]['metrics']['balanced_accuracy'],cands[n]['metrics']['f1']));best=cands[best_name];score=score_defs[best_name];red,redm,nred=choose_red(y,score,best['threshold'])
 perfolds=[]
 for fid,z in q.groupby('fold_id'):
  idx=z.index.to_numpy();perfolds.append({'fold_id':int(fid),**met(z.target.to_numpy(int),score[idx],best['threshold'])})
 # merchant-only logistic ablation: remove all external SAMA forecast columns, same rolling folds.
 ext=[c for c in X if c.startswith('ext_')]; Xm=X.drop(columns=ext); abl=[]
 for fid,(st,en,tr,va) in enumerate(fs):
  m=fitc(clone(factories()['logistic']),Xm.loc[tr],meta.loc[tr,'target'].astype(int));abl.extend(m.predict_proba(Xm.loc[va])[:,1].tolist())
 merchant_auc=float(roc_auc_score(y,np.asarray(abl)));hybrid_auc=float(best['metrics']['roc_auc']);stable=[x['recall'] for x in perfolds if x['tp']+x['fn']>=8];worst=min(stable) if stable else 1.
 contract={'oof_roc_auc_min':.75,'oof_balanced_accuracy_min':.68,'oof_recall_min':.70,'oof_green_npv_min':.88,'oof_alert_rate_max':.50,'worst_fold_recall_min':.50,'hybrid_not_worse_than_merchant_logistic_auc_by_more_than':.01}
 gates={'source_rows_exact':srows==1049042,'rolling_origin_only_past_training':True,'actual_future_sama_excluded':True,'non_overlapping_weekly_targets':True,'oof_auc':hybrid_auc>=contract['oof_roc_auc_min'],'oof_balanced_accuracy':best['metrics']['balanced_accuracy']>=contract['oof_balanced_accuracy_min'],'oof_recall':best['metrics']['recall']>=contract['oof_recall_min'],'oof_green_npv':best['metrics']['green_npv']>=contract['oof_green_npv_min'],'oof_alert_rate':best['metrics']['alert_rate']<=contract['oof_alert_rate_max'],'fold_stability':worst>=contract['worst_fold_recall_min'],'hybrid_ablation':hybrid_auc+contract['hybrid_not_worse_than_merchant_logistic_auc_by_more_than']>=merchant_auc}
 # Freeze on all available labeled weeks; no independent real holdout remains after earlier v4.1 inspection, so this artifact is DEVELOPMENT until new stress/external data passes.
 yf=meta.target.astype(int);models={n:fitc(clone(f),X,yf) for n,f in factories().items()};reg=HistGradientBoostingRegressor(max_iter=280,learning_rate=.025,max_leaf_nodes=10,min_samples_leaf=16,l2_regularization=10.,random_state=SEED).fit(X,meta.future_ratio.clip(0,2.5));joblib.dump({'version':VERSION,'status':'DEVELOPMENT_FROZEN_PENDING_INDEPENDENT_STRESS','feature_columns':list(X.columns),'models':models,'regressor':reg,'score_variant':best_name,'watch_threshold':best['threshold'],'red_threshold':red,'target_definition':'next merchant category week <80% trailing4 mean','sama_external_signal':'previous-origin leakage-safe SAMA sector forecast only'},MODEL)
 rep={'version':VERSION,'status':'DEVELOPMENT_FROZEN_PENDING_INDEPENDENT_STRESS','scientific_boundary':'UCI-derived Saudi-localized synthetic merchant microdata; real SAMA aggregate external context. v4.1 outcomes were already inspected, therefore v4.2 uses rolling-origin OOF as development evidence and does not call any historical period an untouched final test.','source_rows':srows,'eligible_rows':erows,'weekly_panel_rows':len(panel),'supervised_rows':len(meta),'categories':meta.category.nunique(),'feature_count':X.shape[1],'folds':foldmeta,'candidate_oof':cands,'selected':{'score_variant':best_name,'watch_threshold':best['threshold'],'red_threshold':red,'metrics':best['metrics'],'red_metrics':redm,'red_candidates':nred,'worst_fold_recall':worst,'per_fold':perfolds},'ablation':{'merchant_only_logistic_roc_auc':merchant_auc,'selected_hybrid_roc_auc':hybrid_auc,'delta_auc':hybrid_auc-merchant_auc},'contract':contract,'gates':gates,'all_development_gates_passed':bool(all(gates.values())),'next_required_evidence':'Frozen independent stress patterns not used in this development, followed by real merchant external validation when available.'};REPORT.write_text(json.dumps(rep,indent=2),encoding='utf-8');SUMMARY.write_text('# Sales Sentinel v4.2 — Rich Merchant + SAMA\n\n'+f"- Source rows **{srows:,}**\n- Rich features **{X.shape[1]}**\n- Rolling OOF ROC-AUC **{hybrid_auc:.2%}**\n- Balanced Accuracy **{best['metrics']['balanced_accuracy']:.2%}**\n- Recall **{best['metrics']['recall']:.2%}**\n- Precision **{best['metrics']['precision']:.2%}**\n- GREEN NPV **{best['metrics']['green_npv']:.2%}**\n- Alert rate **{best['metrics']['alert_rate']:.2%}**\n- Merchant-only logistic AUC **{merchant_auc:.2%}**\n- Hybrid delta AUC **{hybrid_auc-merchant_auc:+.2%}**\n- Development gates **{all(gates.values())}**\n- Status **DEVELOPMENT_FROZEN_PENDING_INDEPENDENT_STRESS**\n",encoding='utf-8');print(json.dumps(rep,indent=2))

if __name__=='__main__':main()
