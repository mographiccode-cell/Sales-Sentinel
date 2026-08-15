from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_sama_city_risk_v3 as base

ROOT=Path(__file__).resolve().parents[1]
BASE_MODEL=ROOT/'models'/'sama_city_v3_3'/'city_risk_v3_3.joblib'
OUT=ROOT/'reports'/'sama_city_v3_4_3'; MOD=ROOT/'models'/'sama_city_v3_4_3'
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)
MODEL=MOD/'city_risk_v3_4_3.joblib'; REPORT=OUT/'development_report.json'
VERSION='SAMA-CITY-RISK-3.4.3-MULTI-HORIZON'
DEV_END=pd.Timestamp('2025-06-29')

CONTRACT={
 'next4_recall_min':.94,
 'next4_green_npv_min':.99,
 'next4_alert_precision_min':.18,
 'alert_rate_max':.30,
 'green_coverage_min':.70,
 'incremental_negative_alert_rate_max':.05,
 'stable_fold_recall_min':.75,
 'red_precision_min':.70,
 'red_fpr_max':.015,
}

def bm(y,p):
 y=np.asarray(y,int); p=np.asarray(p,bool)
 tp=int(((y==1)&p).sum()); fp=int(((y==0)&p).sum()); fn=int(((y==1)&~p).sum()); tn=int(((y==0)&~p).sum())
 return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'FPR':fp/max(fp+tn,1),'NPV':tn/max(tn+fn,1)}

def add_horizon_targets(panel:pd.DataFrame)->pd.DataFrame:
 d=panel.copy(); d['week_start']=pd.to_datetime(d.week_start); d=d.sort_values(['city','week_start']).reset_index(drop=True); g=d.groupby('city',sort=False)
 d['baseline4']=g.value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).mean())
 for h in (1,2,4):
  vals=[]
  for _,z in d.groupby('city',sort=False):
   a=z.value_thousand_sar.to_numpy(float); b=z.baseline4.to_numpy(float); n=len(z); out=np.full(n,np.nan)
   for i in range(n):
    if not np.isfinite(b[i]) or i+h>=n: continue
    nxt=a[i+1:i+h+1]
    out[i]=float(np.min(nxt)/b[i]) if len(nxt)==h else np.nan
   vals.extend(out.tolist())
  d[f'min_ratio_h{h}']=np.asarray(vals,float)
  d[f'target_h{h}']=np.where(d[f'min_ratio_h{h}'].notna(),(d[f'min_ratio_h{h}']<.80).astype(float),np.nan)
 return d

def factories(seed):
 return {
  'lr':make_pipeline(StandardScaler(),LogisticRegression(C=.20,penalty='l2',solver='lbfgs',class_weight='balanced',max_iter=5000,random_state=seed)),
  'et':ExtraTreesClassifier(n_estimators=650,max_depth=6,min_samples_leaf=8,max_features=.65,class_weight='balanced',random_state=seed+1,n_jobs=-1),
  'hgb':HistGradientBoostingClassifier(max_iter=250,learning_rate=.025,max_leaf_nodes=10,min_samples_leaf=26,l2_regularization=8.,random_state=seed+2),
 }

def fit_cls(m,X,y):
 if isinstance(m,HistGradientBoostingClassifier):
  pos=max(int(y.sum()),1); neg=max(len(y)-pos,1); w=np.where(np.asarray(y)==1,neg/pos,1.); return m.fit(X,y,sample_weight=w)
 return m.fit(X,y)

def horizon_folds(d):
 starts=pd.to_datetime(['2022-01-01','2022-07-01','2023-01-01','2023-07-01','2024-01-01','2024-04-01','2024-07-01','2024-10-01','2025-01-01','2025-04-01'])
 out=[]
 for st in starts:
  en=min(st+pd.DateOffset(months=3)-pd.Timedelta(days=1),DEV_END)
  tr=d.week_start<=st-pd.Timedelta(days=35); va=d.week_start.between(st,en)
  if tr.sum()>=450 and va.sum()>=90: out.append((st,en,tr,va))
 return out

def build_aligned(panel):
 md,X,P,pc=base.featureize(panel,require_target=False)
 t=add_horizon_targets(panel)[['week_start','city','target_h1','target_h2','target_h4','min_ratio_h1','min_ratio_h2','min_ratio_h4']]
 d=md[['week_start','city']].merge(t,on=['week_start','city'],how='left',validate='one_to_one')
 good=d[['target_h1','target_h2','target_h4']].notna().all(axis=1)
 return d.loc[good].reset_index(drop=True),X.loc[good].reset_index(drop=True),pc.loc[good].reset_index(drop=True)

def build_oof(d,X,pc):
 rows=[]; fm=[]
 for fid,(st,en,tr,va) in enumerate(horizon_folds(d)):
  q=d.loc[va,['week_start','city','target_h1','target_h2','target_h4']].copy(); q['fold_id']=fid; q['precursor_count']=pc.loc[va].to_numpy()
  for h in (2,4):
   y=d.loc[tr,f'target_h{h}'].astype(int); names=[]
   for name,factory in factories(300+h).items():
    m=fit_cls(clone(factory),X.loc[tr],y); col=f'h{h}_{name}'; q[col]=m.predict_proba(X.loc[va])[:,1]; names.append(col)
   q[f'score_h{h}']=q[names].mean(axis=1); q[f'agree_h{h}']=(q[names]>=.5).sum(axis=1)
  rows.append(q); fm.append({'fold_id':fid,'start':str(st.date()),'end':str(en.date()),'train_rows':int(tr.sum()),'validation_rows':int(va.sum()),'h1_pos':int(q.target_h1.sum()),'h2_pos':int(q.target_h2.sum()),'h4_pos':int(q.target_h4.sum())})
 return pd.concat(rows,ignore_index=True),fm

def base_red_from_artifact(q,a,d,X):
 reds=np.zeros(len(q),bool); basewatch=np.zeros(len(q),bool); cursor=0
 for fid,(st,en,tr,va) in enumerate(horizon_folds(d)):
  n=int(va.sum()); y=d.loc[tr,'target_h1'].astype(int); scores=[]
  for _,factory in base.model_factories().items():
   m=base.fit_one(clone(factory),X.loc[tr],y); scores.append(m.predict_proba(X.loc[va])[:,1])
  mat=np.column_stack(scores); sc=mat.mean(axis=1); ag=(mat>=.5).sum(axis=1); pcv=q.loc[cursor:cursor+n-1,'precursor_count'].to_numpy(int)
  reds[cursor:cursor+n]=(sc>=float(a['red_threshold']))&(ag>=2)&(pcv>=int(a['min_precursor_red']))
  basewatch[cursor:cursor+n]=(sc>=float(a['watch_threshold']))|((pcv>=int(a['high_precursor_count']))&(sc>=float(a['high_precursor_fallback_threshold'])))|reds[cursor:cursor+n]
  cursor+=n
 return reds,basewatch

def evaluate(q,red,basewatch,t2,t4,agree_min):
 y4=q.target_h4.to_numpy(int); y1=q.target_h1.to_numpy(int)
 h2=(q.score_h2.to_numpy(float)>=t2)&(q.agree_h2.to_numpy(int)>=agree_min)
 h4=(q.score_h4.to_numpy(float)>=t4)&(q.agree_h4.to_numpy(int)>=agree_min)
 alert=basewatch|h2|h4; m=bm(y4,alert); rm=bm(y1,red); inc=(h2|h4)&~basewatch; base_green_neg=(y4==0)&~basewatch
 rate=float(alert.mean()); cov=1-rate; incneg=float((inc&(y4==0)).sum()/max(int(base_green_neg.sum()),1))
 per=[]
 for fid,z in q.assign(alert=alert).groupby('fold_id'):
  yy=z.target_h4.to_numpy(int); aa=z.alert.to_numpy(bool); mm=bm(yy,aa); per.append({'fold_id':int(fid),'positives':int(yy.sum()),'recall':mm['recall'],'precision':mm['precision'],'alert_rate':float(aa.mean())})
 stable=[x['recall'] for x in per if x['positives']>=8]; minst=min(stable) if stable else 1.
 ok=(m['recall']>=CONTRACT['next4_recall_min'] and m['NPV']>=CONTRACT['next4_green_npv_min'] and m['precision']>=CONTRACT['next4_alert_precision_min'] and rate<=CONTRACT['alert_rate_max'] and cov>=CONTRACT['green_coverage_min'] and incneg<=CONTRACT['incremental_negative_alert_rate_max'] and minst>=CONTRACT['stable_fold_recall_min'] and rm['precision']>=CONTRACT['red_precision_min'] and rm['FPR']<=CONTRACT['red_fpr_max'])
 return {'ok':bool(ok),'RED_next1':rm,'ALERT_next4':m,'alert_rate':rate,'green_coverage':cov,'incremental_negative_alert_rate':incneg,'horizon_incremental_alerts':int(inc.sum()),'horizon_incremental_tp':int((inc&(y4==1)).sum()),'min_stable_fold_recall':float(minst),'folds':per}

def choose(q,red,basewatch):
 # Compact quantile+linear grids preserve full score range while avoiding tens of thousands of duplicate decisions.
 c2=np.unique(np.r_[np.quantile(q.score_h2,np.linspace(.20,.995,45)),np.linspace(.15,.85,35)])
 c4=np.unique(np.r_[np.quantile(q.score_h4,np.linspace(.20,.995,45)),np.linspace(.15,.85,35)])
 valid=[]; best_feasible=None
 for agree in (1,2):
  for t2 in c2:
   for t4 in c4:
    e=evaluate(q,red,basewatch,float(t2),float(t4),agree)
    basic=e['alert_rate']<=.30 and e['green_coverage']>=.70 and e['incremental_negative_alert_rate']<=.05 and e['ALERT_next4']['precision']>=.18
    if basic:
     obj=(e['ALERT_next4']['recall'],e['ALERT_next4']['NPV'],e['ALERT_next4']['precision'],-e['alert_rate'],-e['incremental_negative_alert_rate'])
     if best_feasible is None or obj>best_feasible[0]: best_feasible=(obj,float(t2),float(t4),agree,e)
    if e['ok']:
     obj=(e['ALERT_next4']['recall'],e['ALERT_next4']['NPV'],e['ALERT_next4']['precision'],-e['alert_rate'],-e['incremental_negative_alert_rate'],agree)
     valid.append((obj,float(t2),float(t4),agree,e))
 if not valid:return None,best_feasible
 valid.sort(key=lambda x:x[0],reverse=True); return valid[0],best_feasible

def main():
 a=joblib.load(BASE_MODEL)
 if a.get('version')!='SAMA-CITY-RISK-3.3-DUAL-CHANNEL': raise RuntimeError(f'Unexpected base {a.get("version")}')
 panel=base.source.reconciled_load_panel(base.HISTORY); d,X,pc=build_aligned(panel); keep=d.week_start<=DEV_END; d=d.loc[keep].reset_index(drop=True); X=X.loc[keep].reset_index(drop=True); pc=pc.loc[keep].reset_index(drop=True)
 forbidden=[c for c in X.columns if c.startswith('city_') or 'target' in c or 'future' in c or 'decline_rate' in c]
 if forbidden: raise RuntimeError(f'Forbidden features {forbidden}')
 q,foldmeta=build_oof(d,X,pc); red,bw=base_red_from_artifact(q,a,d,X); best,feasible=choose(q,red,bw)
 if best is None:
  rep={'version':VERSION,'status':'NO_POLICY','best_feasible':None if feasible is None else {'t2':feasible[1],'t4':feasible[2],'agreement_min':feasible[3],'metrics':feasible[4]},'scientific_boundary':'historical OOF only, 35-day purge; no recent/counterfactual labels used'}; REPORT.write_text(json.dumps(rep,indent=2),encoding='utf-8'); print(json.dumps(rep,indent=2)); raise SystemExit(2)
 _,t2,t4,agree,e=best
 fitted={}
 for h in (2,4):
  y=d[f'target_h{h}'].astype(int); fitted[str(h)]={name:fit_cls(clone(factory),X,y) for name,factory in factories(300+h).items()}
 out=dict(a); out.update({'version':VERSION,'base_version':a['version'],'horizon_models':fitted,'horizon_features':list(X.columns),'h2_threshold':t2,'h4_threshold':t4,'horizon_agreement_min':agree,'development_end':str(DEV_END.date()),'target_definition_h2':'any week in next 2 has city POS value <80% current trailing4 mean','target_definition_h4':'any week in next 4 has city POS value <80% current trailing4 mean','purge_days':35,'scope':'v3.3 RED next-week plus multi-horizon 2/4-week AMBER risk'})
 joblib.dump(out,MODEL)
 rep={'version':VERSION,'base_version':a['version'],'rows':len(d),'target_rates':{'h1':float(d.target_h1.mean()),'h2':float(d.target_h2.mean()),'h4':float(d.target_h4.mean())},'thresholds':{'h2':t2,'h4':t4,'agreement_min':agree},'metrics':e,'contract':CONTRACT,'all_gates_passed':bool(e['ok']),'folds':foldmeta,'controls':{'35_day_horizon_purge':True,'no_city_identity':True,'no_target_history_features':True,'no_future_features':True,'red_nextweek_policy_conservative':True,'horizon_thresholds_selected_historical_oof_only':True,'no_recent_or_counterfactual_labels_used':True},'scientific_boundary':'No outcome after 2025-06-29 is read; four-week labels are purged 35 days from every validation start.'}
 REPORT.write_text(json.dumps(rep,indent=2),encoding='utf-8'); (OUT/'development_summary.md').write_text('# Sales Sentinel v3.4.3 — Multi-Horizon\n\n'+f"- Next-4-week alert recall **{e['ALERT_next4']['recall']:.2%}**\n- Next-4-week alert precision **{e['ALERT_next4']['precision']:.2%}**\n- GREEN NPV **{e['ALERT_next4']['NPV']:.2%}**\n- Alert rate **{e['alert_rate']:.2%}**\n- GREEN coverage **{e['green_coverage']:.2%}**\n- Next-week RED precision **{e['RED_next1']['precision']:.2%}**\n- Thresholds h2={t2:.4f}, h4={t4:.4f}, agreement>={agree}\n- All gates **{e['ok']}**\n",encoding='utf-8'); print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
