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
import train_sama_city_risk_v3_4_3 as mh

ROOT=Path(__file__).resolve().parents[1]
BASE_MODEL=ROOT/'models'/'sama_city_v3_3'/'city_risk_v3_3.joblib'
OUT=ROOT/'reports'/'sama_city_v3_4_4'; MOD=ROOT/'models'/'sama_city_v3_4_4'
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)
MODEL=MOD/'city_risk_v3_4_4.joblib'; REPORT=OUT/'development_report.json'
VERSION='SAMA-CITY-RISK-3.4.4-SAUDI-SEASONAL-MULTIHORIZON'
DEV_END=pd.Timestamp('2025-06-29')
CONTRACT=dict(mh.CONTRACT)

RAMADAN=[('2020-04-24','2020-05-23'),('2021-04-13','2021-05-12'),('2022-04-02','2022-05-01'),('2023-03-23','2023-04-20'),('2024-03-11','2024-04-09'),('2025-03-01','2025-03-29'),('2026-02-18','2026-03-19')]
EID_FITR=['2020-05-24','2021-05-13','2022-05-02','2023-04-21','2024-04-10','2025-03-30','2026-03-20']
EID_ADHA=['2020-07-31','2021-07-20','2022-07-09','2023-06-28','2024-06-16','2025-06-06','2026-05-27']

def known_window(week_start,starts_ends):
    w=pd.to_datetime(week_start)
    out=np.zeros(len(w),dtype=bool)
    for s,e in starts_ends:
        lo=pd.Timestamp(s); hi=pd.Timestamp(e)
        out |= ((w<=hi)&((w+pd.Timedelta(days=6))>=lo)).to_numpy(dtype=bool)
    return out.astype(float)

def event_window(week_start,dates,before=7,after=7):
    pairs=[(str((pd.Timestamp(x)-pd.Timedelta(days=before)).date()),str((pd.Timestamp(x)+pd.Timedelta(days=after)).date())) for x in dates]
    return known_window(week_start,pairs)

def pre_ramadan_pairs():
    return [(str((pd.Timestamp(s)-pd.Timedelta(days=14)).date()),str((pd.Timestamp(s)-pd.Timedelta(days=1)).date())) for s,_ in RAMADAN]

def post_ramadan_pairs():
    return [(str((pd.Timestamp(e)+pd.Timedelta(days=1)).date()),str((pd.Timestamp(e)+pd.Timedelta(days=14)).date())) for _,e in RAMADAN]

def augment(panel:pd.DataFrame, meta:pd.DataFrame, X:pd.DataFrame)->pd.DataFrame:
    p=panel.copy(); p['week_start']=pd.to_datetime(p.week_start); p=p.sort_values(['city','week_start']).reset_index(drop=True); g=p.groupby('city',sort=False)
    A=pd.DataFrame({'week_start':p.week_start,'city':p.city.astype(str)})
    for col,pre in [('value_thousand_sar','value'),('transaction_count_thousand','count')]:
        lag52=g[col].shift(52)
        A[f'{pre}_ratio_lag52']=p[col].astype(float)/lag52.replace(0,np.nan)
        A[f'{pre}_change_52']=g[col].pct_change(52)
    ticket=p.value_thousand_sar/p.transaction_count_thousand.replace(0,np.nan)
    A['ticket_change_52']=ticket.groupby(p.city,sort=False).pct_change(52)
    nat=p.groupby('week_start',as_index=False).agg(nvalue=('value_thousand_sar','sum'),ncount=('transaction_count_thousand','sum')).sort_values('week_start')
    nat['nvalue_change_52']=nat.nvalue.pct_change(52); nat['ncount_change_52']=nat.ncount.pct_change(52)
    A=A.merge(nat[['week_start','nvalue_change_52','ncount_change_52']],on='week_start',how='left',validate='many_to_one')
    w=A.week_start
    A['saudi_ramadan']=known_window(w,RAMADAN)
    A['saudi_pre_ramadan_2w']=known_window(w,pre_ramadan_pairs())
    A['saudi_post_ramadan_2w']=known_window(w,post_ramadan_pairs())
    A['saudi_eid_fitr_window']=event_window(w,EID_FITR,before=7,after=10)
    A['saudi_hajj_eid_adha_window']=event_window(w,EID_ADHA,before=10,after=10)
    A['saudi_national_day_window']=event_window(w,[f'{y}-09-23' for y in range(2020,2027)],before=7,after=7)
    A['saudi_founding_day_window']=event_window(w,[f'{y}-02-22' for y in range(2022,2027)],before=7,after=7)
    A['saudi_salary_week']=w.dt.day.between(21,27).astype(float)
    key=meta[['week_start','city']].copy(); key['city']=key.city.astype(str)
    Z=key.merge(A,on=['week_start','city'],how='left',validate='one_to_one').drop(columns=['week_start','city'])
    return pd.concat([X.reset_index(drop=True),Z.reset_index(drop=True)],axis=1).replace([np.inf,-np.inf],np.nan)

def factories(seed):
    return {
      'lr':make_pipeline(StandardScaler(),LogisticRegression(C=.16,penalty='l2',solver='lbfgs',class_weight='balanced',max_iter=5000,random_state=seed)),
      'et':ExtraTreesClassifier(n_estimators=750,max_depth=7,min_samples_leaf=7,max_features=.60,class_weight='balanced',random_state=seed+1,n_jobs=-1),
      'hgb':HistGradientBoostingClassifier(max_iter=280,learning_rate=.022,max_leaf_nodes=12,min_samples_leaf=24,l2_regularization=10.,random_state=seed+2),
    }

def fit_cls(m,X,y):
    if isinstance(m,HistGradientBoostingClassifier):
      pos=max(int(y.sum()),1); neg=max(len(y)-pos,1); sw=np.where(np.asarray(y)==1,neg/pos,1.); return m.fit(X,y,sample_weight=sw)
    return m.fit(X,y)

def build_dataset(panel):
    md,X0,P,pc=base.featureize(panel,require_target=False)
    X=augment(panel,md,X0)
    t=mh.add_horizon_targets(panel)[['week_start','city','target_h1','target_h2','target_h4','min_ratio_h1','min_ratio_h2','min_ratio_h4']]
    d=md[['week_start','city']].merge(t,on=['week_start','city'],how='left',validate='one_to_one')
    good=d[['target_h1','target_h2','target_h4']].notna().all(axis=1)&X.notna().all(axis=1)
    return d.loc[good].reset_index(drop=True),X.loc[good].reset_index(drop=True),pc.loc[good].reset_index(drop=True)

def folds(d): return mh.horizon_folds(d)

def build_oof(d,X,pc):
    rows=[]; meta=[]
    for fid,(st,en,tr,va) in enumerate(folds(d)):
      q=d.loc[va,['week_start','city','target_h1','target_h2','target_h4']].copy(); q['fold_id']=fid; q['precursor_count']=pc.loc[va].to_numpy()
      for h in (2,4):
        y=d.loc[tr,f'target_h{h}'].astype(int); names=[]
        for name,factory in factories(400+h).items():
          m=fit_cls(clone(factory),X.loc[tr],y); c=f'h{h}_{name}'; q[c]=m.predict_proba(X.loc[va])[:,1]; names.append(c)
        q[f'score_h{h}']=q[names].mean(axis=1); q[f'agree_h{h}']=(q[names]>=.5).sum(axis=1)
      rows.append(q); meta.append({'fold_id':fid,'start':str(st.date()),'end':str(en.date()),'train_rows':int(tr.sum()),'validation_rows':int(va.sum()),'h1_pos':int(q.target_h1.sum()),'h2_pos':int(q.target_h2.sum()),'h4_pos':int(q.target_h4.sum())})
    return pd.concat(rows,ignore_index=True),meta

def base_red_watch(q,a,d,Xbase):
    red=np.zeros(len(q),bool); watch=np.zeros(len(q),bool); cursor=0; orig=a['features']; xb=Xbase[orig]
    for fid,(st,en,tr,va) in enumerate(folds(d)):
      n=int(va.sum()); y=d.loc[tr,'target_h1'].astype(int); ps=[]
      for _,factory in base.model_factories().items():
        m=base.fit_one(clone(factory),xb.loc[tr],y); ps.append(m.predict_proba(xb.loc[va])[:,1])
      mat=np.column_stack(ps); sc=mat.mean(axis=1); ag=(mat>=.5).sum(axis=1); pv=q.loc[cursor:cursor+n-1,'precursor_count'].to_numpy(int)
      red[cursor:cursor+n]=(sc>=float(a['red_threshold']))&(ag>=2)&(pv>=int(a['min_precursor_red']))
      watch[cursor:cursor+n]=red[cursor:cursor+n]|(sc>=float(a['watch_threshold']))|((pv>=int(a['high_precursor_count']))&(sc>=float(a['high_precursor_fallback_threshold'])))
      cursor+=n
    return red,watch

def evaluate(q,red,bw,t2,t4,agree):
    y4=q.target_h4.to_numpy(int); y1=q.target_h1.to_numpy(int)
    h2=(q.score_h2.to_numpy(float)>=t2)&(q.agree_h2.to_numpy(int)>=agree); h4=(q.score_h4.to_numpy(float)>=t4)&(q.agree_h4.to_numpy(int)>=agree)
    alert=bw|h2|h4; m=mh.bm(y4,alert); rm=mh.bm(y1,red); inc=(h2|h4)&~bw; bgn=(y4==0)&~bw; rate=float(alert.mean()); cov=1-rate; incneg=float((inc&(y4==0)).sum()/max(int(bgn.sum()),1))
    per=[]
    for fid,z in q.assign(alert=alert).groupby('fold_id'):
      yy=z.target_h4.to_numpy(int); aa=z.alert.to_numpy(bool); mm=mh.bm(yy,aa); per.append({'fold_id':int(fid),'positives':int(yy.sum()),'recall':mm['recall'],'precision':mm['precision'],'alert_rate':float(aa.mean())})
    stable=[x['recall'] for x in per if x['positives']>=8]; minst=min(stable) if stable else 1.
    ok=(m['recall']>=CONTRACT['next4_recall_min'] and m['NPV']>=CONTRACT['next4_green_npv_min'] and m['precision']>=CONTRACT['next4_alert_precision_min'] and rate<=CONTRACT['alert_rate_max'] and cov>=CONTRACT['green_coverage_min'] and incneg<=CONTRACT['incremental_negative_alert_rate_max'] and minst>=CONTRACT['stable_fold_recall_min'] and rm['precision']>=CONTRACT['red_precision_min'] and rm['FPR']<=CONTRACT['red_fpr_max'])
    return {'ok':bool(ok),'RED_next1':rm,'ALERT_next4':m,'alert_rate':rate,'green_coverage':cov,'incremental_negative_alert_rate':incneg,'horizon_incremental_alerts':int(inc.sum()),'horizon_incremental_tp':int((inc&(y4==1)).sum()),'min_stable_fold_recall':float(minst),'folds':per}

def choose(q,red,bw):
    c2=np.unique(np.r_[np.quantile(q.score_h2,np.linspace(.15,.995,55)),np.linspace(.12,.88,40)]); c4=np.unique(np.r_[np.quantile(q.score_h4,np.linspace(.15,.995,55)),np.linspace(.12,.88,40)])
    valid=[]; feasible=None
    for agree in (1,2):
      for t2 in c2:
        for t4 in c4:
          e=evaluate(q,red,bw,float(t2),float(t4),agree); basic=e['alert_rate']<=.30 and e['green_coverage']>=.70 and e['incremental_negative_alert_rate']<=.05 and e['ALERT_next4']['precision']>=.18
          if basic:
            obj=(e['ALERT_next4']['recall'],e['ALERT_next4']['NPV'],e['ALERT_next4']['precision'],-e['alert_rate'],-e['incremental_negative_alert_rate'])
            if feasible is None or obj>feasible[0]: feasible=(obj,float(t2),float(t4),agree,e)
          if e['ok']:
            obj=(e['ALERT_next4']['recall'],e['ALERT_next4']['NPV'],e['ALERT_next4']['precision'],-e['alert_rate'],-e['incremental_negative_alert_rate'],agree); valid.append((obj,float(t2),float(t4),agree,e))
    if not valid:return None,feasible
    valid.sort(key=lambda x:x[0],reverse=True); return valid[0],feasible

def main():
    a=joblib.load(BASE_MODEL)
    if a.get('version')!='SAMA-CITY-RISK-3.3-DUAL-CHANNEL':raise RuntimeError(f'Unexpected base {a.get("version")}')
    panel=base.source.reconciled_load_panel(base.HISTORY); d,X,pc=build_dataset(panel); keep=d.week_start<=DEV_END; d=d.loc[keep].reset_index(drop=True); X=X.loc[keep].reset_index(drop=True); pc=pc.loc[keep].reset_index(drop=True)
    forbidden=[c for c in X.columns if c.startswith('city_') or 'target' in c or 'future' in c or 'decline_rate' in c]
    if forbidden:raise RuntimeError(f'Forbidden {forbidden}')
    q,fm=build_oof(d,X,pc); red,bw=base_red_watch(q,a,d,X); best,feasible=choose(q,red,bw)
    if best is None:
      rep={'version':VERSION,'status':'NO_POLICY','feature_count':len(X.columns),'best_feasible':None if feasible is None else {'t2':feasible[1],'t4':feasible[2],'agreement_min':feasible[3],'metrics':feasible[4]},'scientific_boundary':'historical OOF only, 35-day purge; Saudi calendar and 52-week features known at origin; no recent/counterfactual labels used'}; REPORT.write_text(json.dumps(rep,indent=2),encoding='utf-8'); print(json.dumps(rep,indent=2)); raise SystemExit(2)
    _,t2,t4,agree,e=best; fitted={}
    for h in (2,4):
      y=d[f'target_h{h}'].astype(int); fitted[str(h)]={name:fit_cls(clone(factory),X,y) for name,factory in factories(400+h).items()}
    out=dict(a); out.update({'version':VERSION,'base_version':a['version'],'horizon_models':fitted,'horizon_features':list(X.columns),'h2_threshold':t2,'h4_threshold':t4,'horizon_agreement_min':agree,'development_end':str(DEV_END.date()),'purge_days':35,'saudi_calendar_features':True,'yoy_52_features':True,'scope':'v3.3 RED next-week plus Saudi-seasonal 2/4-week AMBER risk'})
    joblib.dump(out,MODEL)
    rep={'version':VERSION,'base_version':a['version'],'rows':len(d),'feature_count':len(X.columns),'target_rates':{'h1':float(d.target_h1.mean()),'h2':float(d.target_h2.mean()),'h4':float(d.target_h4.mean())},'thresholds':{'h2':t2,'h4':t4,'agreement_min':agree},'metrics':e,'contract':CONTRACT,'all_gates_passed':bool(e['ok']),'folds':fm,'controls':{'35_day_horizon_purge':True,'saudi_calendar_known_at_origin':True,'52_week_yoy_source_only':True,'no_city_identity':True,'no_target_history_features':True,'no_future_features':True,'no_recent_or_counterfactual_labels_used':True},'scientific_boundary':'No outcome after 2025-06-29 is read; calendar covariates deterministic from forecast date; four-week labels purged 35 days.'}
    REPORT.write_text(json.dumps(rep,indent=2),encoding='utf-8'); (OUT/'development_summary.md').write_text('# Sales Sentinel v3.4.4 — Saudi Seasonal Multi-Horizon\n\n'+f"- Next-4-week recall **{e['ALERT_next4']['recall']:.2%}**\n- Alert precision **{e['ALERT_next4']['precision']:.2%}**\n- GREEN NPV **{e['ALERT_next4']['NPV']:.2%}**\n- Alert rate **{e['alert_rate']:.2%}**\n- GREEN coverage **{e['green_coverage']:.2%}**\n- Next-week RED precision **{e['RED_next1']['precision']:.2%}**\n- All gates **{e['ok']}**\n",encoding='utf-8'); print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
