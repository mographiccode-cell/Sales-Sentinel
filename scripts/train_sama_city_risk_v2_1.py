from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

ROOT=Path(__file__).resolve().parents[1]
CITY=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2025.csv'
NV=ROOT/'data'/'sama_pos'/'sama_pos_national_weekly_value_2020_2025.csv'
NC=ROOT/'data'/'sama_pos'/'sama_pos_national_weekly_count_2020_2025.csv'
OUT=ROOT/'reports'/'sama_city_v2_1'; MOD=ROOT/'models'/'sama_city_v2_1'
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'development_report.json'; SUMMARY=OUT/'development_summary.md'; MODEL=MOD/'city_market_risk_v2_1.joblib'
VERSION='SAMA-CITY-MARKET-RISK-2.1-FROZEN'; SEED=42; DECLINE=.20
DEVELOPMENT_END=pd.Timestamp('2025-06-29')  # next-week target is 2025-07-06, the last old-source week
SELECT_END=pd.Timestamp('2024-12-31'); POLICY_START=pd.Timestamp('2025-01-01'); POLICY_END=DEVELOPMENT_END
ACCEPTANCE={'red_precision_min':.70,'red_false_positive_rate_max':.05,'alert_recall_min':.90,'green_npv_min':.98,'roc_auc_min':.85,'pr_auc_min':.40}


def sigmoid(x): return 1/(1+np.exp(np.clip(-np.asarray(x,float),-35,35)))

def load_panel(path=CITY,nv_path=NV,nc_path=NC):
    d=pd.read_csv(path,parse_dates=['week_start','week_end']).sort_values(['city','week_start']).reset_index(drop=True)
    nv=pd.read_csv(nv_path,parse_dates=['week_start'])[['week_start','value_thousand_sar']].rename(columns={'value_thousand_sar':'national_value'})
    nc=pd.read_csv(nc_path,parse_dates=['week_start'])[['week_start','transaction_count']].rename(columns={'transaction_count':'national_count'})
    n=nv.merge(nc,on='week_start',how='inner').sort_values('week_start')
    d=d.merge(n,on='week_start',how='left',validate='many_to_one')
    if d[['national_value','national_count']].isna().any().any(): raise RuntimeError('National context missing for city panel')
    return d


def featureize(d):
    d=d.copy().sort_values(['city','week_start']).reset_index(drop=True); g=d.groupby('city',sort=False)
    F=pd.DataFrame(index=d.index)
    for col,pre in [('value_thousand_sar','value'),('transaction_count_thousand','count')]:
        s=d[col].astype(float); F[f'log_{pre}_t0']=np.log1p(s)
        for lag in (1,2,3,4,8,13,26,52): F[f'log_{pre}_lag_{lag}']=np.log1p(g[col].shift(lag))
        for w in (4,8,13,26,52):
            F[f'log_{pre}_mean_{w}']=g[col].transform(lambda x,w=w:np.log1p(x).shift(1).rolling(w,min_periods=w).mean())
            F[f'log_{pre}_std_{w}']=g[col].transform(lambda x,w=w:np.log1p(x).shift(1).rolling(w,min_periods=w).std())
        F[f'{pre}_change_1']=g[col].pct_change(1); F[f'{pre}_change_4']=g[col].pct_change(4); F[f'{pre}_change_13']=g[col].pct_change(13)
    # National context: current completed week is known at the prediction origin.
    nd=d[['week_start','national_value','national_count']].drop_duplicates('week_start').sort_values('week_start').copy()
    for col,pre in [('national_value','nvalue'),('national_count','ncount')]:
        nd[f'log_{pre}_t0']=np.log1p(nd[col])
        for lag in (1,2,4,8,13,26,52): nd[f'log_{pre}_lag_{lag}']=np.log1p(nd[col].shift(lag))
        for w in (4,13,26,52):
            nd[f'log_{pre}_mean_{w}']=np.log1p(nd[col]).shift(1).rolling(w,min_periods=w).mean()
            nd[f'{pre}_change_{w}']=nd[col].pct_change(w)
    nfeatures=[c for c in nd.columns if c not in {'week_start','national_value','national_count'}]
    nd=nd[['week_start']+nfeatures]; dd=d[['week_start']].merge(nd,on='week_start',how='left',validate='many_to_one')
    F=pd.concat([F,dd[nfeatures]],axis=1)
    week=d.week_start.dt.isocalendar().week.astype(float); F['week_sin']=np.sin(2*np.pi*week/52.18); F['week_cos']=np.cos(2*np.pi*week/52.18)
    F=pd.concat([F,pd.get_dummies(d.city,prefix='city',dtype=float)],axis=1)
    # Target: next official city week vs baseline of the four completed weeks including origin.
    baseline=g.value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).mean())
    actual_next=g.value_thousand_sar.shift(-1)
    d['baseline4']=baseline; d['actual_next_value']=actual_next; d['future_ratio']=actual_next/baseline.replace(0,np.nan); d['target']=(d.future_ratio < 1-DECLINE).astype(int)
    F=F.replace([np.inf,-np.inf],np.nan); good=F.notna().all(axis=1)&d.future_ratio.notna()
    return d.loc[good].reset_index(drop=True),F.loc[good].reset_index(drop=True)


def cls_models(pos_weight):
    return {
        'Logistic':make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=3000,class_weight='balanced',random_state=SEED)),
        'ExtraTrees':ExtraTreesClassifier(n_estimators=800,max_depth=8,min_samples_leaf=4,max_features=.7,class_weight='balanced',random_state=SEED,n_jobs=-1),
        'HistGB':HistGradientBoostingClassifier(max_iter=300,learning_rate=.035,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=4,random_state=SEED),
        'XGBoost':XGBClassifier(n_estimators=400,max_depth=3,learning_rate=.025,subsample=.82,colsample_bytree=.8,min_child_weight=7,reg_lambda=5,reg_alpha=.3,scale_pos_weight=pos_weight,eval_metric='logloss',random_state=SEED,n_jobs=-1),
    }

def reg_models():
    return {
        'Ridge':make_pipeline(StandardScaler(),Ridge(alpha=10.0)),
        'ExtraTreesReg':ExtraTreesRegressor(n_estimators=700,max_depth=10,min_samples_leaf=3,max_features=.75,random_state=SEED,n_jobs=-1),
        'HistGBReg':HistGradientBoostingRegressor(max_iter=320,learning_rate=.035,max_leaf_nodes=16,min_samples_leaf=18,l2_regularization=3,random_state=SEED),
        'XGBoostReg':XGBRegressor(n_estimators=420,max_depth=3,learning_rate=.025,subsample=.82,colsample_bytree=.8,min_child_weight=7,reg_lambda=5,reg_alpha=.3,objective='reg:squarederror',random_state=SEED,n_jobs=-1),
    }

def rank(y,s): return {'ROC_AUC':float(roc_auc_score(y,s)),'PR_AUC':float(average_precision_score(y,s))}

def cal(y,p): return {'Brier':float(brier_score_loss(y,np.clip(p,1e-6,1-1e-6)))}

def folds(d):
    starts=pd.to_datetime(['2023-01-01','2023-07-01','2024-01-01','2024-04-01','2024-07-01','2024-10-01','2025-01-01','2025-04-01']); out=[]
    for start in starts:
        end=min(start+pd.DateOffset(months=3)-pd.Timedelta(days=1),DEVELOPMENT_END)
        tr=d.week_start<=start-pd.Timedelta(days=14); va=d.week_start.between(start,end)
        if tr.sum()>=800 and va.sum()>=100 and d.loc[tr,'target'].nunique()==2 and d.loc[va,'target'].nunique()==2: out.append((start,end,tr,va))
    return out

def oof(d,X):
    frames=[]; meta=[]
    for start,end,tr,va in folds(d):
        y=d.loc[tr,'target']; pos=int(y.sum()); neg=len(y)-pos; scores={}
        for name,m in cls_models(neg/max(pos,1)).items(): scores[name]=clone(m).fit(X.loc[tr],y).predict_proba(X.loc[va])[:,1]
        for name,m in reg_models().items():
            fit=clone(m).fit(X.loc[tr],np.log1p(d.loc[tr,'actual_next_value']))
            pred=np.expm1(fit.predict(X.loc[va])); ratio=pred/d.loc[va,'baseline4'].to_numpy(); scores[name]=sigmoid(((1-DECLINE)-ratio)/.055)
        # transparent baselines
        current_ratio=d.loc[va,'value_thousand_sar'].to_numpy()/d.loc[va,'baseline4'].to_numpy(); scores['CurrentWeekRule']=sigmoid(((1-DECLINE)-current_ratio)/.055)
        q=pd.DataFrame({'week_start':d.loc[va,'week_start'].to_numpy(),'y':d.loc[va,'target'].to_numpy()})
        for name,s in scores.items(): q[name]=s
        frames.append(q); meta.append({'start':str(start.date()),'end':str(end.date()),'train_rows':int(tr.sum()),'validation_rows':int(va.sum()),'train_rate':float(y.mean()),'validation_rate':float(d.loc[va,'target'].mean())})
    if not frames: raise RuntimeError('No city walk-forward folds')
    return pd.concat(frames,ignore_index=True).sort_values('week_start').reset_index(drop=True),meta

def binary(y,p,t):
    y=np.asarray(y,int); pred=np.asarray(p)>=t; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'Precision':tp/max(tp+fp,1),'Recall':tp/max(tp+fn,1),'FPR':fp/max(fp+tn,1),'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn)}

def choose_policy(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float); cand=np.unique(np.r_[np.linspace(.005,.995,199),np.quantile(p,np.linspace(0,1,101))])
    watch=[]
    for t in cand:
        alert=p>=t; tp=((y==1)&alert).sum(); fp=((y==0)&alert).sum(); fn=((y==1)&~alert).sum(); tn=((y==0)&~alert).sum()
        rec=tp/max(tp+fn,1); prec=tp/max(tp+fp,1); npv=tn/max(tn+fn,1)
        if rec>=ACCEPTANCE['alert_recall_min'] and npv>=ACCEPTANCE['green_npv_min']: watch.append((prec,-alert.mean(),npv,float(t)))
    if not watch: raise RuntimeError('No WATCH threshold satisfies frozen recall/NPV development contract')
    wt=max(watch)[3]
    reds=[]
    for t in cand[cand>=wt]:
        m=binary(y,p,t); alerts=m['TP']+m['FP']
        if alerts>=8 and m['Precision']>=ACCEPTANCE['red_precision_min'] and m['FPR']<=ACCEPTANCE['red_false_positive_rate_max']: reds.append((m['Recall'],m['Precision'],-m['FPR'],float(t)))
    if not reds: raise RuntimeError('No RED threshold satisfies frozen development contract')
    return wt,max(reds)[3]

def triage(y,p,wt,rt):
    y=np.asarray(y,int); p=np.asarray(p,float); state=np.where(p>=rt,'RED',np.where(p>=wt,'AMBER','GREEN')); red=state=='RED'; green=state=='GREEN'; alert=~green
    def pp(mask): return ((y==1)&mask).sum()/max(mask.sum(),1)
    pos=(y==1).sum(); neg=(y==0).sum(); redtp=((y==1)&red).sum(); redfp=((y==0)&red).sum(); atp=((y==1)&alert).sum(); afp=((y==0)&alert).sum(); gtn=((y==0)&green).sum(); gfn=((y==1)&green).sum()
    return {'RED':{'rows':int(red.sum()),'precision':float(pp(red)),'FPR':float(redfp/max(neg,1)),'recall_contribution':float(redtp/max(pos,1))},'AMBER':{'rows':int((state=='AMBER').sum()),'positive_rate':float(pp(state=='AMBER'))},'GREEN':{'rows':int(green.sum()),'NPV':float(gtn/max(gtn+gfn,1)),'missed_declines':int(gfn),'miss_rate':float(gfn/max(pos,1))},'RED_plus_AMBER':{'rows':int(alert.sum()),'precision':float(atp/max(atp+afp,1)),'recall':float(atp/max(pos,1))}}

def main():
    d,X=featureize(load_panel()); keep=d.week_start<=DEVELOPMENT_END; d=d.loc[keep].reset_index(drop=True); X=X.loc[keep].reset_index(drop=True)
    oo,meta=oof(d,X); sel=oo[oo.week_start<=SELECT_END]; pol=oo[oo.week_start.between(POLICY_START,POLICY_END)]
    names=[c for c in oo.columns if c not in {'week_start','y'}]; metrics={n:rank(sel.y,sel[n]) for n in names}; selected=max(names,key=lambda n:(metrics[n]['PR_AUC'],metrics[n]['ROC_AUC']))
    calibrator=LogisticRegression(max_iter=2000,random_state=SEED).fit(sel[[selected]],sel.y); pp=calibrator.predict_proba(pol[[selected]])[:,1]
    wt,rt=choose_policy(pol.y,pp); t=triage(pol.y,pp,wt,rt); ranking=rank(pol.y,pp); calibration=cal(pol.y,pp)
    gates={'red_precision':t['RED']['precision']>=ACCEPTANCE['red_precision_min'],'red_fpr':t['RED']['FPR']<=ACCEPTANCE['red_false_positive_rate_max'],'alert_recall':t['RED_plus_AMBER']['recall']>=ACCEPTANCE['alert_recall_min'],'green_npv':t['GREEN']['NPV']>=ACCEPTANCE['green_npv_min'],'roc_auc':ranking['ROC_AUC']>=ACCEPTANCE['roc_auc_min'],'pr_auc':ranking['PR_AUC']>=ACCEPTANCE['pr_auc_min']}
    # Fit only selected model on all development data; holdout is not read here.
    pos=int(d.target.sum()); neg=len(d)-pos; models={}; kind='rule'
    if selected in cls_models(neg/max(pos,1)): models[selected]=clone(cls_models(neg/max(pos,1))[selected]).fit(X,d.target); kind='classifier'
    elif selected in reg_models(): models[selected]=clone(reg_models()[selected]).fit(X,np.log1p(d.actual_next_value)); kind='regressor'
    artifact={'version':VERSION,'selected':selected,'selected_kind':kind,'models':models,'calibrator':calibrator,'features':list(X.columns),'watch_threshold':wt,'red_threshold':rt,'decline_threshold':DECLINE,'acceptance':ACCEPTANCE,'development_end':str(DEVELOPMENT_END.date())}
    joblib.dump(artifact,MODEL)
    report={'version':VERSION,'scientific_boundary':'SAMA City Total + National Total only; no sector taxonomy features. Fresh post-development PDFs are not read.','development_rows':len(d),'cities':int(d.city.nunique()),'decline_rate':float(d.target.mean()),'folds':meta,'candidate_selection_end':str(SELECT_END.date()),'policy_window':f'{POLICY_START.date()}..{POLICY_END.date()}','candidate_metrics':metrics,'selected':selected,'selected_kind':kind,'thresholds':{'watch':wt,'red':rt},'policy_triage':t,'policy_ranking':ranking,'policy_calibration':calibration,'acceptance':ACCEPTANCE,'gates':gates,'all_gates_passed':bool(all(gates.values())),'leakage_controls':{'chronological_expanding_folds':True,'one_week_purge':True,'shuffle':False,'future_values_as_features':False,'fresh_2025_2026_holdout_used':False,'selection_and_policy_windows_separated':True}}
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8'); SUMMARY.write_text(f'''# SAMA City Market Risk v2.1 — Frozen Development\n\n- Cities: **{d.city.nunique()}**\n- Development rows: **{len(d):,}**\n- Decline rate: **{d.target.mean():.2%}**\n- Selected: **{selected}** ({kind})\n- RED precision: **{t['RED']['precision']:.2%}**\n- RED FPR: **{t['RED']['FPR']:.2%}**\n- RED+AMBER recall: **{t['RED_plus_AMBER']['recall']:.2%}**\n- GREEN NPV: **{t['GREEN']['NPV']:.2%}**\n- PR-AUC: **{ranking['PR_AUC']:.2%}**\n- ROC-AUC: **{ranking['ROC_AUC']:.2%}**\n- All development gates: **{report['all_gates_passed']}**\n- Fresh holdout used: **No**\n''',encoding='utf-8')
    print(json.dumps({'selected':selected,'kind':kind,'rows':len(d),'triage':t,'ranking':ranking,'gates':gates,'all':report['all_gates_passed']},indent=2))

if __name__=='__main__': main()
