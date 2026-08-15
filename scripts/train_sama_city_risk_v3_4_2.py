from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

import train_sama_city_risk_v3 as base
import train_sama_city_risk_v3_4_1 as trend_helpers

ROOT=Path(__file__).resolve().parents[1]
BASE_MODEL=ROOT/'models'/'sama_city_v3_3'/'city_risk_v3_3.joblib'
OUT=ROOT/'reports'/'sama_city_v3_4_2'; MOD=ROOT/'models'/'sama_city_v3_4_2'
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)
MODEL=MOD/'city_risk_v3_4_2.joblib'; REPORT=OUT/'development_report.json'
VERSION='SAMA-CITY-RISK-3.4.2-DOWNSIDE-RATIO'
FEATURES=trend_helpers.TREND_FEATURES
CONTRACT={
    'alert_recall_min':.94,'green_npv_min':.992,'alert_precision_min':.18,
    'alert_rate_max':.30,'green_coverage_min':.70,
    'incremental_negative_alert_rate_max':.05,
    'min_recall_folds_with_5plus_positives':.70,
    'red_precision_min':.70,'red_fpr_max':.015,
}

def bm(y,p):
    y=np.asarray(y,int); p=np.asarray(p,bool)
    tp=int(((y==1)&p).sum()); fp=int(((y==0)&p).sum()); fn=int(((y==1)&~p).sum()); tn=int(((y==0)&~p).sum())
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'FPR':fp/max(fp+tn,1),'NPV':tn/max(tn+fn,1)}

def reg_factories():
    return {
        'mean_hgb': HistGradientBoostingRegressor(max_iter=280,learning_rate=.035,max_leaf_nodes=12,min_samples_leaf=24,l2_regularization=6.,loss='squared_error',random_state=241),
        'q25_hgb': HistGradientBoostingRegressor(max_iter=280,learning_rate=.035,max_leaf_nodes=12,min_samples_leaf=24,l2_regularization=6.,loss='quantile',quantile=.25,random_state=242),
        'q25_gbr': GradientBoostingRegressor(n_estimators=260,learning_rate=.025,max_depth=2,min_samples_leaf=18,subsample=.85,loss='quantile',alpha=.25,random_state=243),
    }

def fit_reg(m,X,y):
    return m.fit(X,y)

def base_policy(q,a):
    s=q.score.to_numpy(float); pc=q.precursor_count.to_numpy(int); ag=q.agreement.to_numpy(int)
    red=(s>=float(a['red_threshold']))&(ag>=2)&(pc>=int(a['min_precursor_red']))
    alert=red|(s>=float(a['watch_threshold']))|((pc>=int(a['high_precursor_count']))&(s>=float(a['high_precursor_fallback_threshold'])))
    return red,alert

def build_oof(d,X,pc):
    rows=[]; meta=[]
    for fid,(st,en,tr,va) in enumerate(base.folds(d)):
        ytr=d.loc[tr,'target']; q=d.loc[va,['week_start','city','target','future_ratio']].rename(columns={'target':'y'}).copy(); q['fold_id']=fid; q['precursor_count']=pc.loc[va].to_numpy()
        names=[]
        for name,factory in base.model_factories().items():
            m=base.fit_one(clone(factory),X.loc[tr],ytr); q[name]=m.predict_proba(X.loc[va])[:,1]; names.append(name)
        q['score']=q[names].mean(axis=1); q['agreement']=(q[names]>=.5).sum(axis=1)
        yratio=np.clip(d.loc[tr,'future_ratio'].to_numpy(float),.35,1.75); ylog=np.log(yratio)
        preds=[]
        for name,factory in reg_factories().items():
            m=fit_reg(clone(factory),X.loc[tr,FEATURES],ylog); p=np.exp(m.predict(X.loc[va,FEATURES])); q[name]=p; preds.append(p)
        # Dense downside estimate: equal ensemble of mean and lower-quantile regressors.
        q['pred_ratio']=np.column_stack(preds).mean(axis=1)
        q['trend_evidence']=trend_helpers.evidence_count(X.loc[va])
        rows.append(q); meta.append({'fold_id':fid,'start':str(st.date()),'end':str(en.date()),'rows':int(va.sum()),'positives':int(q.y.sum())})
    return pd.concat(rows,ignore_index=True),meta

def evaluate(q,a,ratio_t,emin):
    y=q.y.to_numpy(int); red,base_alert=base_policy(q,a)
    down=(q.pred_ratio.to_numpy(float)<=ratio_t)&(q.trend_evidence.to_numpy(int)>=emin)
    alert=base_alert|down; m=bm(y,alert); r=bm(y,red); inc=down&~base_alert; base_green_neg=(y==0)&~base_alert
    incneg=float((inc&(y==0)).sum()/max(int(base_green_neg.sum()),1)); rate=float(alert.mean()); cov=1-rate
    per=[]
    for fid,z in q.assign(alert=alert).groupby('fold_id'):
        yy=z.y.to_numpy(int); aa=z.alert.to_numpy(bool); mm=bm(yy,aa); per.append({'fold_id':int(fid),'positives':int(yy.sum()),'recall':mm['recall'],'precision':mm['precision'],'alert_rate':float(aa.mean())})
    stable=[x['recall'] for x in per if x['positives']>=5]; minst=min(stable) if stable else 1.
    ok=(m['recall']>=CONTRACT['alert_recall_min'] and m['NPV']>=CONTRACT['green_npv_min'] and m['precision']>=CONTRACT['alert_precision_min'] and rate<=CONTRACT['alert_rate_max'] and cov>=CONTRACT['green_coverage_min'] and incneg<=CONTRACT['incremental_negative_alert_rate_max'] and minst>=CONTRACT['min_recall_folds_with_5plus_positives'] and r['precision']>=CONTRACT['red_precision_min'] and r['FPR']<=CONTRACT['red_fpr_max'])
    return {'ok':bool(ok),'RED':r,'RED_plus_AMBER':m,'alert_rate':rate,'green_coverage':cov,'incremental_negative_alert_rate':incneg,'ratio_incremental_alerts':int(inc.sum()),'ratio_incremental_tp':int((inc&(y==1)).sum()),'min_recall_folds_5plus':float(minst),'folds':per}

def choose(q,a):
    p=q.pred_ratio.to_numpy(float); cand=np.unique(np.r_[np.quantile(p,np.linspace(.01,.70,220)),np.linspace(.55,1.05,220)])
    valid=[]; allrows=[]
    for emin in range(0,7):
        for t in cand:
            e=evaluate(q,a,float(t),emin); allrows.append((float(t),emin,e))
            if e['ok']:
                obj=(e['RED_plus_AMBER']['recall'],e['RED_plus_AMBER']['NPV'],e['RED_plus_AMBER']['precision'],-e['alert_rate'],-e['incremental_negative_alert_rate'],-float(t),emin)
                valid.append((obj,float(t),emin,e))
    if valid:
        valid.sort(key=lambda x:x[0],reverse=True); return valid[0],len(valid),None
    feasible=[x for x in allrows if x[2]['alert_rate']<=.30 and x[2]['green_coverage']>=.70 and x[2]['incremental_negative_alert_rate']<=.05 and x[2]['RED_plus_AMBER']['precision']>=.18]
    best=None
    if feasible:
        best=max(feasible,key=lambda x:(x[2]['RED_plus_AMBER']['recall'],x[2]['RED_plus_AMBER']['NPV'],-x[2]['alert_rate']))
    return None,0,best

def main():
    a=joblib.load(BASE_MODEL)
    if a.get('version')!='SAMA-CITY-RISK-3.3-DUAL-CHANNEL': raise RuntimeError(f'Unexpected base {a.get("version")}')
    panel=base.source.reconciled_load_panel(base.HISTORY); d,X,P,pc=base.featureize(panel,require_target=True); keep=d.week_start<=base.DEV_END
    d=d.loc[keep].reset_index(drop=True); X=X.loc[keep].reset_index(drop=True); pc=pc.loc[keep].reset_index(drop=True)
    forbidden=[c for c in FEATURES if c.startswith('city_') or 'target' in c or 'future' in c or 'decline_rate' in c]
    if forbidden: raise RuntimeError(f'Forbidden {forbidden}')
    q,folds=build_oof(d,X,pc); best,nvalid,closest=choose(q,a)
    if best is None:
        diag={'version':VERSION,'status':'NO_POLICY','closest_feasible':None if closest is None else {'ratio_threshold':closest[0],'evidence_min':closest[1],'metrics':closest[2]},'scientific_boundary':'Historical OOF only; no recent/counterfactual labels used.'}
        REPORT.write_text(json.dumps(diag,indent=2),encoding='utf-8'); print(json.dumps(diag,indent=2)); raise SystemExit(2)
    _,rt,emin,metrics=best
    yratio=np.clip(d.future_ratio.to_numpy(float),.35,1.75); ylog=np.log(yratio); regs={name:fit_reg(clone(factory),X[FEATURES],ylog) for name,factory in reg_factories().items()}
    out=dict(a); out.update({'version':VERSION,'base_version':a['version'],'ratio_models':regs,'ratio_features':FEATURES,'ratio_threshold':rt,'ratio_evidence_min':emin,'ratio_contract':CONTRACT,'development_end':str(base.DEV_END.date()),'scope':'frozen v3.3 RED/base watch plus dense downside next-week-ratio forecast AMBER channel'})
    joblib.dump(out,MODEL)
    rep={'version':VERSION,'base_version':a['version'],'rows':len(d),'positives':int(d.target.sum()),'positive_rate':float(d.target.mean()),'ratio_feature_count':len(FEATURES),'ratio_threshold':rt,'ratio_evidence_min':emin,'valid_policy_count':nvalid,'metrics':metrics,'contract':CONTRACT,'all_gates_passed':bool(metrics['ok']),'folds':folds,'controls':{'base_red_policy_unchanged':True,'ratio_channel_only_adds_amber':True,'dense_continuous_target_used_for_ratio_channel':True,'no_city_identity':True,'no_target_history':True,'no_future_features':True,'selection_historical_oof_only':True,'no_recent_sama_or_counterfactual_labels_used':True},'scientific_boundary':'No outcome after 2025-06-29 and no prior semi-synthetic/counterfactual test result is read by this trainer.'}
    REPORT.write_text(json.dumps(rep,indent=2),encoding='utf-8'); (OUT/'development_summary.md').write_text('# Sales Sentinel v3.4.2 — Downside Ratio Forecast\n\n'+f"- Alert recall **{metrics['RED_plus_AMBER']['recall']:.2%}**\n- Alert precision **{metrics['RED_plus_AMBER']['precision']:.2%}**\n- GREEN NPV **{metrics['RED_plus_AMBER']['NPV']:.2%}**\n- Alert rate **{metrics['alert_rate']:.2%}**\n- GREEN coverage **{metrics['green_coverage']:.2%}**\n- Incremental negative alert rate **{metrics['incremental_negative_alert_rate']:.2%}**\n- Predicted-ratio threshold **{rt:.4f}**, evidence >= **{emin}**\n- All gates **{metrics['ok']}**\n",encoding='utf-8'); print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
