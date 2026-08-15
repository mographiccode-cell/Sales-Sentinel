from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_sama_city_risk_v3 as base

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = ROOT / 'models' / 'sama_city_v3_3' / 'city_risk_v3_3.joblib'
OUT = ROOT / 'reports' / 'sama_city_v3_4_1'
MOD = ROOT / 'models' / 'sama_city_v3_4_1'
OUT.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)
MODEL = MOD / 'city_risk_v3_4_1.joblib'
REPORT = OUT / 'development_report.json'
VERSION = 'SAMA-CITY-RISK-3.4.1-TREND-CLASSIFIER'

TREND_FEATURES = [
    'value_ratio_mean_4','count_ratio_mean_4',
    'value_change_1','value_change_2','value_change_4',
    'count_change_1','count_change_2','count_change_4',
    'value_slope_4','count_slope_4',
    'value_drawdown_13','count_drawdown_13',
    'value_share_ratio_13','count_share_ratio_13',
    'value_share_change_4','count_share_change_4',
    'ticket_change_4',
]

CONTRACT = {
    'alert_recall_min': 0.94,
    'green_npv_min': 0.992,
    'alert_precision_min': 0.18,
    'alert_rate_max': 0.30,
    'green_coverage_min': 0.70,
    'incremental_negative_alert_rate_max': 0.05,
    'min_recall_folds_with_5plus_positives': 0.70,
    'red_precision_min': 0.70,
    'red_fpr_max': 0.015,
}


def bm(y, pred):
    y=np.asarray(y,int); pred=np.asarray(pred,bool)
    tp=int(((y==1)&pred).sum()); fp=int(((y==0)&pred).sum()); fn=int(((y==1)&~pred).sum()); tn=int(((y==0)&~pred).sum())
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'FPR':fp/max(fp+tn,1),'NPV':tn/max(tn+fn,1)}


def trend_factory():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=.14, penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=5000, random_state=137),
    )


def evidence_count(X):
    checks=np.column_stack([
        X['value_ratio_mean_4'].to_numpy(float)<.99,
        X['count_ratio_mean_4'].to_numpy(float)<.99,
        X['value_change_1'].to_numpy(float)<-.015,
        X['count_change_1'].to_numpy(float)<-.015,
        X['value_change_2'].to_numpy(float)<-.03,
        X['count_change_2'].to_numpy(float)<-.03,
        X['value_slope_4'].to_numpy(float)<-.006,
        X['count_slope_4'].to_numpy(float)<-.006,
        X['value_share_change_4'].to_numpy(float)<-.012,
        X['count_share_change_4'].to_numpy(float)<-.012,
    ])
    return checks.sum(axis=1).astype(int)


def base_policy(q, art):
    score=q.score.to_numpy(float); pc=q.precursor_count.to_numpy(int); agree=q.agreement.to_numpy(int)
    red=(score>=float(art['red_threshold']))&(agree>=2)&(pc>=int(art['min_precursor_red']))
    alert=red|(score>=float(art['watch_threshold']))|((pc>=int(art['high_precursor_count']))&(score>=float(art['high_precursor_fallback_threshold'])))
    return red,alert


def build_oof(d,X,pc):
    rows=[]; meta=[]
    for fid,(st,en,tr,va) in enumerate(base.folds(d)):
        ytr=d.loc[tr,'target']; q=d.loc[va,['week_start','city','target']].rename(columns={'target':'y'}).copy(); q['fold_id']=fid; q['precursor_count']=pc.loc[va].to_numpy()
        names=[]
        for name,factory in base.model_factories().items():
            m=base.fit_one(clone(factory),X.loc[tr],ytr); q[name]=m.predict_proba(X.loc[va])[:,1]; names.append(name)
        q['score']=q[names].mean(axis=1); q['agreement']=(q[names]>=.5).sum(axis=1)
        tm=trend_factory().fit(X.loc[tr,TREND_FEATURES],ytr); q['trend_score']=tm.predict_proba(X.loc[va,TREND_FEATURES])[:,1]
        q['trend_evidence']=evidence_count(X.loc[va])
        rows.append(q); meta.append({'fold_id':fid,'start':str(st.date()),'end':str(en.date()),'rows':int(va.sum()),'positives':int(q.y.sum())})
    return pd.concat(rows,ignore_index=True),meta


def evaluate(q,art,t,evidence_min):
    y=q.y.to_numpy(int); red,base_alert=base_policy(q,art)
    trend=(q.trend_score.to_numpy(float)>=t)&(q.trend_evidence.to_numpy(int)>=evidence_min)
    alert=base_alert|trend; m=bm(y,alert); r=bm(y,red)
    inc=trend&~base_alert; base_green_neg=(y==0)&~base_alert
    inc_neg=float((inc&(y==0)).sum()/max(int(base_green_neg.sum()),1)); rate=float(alert.mean()); cov=1-rate
    per=[]
    for fid,z in q.assign(alert=alert).groupby('fold_id'):
        yy=z.y.to_numpy(int); aa=z.alert.to_numpy(bool); mm=bm(yy,aa); per.append({'fold_id':int(fid),'positives':int(yy.sum()),'recall':mm['recall'],'precision':mm['precision'],'alert_rate':float(aa.mean())})
    stable=[x['recall'] for x in per if x['positives']>=5]; minstable=min(stable) if stable else 1.
    ok=(m['recall']>=CONTRACT['alert_recall_min'] and m['NPV']>=CONTRACT['green_npv_min'] and m['precision']>=CONTRACT['alert_precision_min'] and rate<=CONTRACT['alert_rate_max'] and cov>=CONTRACT['green_coverage_min'] and inc_neg<=CONTRACT['incremental_negative_alert_rate_max'] and minstable>=CONTRACT['min_recall_folds_with_5plus_positives'] and r['precision']>=CONTRACT['red_precision_min'] and r['FPR']<=CONTRACT['red_fpr_max'])
    return {'ok':bool(ok),'RED':r,'RED_plus_AMBER':m,'alert_rate':rate,'green_coverage':cov,'incremental_negative_alert_rate':inc_neg,'trend_incremental_alerts':int(inc.sum()),'trend_incremental_tp':int((inc&(y==1)).sum()),'min_recall_folds_5plus':float(minstable),'folds':per}


def choose(q,art):
    s=q.trend_score.to_numpy(float); cand=np.unique(np.r_[np.quantile(s,np.linspace(.35,.995,180)),np.linspace(.10,.90,160)])
    valid=[]
    for emin in (1,2,3,4,5,6):
        for t in cand:
            e=evaluate(q,art,float(t),emin)
            if e['ok']:
                obj=(e['RED_plus_AMBER']['recall'],e['RED_plus_AMBER']['NPV'],e['RED_plus_AMBER']['precision'],-e['alert_rate'],-e['incremental_negative_alert_rate'],float(t),emin)
                valid.append((obj,float(t),emin,e))
    if not valid: raise RuntimeError('No v3.4.1 trend-classifier policy meets historical OOF contract')
    valid.sort(key=lambda x:x[0],reverse=True)
    return valid[0],len(valid)


def main():
    a=joblib.load(BASE_MODEL)
    if a.get('version')!='SAMA-CITY-RISK-3.3-DUAL-CHANNEL': raise RuntimeError(f'Unexpected base {a.get("version")}')
    panel=base.source.reconciled_load_panel(base.HISTORY); d,X,P,pc=base.featureize(panel,require_target=True); keep=d.week_start<=base.DEV_END
    d=d.loc[keep].reset_index(drop=True); X=X.loc[keep].reset_index(drop=True); pc=pc.loc[keep].reset_index(drop=True)
    missing=[c for c in TREND_FEATURES if c not in X.columns]
    if missing: raise RuntimeError(f'Missing trend features {missing}')
    forbidden=[c for c in TREND_FEATURES if c.startswith('city_') or 'target' in c or 'future' in c or 'decline_rate' in c]
    if forbidden: raise RuntimeError(f'Forbidden {forbidden}')
    q,folds=build_oof(d,X,pc); best,nvalid=choose(q,a); _,t,emin,metrics=best
    tm=trend_factory().fit(X[TREND_FEATURES],d.target)
    out=dict(a); out.update({'version':VERSION,'base_version':a['version'],'trend_model':tm,'trend_features':TREND_FEATURES,'trend_threshold':t,'trend_evidence_min':emin,'trend_contract':CONTRACT,'development_end':str(base.DEV_END.date()),'scope':'frozen v3.3 RED/base watch plus independent supervised trend AMBER channel'})
    joblib.dump(out,MODEL)
    rep={'version':VERSION,'base_version':a['version'],'rows':len(d),'positives':int(d.target.sum()),'positive_rate':float(d.target.mean()),'trend_feature_count':len(TREND_FEATURES),'trend_features':TREND_FEATURES,'trend_threshold':t,'trend_evidence_min':emin,'valid_policy_count':nvalid,'metrics':metrics,'contract':CONTRACT,'all_gates_passed':bool(metrics['ok']),'folds':folds,'controls':{'base_red_policy_unchanged':True,'trend_channel_only_adds_amber':True,'no_city_identity':True,'no_target_history':True,'no_future_features':True,'threshold_selected_historical_oof_only':True,'no_recent_sama_or_counterfactual_labels_used':True},'scientific_boundary':'No outcome after 2025-06-29 and no prior semi-synthetic/counterfactual test result is read by this trainer.'}
    REPORT.write_text(json.dumps(rep,indent=2),encoding='utf-8')
    (OUT/'development_summary.md').write_text('# Sales Sentinel v3.4.1 — Independent Trend Classifier\n\n'+f"- Alert recall **{metrics['RED_plus_AMBER']['recall']:.2%}**\n- Alert precision **{metrics['RED_plus_AMBER']['precision']:.2%}**\n- GREEN NPV **{metrics['RED_plus_AMBER']['NPV']:.2%}**\n- Alert rate **{metrics['alert_rate']:.2%}**\n- GREEN coverage **{metrics['green_coverage']:.2%}**\n- Incremental negative alert rate **{metrics['incremental_negative_alert_rate']:.2%}**\n- Trend threshold **{t:.4f}**, evidence >= **{emin}**\n- All gates **{metrics['ok']}**\n",encoding='utf-8')
    print(json.dumps(rep,indent=2))
    if not metrics['ok']: raise SystemExit(2)

if __name__=='__main__': main()
