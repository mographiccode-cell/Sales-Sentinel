from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

import run_sama_city_risk_v2_1 as source
import train_sama_city_risk_v2_2 as v22

ROOT=Path(__file__).resolve().parents[1]
HISTORY=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2025.csv'
BASE=ROOT/'models'/'sama_city_v2_2'/'city_market_risk_v2_2.joblib'
OUT=ROOT/'reports'/'sama_city_v2_4'; MOD=ROOT/'models'/'sama_city_v2_4'
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'policy_development_report.json'; SUMMARY=OUT/'policy_development_summary.md'; POLICY=MOD/'prior_shift_policy_v2_4.joblib'
VERSION='SAMA-CITY-RISK-2.4-PRIOR-SHIFT'
SELECT_END=pd.Timestamp('2024-12-31'); POLICY_START=pd.Timestamp('2025-01-01'); POLICY_END=pd.Timestamp('2025-06-29')
GLOBAL_LOOKBACK_WEEKS=52
CITY_LOOKBACK_WEEKS=104
CITY_PRIOR_STRENGTH=26.0   # half-year pseudo-observations toward current global prevalence
PRIOR_FLOOR=.003
PRIOR_CEIL=.20


def oof_selected(d,X,selected):
    frames=[]; meta=[]
    for st,en,tr,va in v22.folds(d):
        y=d.loc[tr,'target']; pos=int(y.sum()); neg=len(y)-pos
        model=clone(v22.models(neg/max(pos,1))[selected]).fit(X.loc[tr],y)
        raw=model.predict_proba(X.loc[va])[:,1]
        q=d.loc[va,['week_start','city','target']].copy().rename(columns={'target':'y'}); q['raw_score']=raw; frames.append(q)
        meta.append({'start':str(st.date()),'end':str(en.date()),'train_rows':int(tr.sum()),'validation_rows':int(va.sum())})
    if not frames: raise RuntimeError('No OOF folds')
    return pd.concat(frames,ignore_index=True).sort_values(['week_start','city']).reset_index(drop=True),meta


def odds_shift(p,from_prior,to_prior):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    fp=float(np.clip(from_prior,1e-6,1-1e-6)); tp=np.clip(np.asarray(to_prior,float),1e-6,1-1e-6)
    odds=p/(1-p); ref=fp/(1-fp); new=tp/(1-tp)
    o=odds*(new/ref)
    return o/(1+o)


def estimate_prior(history,week,city,reference_prior):
    h=history[history.week_start<week].copy()
    if h.empty: return reference_prior,reference_prior,reference_prior
    gcut=week-pd.Timedelta(weeks=GLOBAL_LOOKBACK_WEEKS)
    gh=h[h.week_start>=gcut]
    if len(gh)<100: gh=h
    gp=float(np.clip(gh.y.mean(),PRIOR_FLOOR,PRIOR_CEIL))
    ccut=week-pd.Timedelta(weeks=CITY_LOOKBACK_WEEKS)
    ch=h[(h.city==city)&(h.week_start>=ccut)]
    # Empirical-Bayes city prior shrinks quiet/noisy city rates toward the CURRENT global regime.
    cp=float((ch.y.sum()+CITY_PRIOR_STRENGTH*gp)/(len(ch)+CITY_PRIOR_STRENGTH)) if len(ch) else gp
    cp=float(np.clip(cp,PRIOR_FLOOR,PRIOR_CEIL))
    # Conservative geometric blend prevents a single quiet city from driving risk to zero.
    effective=float(np.sqrt(gp*cp))
    return gp,cp,effective


def prequential_adjust(rows,history,reference_prior):
    history=history.copy().sort_values(['week_start','city']); out=[]
    for week in sorted(rows.week_start.unique()):
        current=rows[rows.week_start==week].copy(); adjusted=[]; gps=[]; cps=[]; eps=[]
        for _,r in current.iterrows():
            gp,cp,ep=estimate_prior(history,pd.Timestamp(week),r.city,reference_prior)
            adjusted.append(float(odds_shift([r.base_probability],reference_prior,[ep])[0])); gps.append(gp); cps.append(cp); eps.append(ep)
        current['global_prior']=gps; current['city_prior']=cps; current['effective_prior']=eps; current['adjusted_probability']=adjusted; out.append(current)
        # Only after predictions for this origin are fixed does its realized target enter later priors.
        history=pd.concat([history,current[['week_start','city','y']]],ignore_index=True).sort_values(['week_start','city'])
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()


def confusion(y,p,t):
    y=np.asarray(y,int); pred=np.asarray(p)>=t; tp=int(((y==1)&pred).sum()); fp=int(((y==0)&pred).sum()); fn=int(((y==1)&~pred).sum()); tn=int(((y==0)&~pred).sum())
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':float(tp/max(tp+fp,1)),'recall':float(tp/max(tp+fn,1)),'FPR':float(fp/max(fp+tn,1)),'NPV':float(tn/max(tn+fn,1))}


def choose_thresholds(y,p,contract):
    cand=np.unique(np.r_[np.linspace(.001,.999,999),np.quantile(p,np.linspace(0,1,401))])
    watch=[]
    for t in cand:
        m=confusion(y,p,t)
        if m['recall']>=contract['alert_recall_min'] and m['NPV']>=contract['green_npv_min']:
            watch.append((m['precision'],m['NPV'],-t,float(t),m))
    if not watch: raise RuntimeError('No WATCH threshold satisfies historical prior-shift contract')
    wm=max(watch,key=lambda x:x[:3]); wt=wm[3]
    red=[]
    for t in cand[cand>=wt]:
        m=confusion(y,p,t); alerts=m['TP']+m['FP']
        if alerts>=5 and m['precision']>=contract['red_precision_min'] and m['FPR']<=contract['red_fpr_max']:
            red.append((m['recall'],m['precision'],-m['FPR'],float(t),m))
    if not red: raise RuntimeError('No RED threshold satisfies historical prior-shift contract')
    rm=max(red,key=lambda x:x[:3]); return wt,rm[3],wm[4],rm[4]


def main():
    art=joblib.load(BASE)
    d,X=v22.featureize(source.reconciled_load_panel(HISTORY)); keep=d.week_start<=pd.Timestamp(art['development_end']); d=d.loc[keep].reset_index(drop=True); X=X.loc[keep].reset_index(drop=True)
    oo,folds=oof_selected(d,X,art['selected']); oo['base_probability']=art['calibrator'].predict_proba(pd.DataFrame({art['selected']:oo.raw_score}))[:,1]
    selection=oo[oo.week_start<=SELECT_END].copy(); reference_prior=float(selection.y.mean())
    history=oo[oo.week_start<POLICY_START][['week_start','city','y']].copy()
    policy_rows=oo[oo.week_start.between(POLICY_START,POLICY_END)].copy()
    adj=prequential_adjust(policy_rows,history,reference_prior)
    contract=art['contract']; wt,rt,wm,rm=choose_thresholds(adj.y,adj.adjusted_probability.to_numpy(),contract)
    gates={'red_precision':rm['precision']>=contract['red_precision_min'],'red_fpr':rm['FPR']<=contract['red_fpr_max'],'alert_recall':wm['recall']>=contract['alert_recall_min'],'green_npv':wm['NPV']>=contract['green_npv_min']}
    policy={'version':VERSION,'base_model_version':art['version'],'reference_prior':reference_prior,'global_lookback_weeks':GLOBAL_LOOKBACK_WEEKS,'city_lookback_weeks':CITY_LOOKBACK_WEEKS,'city_prior_strength':CITY_PRIOR_STRENGTH,'prior_floor':PRIOR_FLOOR,'prior_ceil':PRIOR_CEIL,'watch_threshold':wt,'red_threshold':rt,'contract':contract,'initial_label_history':oo[oo.week_start<=POLICY_END][['week_start','city','y']].copy(),'production_update_rule':'Estimate current prior from previously realized labels only; classify current origin; append its outcome only after the next week closes.'}
    joblib.dump(policy,POLICY)
    report={'version':VERSION,'base_model':art['version'],'scientific_boundary':'Prior-shift method and thresholds are evaluated on historical OOF only through 2025-06-29. Consumed 2025-2026 stress outcomes are not used to estimate parameters or thresholds.','reference_prior':reference_prior,'parameters':{'global_lookback_weeks':GLOBAL_LOOKBACK_WEEKS,'city_lookback_weeks':CITY_LOOKBACK_WEEKS,'city_prior_strength':CITY_PRIOR_STRENGTH,'prior_floor':PRIOR_FLOOR,'prior_ceil':PRIOR_CEIL},'policy_rows':len(adj),'policy_decline_rate':float(adj.y.mean()),'observed_prior_range':{'global_min':float(adj.global_prior.min()),'global_max':float(adj.global_prior.max()),'city_min':float(adj.city_prior.min()),'city_max':float(adj.city_prior.max()),'effective_min':float(adj.effective_prior.min()),'effective_max':float(adj.effective_prior.max())},'thresholds':{'watch':wt,'red':rt},'historical_RED':rm,'historical_RED_plus_AMBER':wm,'contract':contract,'gates':gates,'all_gates_passed':bool(all(gates.values())),'folds':folds,'controls':{'prior_estimate_uses_only_past_realized_labels':True,'city_prior_shrunk_to_current_global_prior':True,'no_fresh_2025_2026_labels_used':True,'no_base_model_retraining':True}}
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8'); SUMMARY.write_text(f'''# SAMA City Risk v2.4 — Prior-Shift Policy\n\n- Base model: **{art['version']}**\n- Reference decline prior: **{reference_prior:.2%}**\n- Global prior lookback: **{GLOBAL_LOOKBACK_WEEKS} weeks**\n- City prior lookback: **{CITY_LOOKBACK_WEEKS} weeks**\n- Historical policy rows: **{len(adj):,}**\n- RED precision: **{rm['precision']:.2%}** ({rm['TP']} TP / {rm['FP']} FP)\n- RED FPR: **{rm['FPR']:.2%}**\n- RED+AMBER recall: **{wm['recall']:.2%}**\n- GREEN NPV: **{wm['NPV']:.2%}**\n- All historical gates passed: **{report['all_gates_passed']}**\n- Fresh 2025-2026 labels used to select policy: **No**\n''',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
