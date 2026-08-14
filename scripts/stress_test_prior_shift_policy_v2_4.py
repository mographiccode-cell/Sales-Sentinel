from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import run_sama_city_risk_v2_1 as source
import train_sama_city_risk_v2_2 as v22
import build_prior_shift_policy_v2_4 as p24
import stress_test_conformal_policy_v2_3 as schema

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'
FRESH=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2025_2026_holdout.csv'
BASE=ROOT/'models'/'sama_city_v2_2'/'city_market_risk_v2_2.joblib'
POLICY=ROOT/'models'/'sama_city_v2_4'/'prior_shift_policy_v2_4.joblib'
OUT=ROOT/'reports'/'sama_city_v2_4'/'post_diagnosis_prequential_stress.json'
SUMMARY=ROOT/'reports'/'sama_city_v2_4'/'post_diagnosis_prequential_stress.md'


def main():
    base=joblib.load(BASE); policy=joblib.load(POLICY)
    if base['version']!=v22.VERSION: raise RuntimeError('Unexpected frozen v2.2 base')
    if policy['version']!=p24.VERSION: raise RuntimeError('Unexpected v2.4 policy')

    d,X=v22.featureize(source.reconciled_load_panel(EXT))
    Xf=schema.enforce_feature_contract(d,X,base['features'])
    fresh=pd.read_csv(FRESH,parse_dates=['week_start']); fsets={c:set(q.week_start.dt.normalize()) for c,q in fresh.groupby('city')}
    mask=[]
    for w,c in zip(d.week_start,d.city):
        ww=pd.Timestamp(w).normalize(); weeks=sorted(fsets.get(c,set()))
        mask.append(bool(weeks) and ww in fsets[c] and ww!=weeks[-1])
    d=d.loc[mask].reset_index(drop=True); Xf=Xf.loc[mask].reset_index(drop=True)
    if len(d)<500: raise RuntimeError(f'Too few stress rows: {len(d)}')

    raw=base['model'].predict_proba(Xf)[:,1]
    base_p=base['calibrator'].predict_proba(pd.DataFrame({base['selected']:raw}))[:,1]
    rows=d[['week_start','city','target']].rename(columns={'target':'y'}).copy(); rows['base_probability']=base_p

    history=policy['initial_label_history'].copy(); history.week_start=pd.to_datetime(history.week_start); history=history.sort_values(['week_start','city'])
    results=[]; weekly=[]
    for week in sorted(rows.week_start.unique()):
        current=rows[rows.week_start==week].copy(); adj=[]; gps=[]; cps=[]; eps=[]
        for _,r in current.iterrows():
            gp,cp,ep=p24.estimate_prior(history,pd.Timestamp(week),r.city,float(policy['reference_prior']))
            ap=float(p24.odds_shift([r.base_probability],float(policy['reference_prior']),[ep])[0])
            gps.append(gp); cps.append(cp); eps.append(ep); adj.append(ap)
        current['global_prior']=gps; current['city_prior']=cps; current['effective_prior']=eps; current['adjusted_probability']=adj
        current['state']=np.where(current.adjusted_probability>=float(policy['red_threshold']),'RED',np.where(current.adjusted_probability>=float(policy['watch_threshold']),'AMBER','GREEN'))
        results.append(current)
        weekly.append({'week_start':str(pd.Timestamp(week).date()),'global_prior_mean':float(np.mean(gps)),'effective_prior_mean':float(np.mean(eps)),'red':int(current.state.eq('RED').sum()),'amber':int(current.state.eq('AMBER').sum()),'green':int(current.state.eq('GREEN').sum()),'declines_realized_after_prediction':int(current.y.sum())})
        # Current labels become available only after prediction and are appended for later weeks.
        history=pd.concat([history,current[['week_start','city','y']]],ignore_index=True).sort_values(['week_start','city'])

    ev=pd.concat(results,ignore_index=True); red=ev.state.eq('RED').to_numpy(); green=ev.state.eq('GREEN').to_numpy(); alert=~green; y=ev.y.astype(int).to_numpy(); pos=y==1; neg=y==0
    redtp=int((red&pos).sum()); redfp=int((red&neg).sum()); atp=int((alert&pos).sum()); afp=int((alert&neg).sum()); gtn=int((green&neg).sum()); gfn=int((green&pos).sum())
    metrics={'rows':len(ev),'declines':int(pos.sum()),'decline_rate':float(pos.mean()),
             'RED':{'rows':int(red.sum()),'TP':redtp,'FP':redfp,'precision':float(redtp/max(redtp+redfp,1)),'FPR':float(redfp/max(neg.sum(),1)),'recall_contribution':float(redtp/max(pos.sum(),1))},
             'AMBER':{'rows':int(ev.state.eq('AMBER').sum()),'declines':int((ev.state.eq('AMBER').to_numpy()&pos).sum())},
             'GREEN':{'rows':int(green.sum()),'TN':gtn,'FN':gfn,'NPV':float(gtn/max(gtn+gfn,1)),'miss_rate':float(gfn/max(pos.sum(),1))},
             'RED_plus_AMBER':{'rows':int(alert.sum()),'TP':atp,'FP':afp,'precision':float(atp/max(atp+afp,1)),'recall':float(atp/max(pos.sum(),1))}}
    rank={'ROC_AUC':float(roc_auc_score(y,ev.adjusted_probability)),'PR_AUC':float(average_precision_score(y,ev.adjusted_probability))}
    brier=float(brier_score_loss(y,np.clip(ev.adjusted_probability,1e-6,1-1e-6)))
    contract=policy['contract']; gates={'red_precision':metrics['RED']['precision']>=contract['red_precision_min'],'red_fpr':metrics['RED']['FPR']<=contract['red_fpr_max'],'alert_recall':metrics['RED_plus_AMBER']['recall']>=contract['alert_recall_min'],'green_npv':metrics['GREEN']['NPV']>=contract['green_npv_min'],'roc_auc':rank['ROC_AUC']>=contract['roc_auc_min'],'pr_auc':rank['PR_AUC']>=contract['pr_auc_min']}

    by_city={}
    for c,q in ev.groupby('city'):
        yy=q.y.astype(int).to_numpy(); rr=q.state.eq('RED').to_numpy(); gg=q.state.eq('GREEN').to_numpy(); aa=~gg; pp=yy==1; nn=yy==0
        by_city[c]={'rows':len(q),'declines':int(pp.sum()),'red_tp':int((rr&pp).sum()),'red_fp':int((rr&nn).sum()),'green_fn':int((gg&pp).sum()),'alert_recall':float((aa&pp).sum()/max(pp.sum(),1)) if pp.sum() else None,'prior_mean':float(q.effective_prior.mean())}

    report={'version':'SAMA-CITY-RISK-2.4-PREQUENTIAL-STRESS','independence_status':'NOT AN INDEPENDENT HOLDOUT: architecture was motivated by diagnosis of this period, but v2.4 parameters/thresholds were selected on historical OOF only.','base_model':base['version'],'policy':policy['version'],'rows':len(ev),'weeks':int(ev.week_start.nunique()),'period':{'start':str(ev.week_start.min().date()),'end':str(ev.week_start.max().date())},'fixed_thresholds':{'watch':float(policy['watch_threshold']),'red':float(policy['red_threshold'])},'reference_prior':float(policy['reference_prior']),'stress_prior_range':{'global_min':float(ev.global_prior.min()),'global_max':float(ev.global_prior.max()),'effective_min':float(ev.effective_prior.min()),'effective_max':float(ev.effective_prior.max())},'metrics':metrics,'ranking':rank,'Brier':brier,'contract':contract,'gates':gates,'all_gates_passed':bool(all(gates.values())),'by_city':by_city,'weekly':weekly,'anti_leakage':{'current_label_used_before_current_prediction':False,'fresh_labels_enter_prior_only_after_prediction':True,'base_model_retrained_on_fresh':False,'thresholds_selected_on_fresh':False,'prior_parameters_selected_on_fresh':False,'feature_contract_frozen':True}}
    OUT.write_text(json.dumps(report,indent=2),encoding='utf-8'); SUMMARY.write_text(f'''# SAMA City Risk v2.4 — Prequential Stress Test\n\n**Stress test, not an independent fresh holdout.**\n\n- Rows: **{len(ev):,}** across **{ev.week_start.nunique()} weeks**\n- Declines: **{pos.sum()} ({pos.mean():.2%})**\n- RED precision: **{metrics['RED']['precision']:.2%}** ({redtp} TP / {redfp} FP)\n- RED FPR: **{metrics['RED']['FPR']:.2%}**\n- RED+AMBER recall: **{metrics['RED_plus_AMBER']['recall']:.2%}**\n- GREEN NPV: **{metrics['GREEN']['NPV']:.2%}**\n- Missed declines: **{gfn} / {pos.sum()}**\n- PR-AUC: **{rank['PR_AUC']:.2%}**\n- ROC-AUC: **{rank['ROC_AUC']:.2%}**\n- Stress gates passed: **{report['all_gates_passed']}**\n- Current/future label used before prediction: **No**\n''',encoding='utf-8')
    print(json.dumps({'metrics':metrics,'ranking':rank,'gates':gates,'all':report['all_gates_passed']},indent=2))

if __name__=='__main__':main()
