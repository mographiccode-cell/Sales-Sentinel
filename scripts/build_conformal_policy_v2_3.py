from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import run_sama_city_risk_v2_1 as source
import train_sama_city_risk_v2_2 as v22

ROOT=Path(__file__).resolve().parents[1]
HISTORY=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2025.csv'
V22_MODEL=ROOT/'models'/'sama_city_v2_2'/'city_market_risk_v2_2.joblib'
OUT=ROOT/'reports'/'sama_city_v2_3'; MOD=ROOT/'models'/'sama_city_v2_3'
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'policy_development_report.json'; SUMMARY=OUT/'policy_development_summary.md'; POLICY=MOD/'conformal_policy_v2_3.joblib'
VERSION='SAMA-CITY-RISK-2.3-ROLLING-CONFORMAL'
# Class-conditional significance budgets are requirements, not fitted probability thresholds.
ALPHA_RED_GLOBAL=.0075
ALPHA_RED_CITY=.05
ALPHA_GREEN_DECLINE=.08   # target <=8% class-conditional misses -> >=92% alert recall
MIN_GLOBAL_NEG=300
MIN_GLOBAL_POS=20
MIN_CITY_NEG=30
CAL_LOOKBACK_WEEKS=104
POLICY_EVAL_START=pd.Timestamp('2025-01-01')
POLICY_EVAL_END=pd.Timestamp('2025-06-29')


def conformal_p_ge(values,x):
    a=np.asarray(values,float)
    return float((1+np.sum(a>=x))/(len(a)+1))

def conformal_decline_p(pos_scores,x):
    # Nonconformity for class=decline is 1-score. p is fraction with calibration score <= current score.
    a=np.asarray(pos_scores,float)
    return float((1+np.sum(a<=x))/(len(a)+1))

def classify_week(current,history):
    cutoff=current.week_start.iloc[0]-pd.Timedelta(weeks=CAL_LOOKBACK_WEEKS)
    h=history[history.week_start>=cutoff].copy()
    # Expand farther back only when rare positive samples are insufficient.
    if (h.y==1).sum()<MIN_GLOBAL_POS or (h.y==0).sum()<MIN_GLOBAL_NEG:
        h=history.copy()
    neg=h[h.y==0]; pos=h[h.y==1]
    if len(neg)<MIN_GLOBAL_NEG or len(pos)<MIN_GLOBAL_POS:
        raise RuntimeError(f'Conformal calibration insufficient: neg={len(neg)}, pos={len(pos)}')
    rows=[]
    for _,r in current.iterrows():
        city_neg=neg[neg.city==r.city]
        if len(city_neg)<MIN_CITY_NEG: city_neg=neg
        p0_global=conformal_p_ge(neg.score,r.score)
        p0_city=conformal_p_ge(city_neg.score,r.score)
        p1=conformal_decline_p(pos.score,r.score)
        red=(p0_global<=ALPHA_RED_GLOBAL and p0_city<=ALPHA_RED_CITY)
        green=(p1<=ALPHA_GREEN_DECLINE and not red)
        state='RED' if red else ('GREEN' if green else 'AMBER')
        rows.append({**r.to_dict(),'p_non_decline_global':p0_global,'p_non_decline_city':p0_city,'p_decline':p1,'state':state,'cal_neg':len(neg),'cal_pos':len(pos),'cal_city_neg':len(city_neg)})
    return pd.DataFrame(rows)

def metrics(q):
    y=q.y.astype(int).to_numpy(); red=q.state.eq('RED').to_numpy(); green=q.state.eq('GREEN').to_numpy(); alert=~green; pos=y==1; neg=y==0
    redtp=int((red&pos).sum()); redfp=int((red&neg).sum()); gtn=int((green&neg).sum()); gfn=int((green&pos).sum()); atp=int((alert&pos).sum()); afp=int((alert&neg).sum())
    return {
        'rows':len(q),'declines':int(pos.sum()),'decline_rate':float(pos.mean()),
        'RED':{'rows':int(red.sum()),'TP':redtp,'FP':redfp,'precision':float(redtp/max(redtp+redfp,1)),'FPR':float(redfp/max(neg.sum(),1)),'recall_contribution':float(redtp/max(pos.sum(),1))},
        'AMBER':{'rows':int(q.state.eq('AMBER').sum()),'declines':int((q.state.eq('AMBER').to_numpy()&pos).sum())},
        'GREEN':{'rows':int(green.sum()),'TN':gtn,'FN':gfn,'NPV':float(gtn/max(gtn+gfn,1)),'miss_rate':float(gfn/max(pos.sum(),1))},
        'RED_plus_AMBER':{'rows':int(alert.sum()),'TP':atp,'FP':afp,'precision':float(atp/max(atp+afp,1)),'recall':float(atp/max(pos.sum(),1))},
    }

def main():
    art=joblib.load(V22_MODEL)
    if art['version']!=v22.VERSION: raise RuntimeError('v2.2 frozen model artifact missing/unexpected')
    d,X=v22.featureize(source.reconciled_load_panel(HISTORY)); keep=d.week_start<=pd.Timestamp(art['development_end']); d=d.loc[keep].reset_index(drop=True); X=X.loc[keep].reset_index(drop=True)
    # Recreate leakage-safe OOF raw scores. Model family/hyperparameters already frozen by v2.2.
    oo,_=v22.oof(d,X)
    if art['selected'] not in oo.columns: raise RuntimeError('Selected v2.2 score absent from OOF')
    oo=oo.rename(columns={'y':'y'}).copy()
    # Apply the frozen v2.2 monotonic calibrator; no fresh outcomes involved.
    oo['score']=art['calibrator'].predict_proba(oo[[art['selected']]])[:,1]
    # Attach city safely by unique week+target ordering. OOF rows are all 11 cities per validation week.
    keys=d[['week_start','city','target']].rename(columns={'target':'y'})
    # Use within-week occurrence order to avoid ambiguity from equal labels.
    oo['_n']=oo.groupby('week_start').cumcount(); keys['_n']=keys.groupby('week_start').cumcount()
    oo=oo.merge(keys[['week_start','_n','city']],on=['week_start','_n'],how='left',validate='one_to_one').drop(columns='_n')
    if oo.city.isna().any(): raise RuntimeError('Could not attach city to OOF scores')

    history=oo[oo.week_start<POLICY_EVAL_START].copy().sort_values('week_start')
    eval_rows=[]
    for week in sorted(oo.loc[oo.week_start.between(POLICY_EVAL_START,POLICY_EVAL_END),'week_start'].unique()):
        current=oo[oo.week_start==week].copy()
        if current.empty: continue
        pred=classify_week(current,history); eval_rows.append(pred)
        # At the next origin, the current target is realized and may enter calibration.
        history=pd.concat([history,current],ignore_index=True).sort_values('week_start')
    ev=pd.concat(eval_rows,ignore_index=True) if eval_rows else pd.DataFrame()
    if len(ev)<200: raise RuntimeError(f'Too few conformal policy evaluation rows: {len(ev)}')
    m=metrics(ev)
    contract=art['contract']
    gates={
        'red_precision':m['RED']['precision']>=contract['red_precision_min'],
        'red_fpr':m['RED']['FPR']<=contract['red_fpr_max'],
        'alert_recall':m['RED_plus_AMBER']['recall']>=contract['alert_recall_min'],
        'green_npv':m['GREEN']['NPV']>=contract['green_npv_min'],
        'has_nonzero_red_alerts':m['RED']['rows']>=3,
    }
    policy_artifact={
        'version':VERSION,'base_model_version':art['version'],'alpha_red_global':ALPHA_RED_GLOBAL,'alpha_red_city':ALPHA_RED_CITY,'alpha_green_decline':ALPHA_GREEN_DECLINE,
        'min_global_neg':MIN_GLOBAL_NEG,'min_global_pos':MIN_GLOBAL_POS,'min_city_neg':MIN_CITY_NEG,'calibration_lookback_weeks':CAL_LOOKBACK_WEEKS,
        'initial_oof_calibration_history':oo[oo.week_start<=POLICY_EVAL_END][['week_start','city','y','score']].copy(),
        'policy_semantics':{'RED':'reject non-decline globally and within city at fixed class-conditional significance levels','GREEN':'reject decline at fixed class-conditional significance level','AMBER':'abstain/monitor when neither class can be rejected safely'},
        'production_update_rule':'After each next week closes, append the previous origin score and realized label to rolling calibration history. Never use an unrealized future label.',
    }
    joblib.dump(policy_artifact,POLICY)
    report={'version':VERSION,'base_model':art['version'],'scientific_boundary':'Conformal policy parameters are fixed from desired class-conditional error budgets and evaluated only on historical 2025-H1 OOF. Fresh 2025-2026 outcomes are not used to choose alphas.','parameters':{'alpha_red_global':ALPHA_RED_GLOBAL,'alpha_red_city':ALPHA_RED_CITY,'alpha_green_decline':ALPHA_GREEN_DECLINE,'lookback_weeks':CAL_LOOKBACK_WEEKS},'historical_policy_evaluation':m,'contract':contract,'gates':gates,'all_policy_gates_passed':bool(all(gates.values())),'controls':{'class_conditional_conformal':True,'rolling_only_past_realized_labels':True,'city_specific_negative_calibration':True,'abstention_state_AMBER':True,'fixed_probability_thresholds_for_red_green':False,'fresh_2025_2026_used_to_select_policy_parameters':False}}
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8'); SUMMARY.write_text(f'''# SAMA City Risk v2.3 — Rolling Conformal Policy\n\n- Base model: **{art['version']}**\n- RED rule: global non-decline p ≤ **{ALPHA_RED_GLOBAL:.2%}** AND city non-decline p ≤ **{ALPHA_RED_CITY:.2%}**\n- GREEN rule: decline p ≤ **{ALPHA_GREEN_DECLINE:.2%}**\n- Otherwise: **AMBER / abstain**\n- Historical policy rows: **{m['rows']:,}**\n- RED precision: **{m['RED']['precision']:.2%}** ({m['RED']['TP']} TP / {m['RED']['FP']} FP)\n- RED FPR: **{m['RED']['FPR']:.2%}**\n- RED+AMBER recall: **{m['RED_plus_AMBER']['recall']:.2%}**\n- GREEN NPV: **{m['GREEN']['NPV']:.2%}**\n- Missed declines: **{m['GREEN']['FN']} / {m['declines']}**\n- All historical conformal-policy gates passed: **{report['all_policy_gates_passed']}**\n- Fresh 2025-2026 labels used to choose policy: **No**\n''',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
