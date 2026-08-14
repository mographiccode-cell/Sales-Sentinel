from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import run_sama_city_risk_v2_1 as source
import train_sama_city_risk_v2_2 as v22
import build_conformal_policy_v2_3 as conf

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'
FRESH=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2025_2026_holdout.csv'
BASE=ROOT/'models'/'sama_city_v2_2'/'city_market_risk_v2_2.joblib'
POLICY=ROOT/'models'/'sama_city_v2_3'/'conformal_policy_v2_3.joblib'
OUT=ROOT/'reports'/'sama_city_v2_3'/'post_diagnosis_prequential_stress.json'
SUMMARY=ROOT/'reports'/'sama_city_v2_3'/'post_diagnosis_prequential_stress.md'


def main():
    base=joblib.load(BASE); policy=joblib.load(POLICY)
    if base['version']!=v22.VERSION: raise RuntimeError('Unexpected base model')
    if not str(policy['version']).startswith('SAMA-CITY-RISK-2.3'): raise RuntimeError('Unexpected conformal policy')
    d,X=v22.featureize(source.reconciled_load_panel(EXT)); fresh=pd.read_csv(FRESH,parse_dates=['week_start']); fsets={c:set(q.week_start.dt.normalize()) for c,q in fresh.groupby('city')}
    mask=[]
    for w,c in zip(d.week_start,d.city):
        ww=pd.Timestamp(w).normalize(); weeks=sorted(fsets.get(c,set())); mask.append(bool(weeks) and ww in fsets[c] and ww!=weeks[-1])
    d=d.loc[mask].reset_index(drop=True); X=X.loc[mask].reset_index(drop=True)
    if len(d)<500: raise RuntimeError(f'Too few fresh stress rows: {len(d)}')
    raw=base['model'].predict_proba(X[base['features']])[:,1]
    score=base['calibrator'].predict_proba(pd.DataFrame({base['selected']:raw}))[:,1]
    current_all=d[['week_start','city','target']].rename(columns={'target':'y'}).copy(); current_all['score']=score

    history=policy['initial_oof_calibration_history'].copy(); history.week_start=pd.to_datetime(history.week_start); history=history.sort_values(['week_start','city'])
    results=[]; weekly_diagnostics=[]
    for week in sorted(current_all.week_start.unique()):
        current=current_all[current_all.week_start==week].copy()
        # Classification happens BEFORE the current week's realized target enters calibration history.
        pred=conf.classify_week(current,history); results.append(pred)
        weekly_diagnostics.append({'week_start':str(pd.Timestamp(week).date()),'rows':len(current),'red':int(pred.state.eq('RED').sum()),'amber':int(pred.state.eq('AMBER').sum()),'green':int(pred.state.eq('GREEN').sum()),'calibration_rows_before_prediction':int(len(history)),'calibration_positive_before_prediction':int((history.y==1).sum())})
        history=pd.concat([history,current],ignore_index=True).sort_values(['week_start','city'])
    ev=pd.concat(results,ignore_index=True); m=conf.metrics(ev)
    contract=base['contract']; gates={'red_precision':m['RED']['precision']>=contract['red_precision_min'],'red_fpr':m['RED']['FPR']<=contract['red_fpr_max'],'alert_recall':m['RED_plus_AMBER']['recall']>=contract['alert_recall_min'],'green_npv':m['GREEN']['NPV']>=contract['green_npv_min'],'has_red_alerts':m['RED']['rows']>=3}
    by_city={}
    for c,q in ev.groupby('city'): by_city[c]=conf.metrics(q)
    report={'version':'SAMA-CITY-RISK-2.3-PREQUENTIAL-STRESS','independence_status':'NOT AN INDEPENDENT HOLDOUT: the v2.3 architecture was designed after diagnosing aggregate v2.1/v2.2 behavior on this period. Policy parameters themselves were not fitted on these fresh labels.','base_model':base['version'],'policy':policy['version'],'rows':len(ev),'weeks':int(ev.week_start.nunique()),'period':{'start':str(ev.week_start.min().date()),'end':str(ev.week_start.max().date())},'prequential_semantics':'Each week is classified first. Only after that prediction is recorded is its realized label appended for use by later weeks.','metrics':m,'contract':contract,'gates':gates,'all_operational_gates_passed':bool(all(gates.values())),'by_city':by_city,'weekly_diagnostics':weekly_diagnostics,'anti_leakage':{'current_or_future_label_used_before_its_prediction':False,'fresh_labels_used_only_after_prediction_for_later_weeks':True,'base_model_retrained_on_fresh':False,'conformal_alpha_parameters_fitted_on_fresh':False}}
    OUT.write_text(json.dumps(report,indent=2,default=str),encoding='utf-8'); SUMMARY.write_text(f'''# SAMA City Risk v2.3 — Prequential Stress Test\n\n**Diagnostic stress test, not an independent fresh holdout.**\n\n- Rows: **{len(ev):,}** across **{ev.week_start.nunique()} weeks**\n- Declines: **{m['declines']} ({m['decline_rate']:.2%})**\n- RED precision: **{m['RED']['precision']:.2%}** ({m['RED']['TP']} TP / {m['RED']['FP']} FP)\n- RED FPR: **{m['RED']['FPR']:.2%}**\n- RED+AMBER recall: **{m['RED_plus_AMBER']['recall']:.2%}**\n- GREEN NPV: **{m['GREEN']['NPV']:.2%}**\n- Missed declines in GREEN: **{m['GREEN']['FN']} / {m['declines']}**\n- Operational gates passed: **{report['all_operational_gates_passed']}**\n- Current/future label used before prediction: **No**\n''',encoding='utf-8')
    print(json.dumps({'metrics':m,'gates':gates,'all':report['all_operational_gates_passed']},indent=2))

if __name__=='__main__':main()
