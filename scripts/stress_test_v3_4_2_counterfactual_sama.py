from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from production_city_risk_engine_v3_4_2 import MODEL, predict_latest

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'
OUT=ROOT/'reports'/'sama_city_v3_4_2'/'counterfactual_stress'; OUT.mkdir(parents=True,exist_ok=True)
EXPECTED='SAMA-CITY-RISK-3.4.2-DOWNSIDE-RATIO'
CITIES=['ABHA','BURAIDAH','DAMMAM','HAIL','JEDDAH','KHOBAR','MADINA','MAKKAH','OTHER','RIYADH','TABOUK']

# New patterns: four visible precursor weeks, not used in any prior v3.3/v3.4 stress test.
PATTERNS={
 'slow_four_week':{
   'value':{-3:.99,-2:.96,-1:.92,0:.88},'count':{-3:.99,-2:.96,-1:.92,0:.88},'event_ratio':.73},
 'ticket_compression':{
   'value':{-3:.98,-2:.94,-1:.88,0:.83},'count':{-3:.995,-2:.99,-1:.98,0:.97},'event_ratio':.70},
 'volume_compression':{
   'value':{-3:.995,-2:.99,-1:.98,0:.97},'count':{-3:.98,-2:.94,-1:.88,0:.83},'event_ratio':.68},
 'accelerating_both':{
   'value':{-3:.995,-2:.97,-1:.91,0:.84},'count':{-3:.995,-2:.97,-1:.91,0:.84},'event_ratio':.71},
}

def pmap(res):
    if res.get('status')!='OK': raise RuntimeError(json.dumps(res))
    if res.get('model_version')!=EXPECTED: raise RuntimeError(f"serving mismatch {res.get('model_version')}")
    return {str(x['city']):x for x in res['predictions']}

def inject(panel,city,origin,pattern):
    cfg=PATTERNS[pattern]; d=panel.copy(); weeks=[pd.Timestamp(x) for x in sorted(d.week_start.unique())]; idx=weeks.index(pd.Timestamp(origin)); event_week=weeks[idx+1]
    for off in (-3,-2,-1,0):
        w=weeks[idx+off]; m=d.week_start.eq(w)&d.city.eq(city)
        if int(m.sum())!=1: raise RuntimeError(f'missing {city} {w}')
        d.loc[m,'value_thousand_sar']*=cfg['value'][off]; d.loc[m,'transaction_count_thousand']*=cfg['count'][off]
    cr=d[d.city.eq(city)].sort_values('week_start').reset_index(drop=True); pos=int(cr.index[cr.week_start.eq(origin)][0]); bval=float(cr.loc[pos-3:pos,'value_thousand_sar'].mean()); bcnt=float(cr.loc[pos-3:pos,'transaction_count_thousand'].mean()); m=d.week_start.eq(event_week)&d.city.eq(city)
    d.loc[m,'value_thousand_sar']=bval*cfg['event_ratio']; d.loc[m,'transaction_count_thousand']=bcnt*cfg['event_ratio']
    return d,{'city':city,'origin':str(origin.date()),'event_week':str(event_week.date()),'pattern':pattern,'event_ratio':float(cfg['event_ratio'])}

def main():
    a=joblib.load(MODEL)
    if a.get('version')!=EXPECTED: raise RuntimeError(f'wrong artifact {a.get("version")}')
    panel=pd.read_csv(DATA,parse_dates=['week_start','week_end']).sort_values(['week_start','city']).reset_index(drop=True)
    weeks=[pd.Timestamp(x) for x in sorted(panel.week_start.unique())]; eligible=[w for w in weeks if pd.Timestamp('2025-09-07')<=w<=pd.Timestamp('2026-06-07')]
    if len(eligible)<36: raise RuntimeError(f'not enough eligible weeks {len(eligible)}')
    positions=np.linspace(4,len(eligible)-3,44,dtype=int); origins=[eligible[i] for i in positions]; pnames=list(PATTERNS); scenarios=[(CITIES[i%11],origins[i],pnames[i//11]) for i in range(44)]
    erows=[]; crows=[]
    for sid,(city,origin,pat) in enumerate(scenarios,1):
        bh=panel[panel.week_start<=origin][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']].copy(); bm=pmap(predict_latest(bh))
        inj,truth=inject(panel,city,origin,pat); ih=inj[inj.week_start<=origin][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']].copy(); im=pmap(predict_latest(ih)); b=bm[city]; p=im[city]
        erows.append({'scenario_id':sid,**truth,'baseline_state':b['state'],'baseline_score':b['risk_score'],'baseline_pred_ratio':b['predicted_next_week_ratio'],'injected_state':p['state'],'injected_reason':p['reason'],'injected_score':p['risk_score'],'injected_pred_ratio':p['predicted_next_week_ratio'],'ratio_lift':p['predicted_next_week_ratio']-b['predicted_next_week_ratio'],'risk_lift':p['risk_score']-b['risk_score'],'trend_evidence_count':p['trend_evidence_count'],'precursor_count':p['precursor_count'],'ood_fraction':p['ood_fraction'],'alerted':int(p['state'] in {'RED','AMBER'}),'red':int(p['state']=='RED'),'ratio_warning':int(p['reason']=='DOWNSIDE_RATIO_FORECAST'),'ood_abstain':int(p['reason']=='OOD_ABSTAIN')})
        for other in CITIES:
            if other==city: continue
            bp=bm[other]; ip=im[other]; crows.append({'scenario_id':sid,'origin':str(origin.date()),'pattern':pat,'injected_city':city,'control_city':other,'baseline_state':bp['state'],'injected_state':ip['state'],'baseline_score':bp['risk_score'],'injected_score':ip['risk_score'],'baseline_pred_ratio':bp['predicted_next_week_ratio'],'injected_pred_ratio':ip['predicted_next_week_ratio'],'new_red':int(bp['state']!='RED' and ip['state']=='RED'),'new_alert':int(bp['state']=='GREEN' and ip['state'] in {'RED','AMBER'}),'score_change':ip['risk_score']-bp['risk_score'],'ratio_change':ip['predicted_next_week_ratio']-bp['predicted_next_week_ratio']})
    e=pd.DataFrame(erows); c=pd.DataFrame(crows); stats={}
    for pat,z in e.groupby('pattern'):
        stats[pat]={'n':len(z),'alerted':int(z.alerted.sum()),'recall':float(z.alerted.mean()),'red':int(z.red.sum()),'ratio_warnings':int(z.ratio_warning.sum()),'ood_abstentions':int(z.ood_abstain.sum()),'median_pred_ratio_change':float(z.ratio_lift.median()),'median_risk_lift':float(z.risk_lift.median())}
    recall=float(e.alerted.mean()); cr=float(c.new_red.mean()); ca=float(c.new_alert.mean()); ood=float(e.ood_abstain.mean())
    acceptance={'overall_recall_ge_90pct':recall>=.90,'each_pattern_recall_ge_80pct':all(x['recall']>=.80 for x in stats.values()),'control_new_red_rate_le_1pct':cr<=.01,'control_new_alert_rate_le_5pct':ca<=.05,'ood_abstain_rate_le_30pct':ood<=.30,'all_events_gt20pct':bool((e.event_ratio<.80).all()),'serving_exact_v3_4_2':a.get('version')==EXPECTED}
    rep={'version':'SAMA-CITY-V3.4.2-COUNTERFACTUAL-NEW-1','frozen_model':EXPECTED,'model_development_end':a.get('development_end'),'scenario_count':len(e),'control_rows':len(c),'patterns':PATTERNS,'pattern_results':stats,'overall':{'alerted':int(e.alerted.sum()),'recall':recall,'RED':int(e.red.sum()),'AMBER':int(e.injected_state.eq('AMBER').sum()),'GREEN':int(e.injected_state.eq('GREEN').sum()),'ratio_warnings':int(e.ratio_warning.sum()),'ood_abstentions':int(e.ood_abstain.sum()),'median_pred_ratio_change':float(e.ratio_lift.median()),'median_risk_lift':float(e.risk_lift.median())},'controls':{'new_red':int(c.new_red.sum()),'new_red_rate':cr,'new_alert':int(c.new_alert.sum()),'new_alert_rate':ca,'max_abs_score_change':float(c.score_change.abs().max()),'max_abs_ratio_change':float(c.ratio_change.abs().max())},'acceptance':acceptance,'all_acceptance_passed':bool(all(acceptance.values())),'scientific_boundary':'Frozen v3.4.2; no fitting in this script. These four injection patterns were not used in v3.4.2 training or prior v3.3/v3.4 stress tests.'}
    e.to_csv(OUT/'injected_scenarios.csv',index=False); c.to_csv(OUT/'unaffected_controls.csv',index=False); (OUT/'report.json').write_text(json.dumps(rep,indent=2),encoding='utf-8'); (OUT/'summary.md').write_text('# v3.4.2 New Counterfactual Stress\n\n'+f'- Overall recall **{recall:.2%}** ({int(e.alerted.sum())}/{len(e)})\n'+''.join(f'- {k}: **{v["recall"]:.2%}** ({v["alerted"]}/{v["n"]})\n' for k,v in stats.items())+f'- Control new RED **{int(c.new_red.sum())}/{len(c)} ({cr:.2%})**\n- Control new alerts **{int(c.new_alert.sum())}/{len(c)} ({ca:.2%})**\n- All gates **{rep["all_acceptance_passed"]}**\n',encoding='utf-8'); print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
