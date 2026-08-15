from __future__ import annotations
import json
from pathlib import Path
import joblib, numpy as np
import train_sama_city_risk_v3 as base
import train_sama_city_risk_v3_4_1 as v

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'sama_city_v3_4_1'; OUT.mkdir(parents=True,exist_ok=True)

def main():
    a=joblib.load(v.BASE_MODEL)
    panel=base.source.reconciled_load_panel(base.HISTORY); d,X,P,pc=base.featureize(panel,require_target=True); keep=d.week_start<=base.DEV_END
    d=d.loc[keep].reset_index(drop=True); X=X.loc[keep].reset_index(drop=True); pc=pc.loc[keep].reset_index(drop=True)
    q,folds=v.build_oof(d,X,pc)
    s=q.trend_score.to_numpy(float); cand=np.unique(np.r_[np.quantile(s,np.linspace(.05,.999,250)),np.linspace(.02,.98,240)])
    rows=[]
    for emin in range(0,7):
        for t in cand:
            e=v.evaluate(q,a,float(t),emin)
            rows.append({'threshold':float(t),'evidence_min':emin,'recall':e['RED_plus_AMBER']['recall'],'precision':e['RED_plus_AMBER']['precision'],'NPV':e['RED_plus_AMBER']['NPV'],'alert_rate':e['alert_rate'],'green_coverage':e['green_coverage'],'inc_neg_rate':e['incremental_negative_alert_rate'],'stable_min':e['min_recall_folds_5plus'],'inc_tp':e['trend_incremental_tp'],'inc_alerts':e['trend_incremental_alerts']})
    feasible=[r for r in rows if r['alert_rate']<=.30 and r['green_coverage']>=.70 and r['inc_neg_rate']<=.05 and r['precision']>=.18]
    feasible_stable=[r for r in feasible if r['stable_min']>=.70]
    def best(arr,key):
        if not arr:return None
        return sorted(arr,key=lambda r:(r[key],r['NPV'],r['precision'],-r['alert_rate']),reverse=True)[0]
    rep={'rows_tested':len(rows),'feasible_basic':len(feasible),'feasible_with_stability':len(feasible_stable),'best_recall_basic':best(feasible,'recall'),'best_recall_stable':best(feasible_stable,'recall'),'best_npv_stable':best(feasible_stable,'NPV')}
    (OUT/'trend_pareto_diagnostic.json').write_text(json.dumps(rep,indent=2),encoding='utf-8'); print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
