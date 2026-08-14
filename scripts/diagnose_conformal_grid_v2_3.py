from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

import build_conformal_policy_v2_3 as c
import run_sama_city_risk_v2_1 as source
import train_sama_city_risk_v2_2 as v22

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'sama_city_v2_3'; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'conformal_grid_diagnostics.json'


def main():
    art=joblib.load(c.V22_MODEL)
    d,X=v22.featureize(source.reconciled_load_panel(c.HISTORY)); keep=d.week_start<=pd.Timestamp(art['development_end']); d=d.loc[keep].reset_index(drop=True); X=X.loc[keep].reset_index(drop=True)
    oo,fold_meta=c.oof_selected_with_city(d,X,art['selected']); oo['score']=art['calibrator'].predict_proba(pd.DataFrame({art['selected']:oo.raw_score}))[:,1]
    history=oo[oo.week_start<c.POLICY_EVAL_START].copy().sort_values(['week_start','city']); pvals=[]
    for week in sorted(oo.loc[oo.week_start.between(c.POLICY_EVAL_START,c.POLICY_EVAL_END),'week_start'].unique()):
        current=oo[oo.week_start==week].copy(); pvals.append(c.pvalue_week(current,history)); history=pd.concat([history,current],ignore_index=True).sort_values(['week_start','city'])
    pv=pd.concat(pvals,ignore_index=True); contract=art['contract']; rows=[]
    for ag in c.RED_GLOBAL_GRID:
        for ac in c.RED_CITY_GRID:
            for green in c.GREEN_GRID:
                m=c.metrics(c.apply_policy(pv,ag,ac,green)); gates={
                    'red_precision':m['RED']['precision']>=contract['red_precision_min'],
                    'red_fpr':m['RED']['FPR']<=contract['red_fpr_max'],
                    'alert_recall':m['RED_plus_AMBER']['recall']>=contract['alert_recall_min'],
                    'green_npv':m['GREEN']['NPV']>=contract['green_npv_min'],
                    'has_red':m['RED']['rows']>=3,
                }
                rows.append({'alpha_red_global':ag,'alpha_red_city':ac,'alpha_green':green,'passed':sum(gates.values()),'all':all(gates.values()),'gates':gates,'metrics':m})
    def key(r):
        m=r['metrics']; return (r['passed'],m['RED']['precision'],m['RED_plus_AMBER']['recall'],m['GREEN']['NPV'],-m['RED']['FPR'],m['RED']['rows'])
    top=sorted(rows,key=key,reverse=True)[:30]
    gate_pass_counts={g:sum(1 for r in rows if r['gates'][g]) for g in ['red_precision','red_fpr','alert_recall','green_npv','has_red']}
    # Pareto-style summaries for each operational concern.
    best_red_precision=max(rows,key=lambda r:(r['metrics']['RED']['precision'],r['metrics']['RED']['rows']))
    best_red_recall=max(rows,key=lambda r:(r['metrics']['RED']['recall_contribution'],r['metrics']['RED']['precision']))
    best_alert_recall=max(rows,key=lambda r:(r['metrics']['RED_plus_AMBER']['recall'],r['metrics']['GREEN']['NPV']))
    best_green_npv=max(rows,key=lambda r:(r['metrics']['GREEN']['NPV'],r['metrics']['GREEN']['rows']))
    report={
        'version':'SAMA-CITY-RISK-2.3-CONFORMAL-GRID-DIAGNOSIS',
        'diagnostic_source':'historical OOF through 2025-06-29 only; no fresh 2025-2026 labels',
        'rows':len(pv),'declines':int(pv.y.sum()),'decline_rate':float(pv.y.mean()),
        'contract':contract,'grid_candidates':len(rows),'fully_valid_candidates':sum(r['all'] for r in rows),
        'gate_pass_counts':gate_pass_counts,
        'top_30':top,
        'frontier_examples':{'best_red_precision':best_red_precision,'best_red_recall':best_red_recall,'best_alert_recall':best_alert_recall,'best_green_npv':best_green_npv},
        'folds':fold_meta,
    }
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'rows':report['rows'],'declines':report['declines'],'fully_valid':report['fully_valid_candidates'],'gate_pass_counts':gate_pass_counts,'top_10':top[:10],'frontier':report['frontier_examples']},indent=2))

if __name__=='__main__': main()
