from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models'/'merchant_total_triage_v4_4'/'merchant_total_triage_v4_4.joblib'
FEATURE_PANEL=ROOT/'data'/'merchant_v4_3'/'merchant_total_feature_panel_v4_3.csv'
ENGINE_VERSION='SALES-SENTINEL-MERCHANT-TRIAGE-SERVING-4.4'
DEPLOYMENT_STATUS='BLOCKED_PENDING_REAL_MERCHANT_VALIDATION'


def _load(path:Path=MODEL):
    if not path.exists():
        raise RuntimeError(f'v4.4 artifact missing: {path}')
    a=joblib.load(path)
    if a.get('version')!='SALES-SENTINEL-MERCHANT-TOTAL-TRIAGE-4.4':
        raise RuntimeError(f'Unexpected artifact version: {a.get("version")}')
    return a


def _schema_check(frame:pd.DataFrame,a:dict[str,Any]):
    expected=list(a['feature_columns'])
    missing=[c for c in expected if c not in frame.columns]
    forbidden=[c for c in frame.columns if c.lower().startswith('actual_') or 'future_ratio' in c.lower() or c.lower()=='target']
    if missing:
        return False,f'MISSING_FEATURES:{missing[:12]}',expected
    if forbidden:
        return False,f'FORBIDDEN_FUTURE_OR_LABEL_COLUMNS:{forbidden}',expected
    X=frame[expected].copy().replace([np.inf,-np.inf],np.nan)
    if X.isna().any().any():
        bad=X.columns[X.isna().any()].tolist()
        return False,f'NULL_OR_NONFINITE_FEATURES:{bad[:12]}',expected
    return True,'OK',expected


def predict_features(frame:pd.DataFrame,model_path:Path=MODEL)->dict[str,Any]:
    a=_load(model_path)
    ok,reason,expected=_schema_check(frame,a)
    if not ok:
        return {'engine_version':ENGINE_VERSION,'model_version':a['version'],'deployment_status':DEPLOYMENT_STATUS,'status':'NO_DECISION','reason':reason}
    X=frame[expected].copy()
    et=a['models']['extra_trees_reg'].predict(X)
    q25=a['models']['hist_gb_q25'].predict(X)
    early=.55*et+.45*q25
    severe=q25
    early_score=-early
    severe_score=-severe
    amber=early_score>=float(a['amber_threshold_score'])
    red_t=a.get('red_threshold_score')
    red=np.zeros(len(X),dtype=bool) if red_t is None else severe_score>=float(red_t)
    state=np.where(red,'RED',np.where(amber,'AMBER','GREEN'))
    rows=[]
    for i in range(len(X)):
        er=float(early[i]);sr=float(severe[i]);st=str(state[i])
        rows.append({
            'row':int(i),
            'state':st,
            'predicted_future_7d_ratio':er,
            'predicted_downside_pct':float(max(0.0,1.0-er)*100.0),
            'conservative_q25_future_ratio':sr,
            'conservative_downside_pct':float(max(0.0,1.0-sr)*100.0),
            'amber_threshold_score':float(a['amber_threshold_score']),
            'red_supported':bool(red_t is not None),
            'reason':('SEVERE_DOWNSIDE_THRESHOLD' if st=='RED' else 'EARLY_DOWNSIDE_THRESHOLD' if st=='AMBER' else 'NO_DOWNSIDE_THRESHOLD_CROSSED'),
        })
    return {
        'engine_version':ENGINE_VERSION,
        'model_version':a['version'],
        'deployment_status':DEPLOYMENT_STATUS,
        'status':'OK',
        'rows':rows,
        'safety':{
            'real_merchant_external_validation_completed':False,
            'red_supported':bool(red_t is not None),
            'future_label_feature_used':False,
            'future_actual_sama_feature_used':False,
            'stress_gate_passed':False,
            'website_integration_allowed':False,
        },
        'scientific_boundary':'Predictions are from a development model trained on UCI-derived Saudi-localized synthetic merchant microdata plus official aggregate SAMA context. Do not present these outputs as validated real-Saudi-merchant accuracy.'
    }


def predict_latest_evidence(panel_path:Path=FEATURE_PANEL)->dict[str,Any]:
    d=pd.read_csv(panel_path)
    if d.empty:
        return {'engine_version':ENGINE_VERSION,'deployment_status':DEPLOYMENT_STATUS,'status':'NO_DECISION','reason':'EMPTY_FEATURE_PANEL'}
    # Date/target columns are evidence metadata and intentionally removed before serving.
    drop=[c for c in ['date','future_ratio','target'] if c in d.columns]
    return predict_features(d.iloc[[-1]].drop(columns=drop))


def main():
    print(json.dumps(predict_latest_evidence(),indent=2))

if __name__=='__main__':
    main()
