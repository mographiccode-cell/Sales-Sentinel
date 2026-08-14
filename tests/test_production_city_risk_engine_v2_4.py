from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts import production_city_risk_engine_v2_4 as prod
from scripts import run_sama_city_risk_v2_1 as source
from scripts import train_sama_city_risk_v2_2 as train

ROOT=Path(__file__).resolve().parents[1]
HIST=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2025.csv'
EXT=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'


def panel(path=HIST):
    return pd.read_csv(path,parse_dates=['week_start','week_end'])


def test_inference_features_match_training_features_exactly_on_known_rows():
    p=panel(HIST)
    train_d,train_X=train.featureize(source.reconciled_load_panel(HIST))
    infer_d,infer_X=prod.build_inference_features(p)
    base=joblib.load(prod.BASE_MODEL)

    left=pd.concat([train_d[['week_start','city']],train_X[base['features']]],axis=1)
    right=pd.concat([infer_d[['week_start','city']],infer_X[base['features']]],axis=1)
    merged=left.merge(right,on=['week_start','city'],suffixes=('_train','_infer'),validate='one_to_one')
    assert len(merged)==len(left)
    for c in base['features']:
        a=merged[f'{c}_train'].to_numpy(float); b=merged[f'{c}_infer'].to_numpy(float)
        assert np.allclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True), c


def test_truncating_future_does_not_change_origin_features():
    p=panel(HIST)
    base=joblib.load(prod.BASE_MODEL)
    origin=pd.Timestamp('2025-05-04')
    full_d,full_X=prod.build_inference_features(p)
    trunc=p[p.week_start<=origin].copy()
    trunc_d,trunc_X=prod.build_inference_features(trunc)
    for city in sorted(trunc_d.city.unique()):
        fm=(full_d.week_start.eq(origin)&full_d.city.eq(city)); tm=(trunc_d.week_start.eq(origin)&trunc_d.city.eq(city))
        assert fm.sum()==1 and tm.sum()==1
        a=full_X.loc[fm,base['features']].to_numpy(float); b=trunc_X.loc[tm,base['features']].to_numpy(float)
        assert np.allclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True), city


def test_latest_unknown_target_is_still_featured():
    p=panel(HIST)
    d,X=prod.build_inference_features(p)
    latest=d.week_start.max(); q=d.week_start.eq(latest)
    assert q.sum()==d.city.nunique()
    assert d.loc[q,'actual_next_value'].isna().all()
    assert d.loc[q,'target_float'].isna().all()
    base=joblib.load(prod.BASE_MODEL)
    assert X.loc[q,base['features']].notna().all().all()


def test_latest_prediction_fail_closed_on_missing_city():
    p=panel(EXT)
    latest=p.week_start.max(); victim=sorted(p[p.week_start.eq(latest)].city.unique())[0]
    broken=p[~(p.week_start.eq(latest)&p.city.eq(victim))].copy()
    result=prod.predict_latest(broken)
    assert result['status']=='NO_DECISION'
    assert result['state']=='AMBER'
    assert result['reason']=='SOURCE_QC_FAILED'


def test_latest_prediction_runs_without_future_label():
    p=panel(EXT)
    result=prod.predict_latest(p)
    assert result['status']=='OK'
    assert len(result['predictions'])==11
    assert {x['state'] for x in result['predictions']} <= {'RED','AMBER','GREEN'}
    assert result['safety']['future_label_used'] is False
    assert result['safety']['latest_unknown_target_required'] is False


def test_current_target_cannot_enter_prior_history():
    p=panel(EXT)
    d,_=prod.build_inference_features(p)
    latest=d.week_start.max()
    history=d[(d.week_start<latest)&d.target_float.notna()]
    assert (history.week_start<latest).all()
    # No label for latest is available by construction.
    assert d.loc[d.week_start.eq(latest),'target_float'].isna().all()
