from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ROOT=Path(__file__).resolve().parents[1]
SECTORS=ROOT/'data'/'sama_pos'/'sama_sector_weekly_value_count_2020_2026_extended.csv'
NAT_VALUE=ROOT/'data'/'sama_pos'/'sama_pos_national_weekly_value_2020_2026_extended.csv'
NAT_COUNT=ROOT/'data'/'sama_pos'/'sama_pos_national_weekly_count_2020_2026_extended.csv'
HOLDOUT=ROOT/'data'/'sama_pos'/'sama_sector_weekly_value_count_2025_2026_holdout.csv'
OUT=ROOT/'data'/'sama_pos'/'sama_sector_walkforward_forecasts_2025_2026_v2_0.csv'
REPORT_DIR=ROOT/'reports'/'sales_sentinel_v2_0'; REPORT_DIR.mkdir(parents=True,exist_ok=True)
REPORT=REPORT_DIR/'extended_forecaster_report.json'
SEED=42


def model():
    # Frozen v1.7 hyperparameters; not retuned on holdout.
    return HistGradientBoostingRegressor(
        learning_rate=.045,max_iter=220,max_leaf_nodes=18,min_samples_leaf=18,
        l2_regularization=2.5,random_state=SEED,
    )


def wape(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return float(np.abs(y-p).sum()/max(np.abs(y).sum(),1e-9))


def build_features(d,nat):
    x=d.copy().sort_values(['sector','week_start']).reset_index(drop=True)
    g=x.groupby('sector',sort=False,group_keys=False)
    for col,prefix in [('value_thousand_sar','value'),('transaction_count_thousand','count')]:
        log=np.log1p(x[col].astype(float)); x[f'log_{prefix}_t0']=log
        for lag in (1,2,3,4,8,13,26,52): x[f'log_{prefix}_lag_{lag}']=g[col].shift(lag).pipe(np.log1p)
        for w in (4,8,13,26,52):
            x[f'log_{prefix}_mean_{w}']=g[col].transform(lambda s,w=w:np.log1p(s).rolling(w,min_periods=w).mean())
            x[f'log_{prefix}_std_{w}']=g[col].transform(lambda s,w=w:np.log1p(s).rolling(w,min_periods=w).std())
        x[f'{prefix}_change_1']=g[col].pct_change(1); x[f'{prefix}_change_4']=g[col].pct_change(4); x[f'{prefix}_change_13']=g[col].pct_change(13)

    n=nat.copy().sort_values('week_start')
    n['log_national_value']=np.log1p(n.national_value.astype(float)); n['log_national_count']=np.log1p(n.national_count.astype(float))
    for c in ['log_national_value','log_national_count']:
        for lag in (1,4,13,52): n[f'{c}_lag_{lag}']=n[c].shift(lag)
    x=x.merge(n.drop(columns=['national_value','national_count']),on='week_start',how='left',validate='many_to_one')
    week=x.week_start.dt.isocalendar().week.astype(float); x['week_sin']=np.sin(2*np.pi*week/52.18); x['week_cos']=np.cos(2*np.pi*week/52.18)
    x=pd.concat([x,pd.get_dummies(x[['sector']],prefix='sector',dtype=float)],axis=1)
    gx=x.groupby('sector',sort=False)
    x['value_h1']=gx.value_thousand_sar.shift(-1).pipe(np.log1p); x['value_h2']=gx.value_thousand_sar.shift(-2).pipe(np.log1p)
    x['count_h1']=gx.transaction_count_thousand.shift(-1).pipe(np.log1p); x['count_h2']=gx.transaction_count_thousand.shift(-2).pipe(np.log1p)
    targets=['value_h1','value_h2','count_h1','count_h2']
    exclude={'week_start','week_end','sector','value_thousand_sar','transaction_count_thousand',*targets}
    fcols=[c for c in x.columns if c not in exclude]
    return x.replace([np.inf,-np.inf],np.nan),fcols,targets


def main():
    d=pd.read_csv(SECTORS,parse_dates=['week_start','week_end'])
    h=pd.read_csv(HOLDOUT,parse_dates=['week_start','week_end'])
    safe=sorted(h.sector.unique())
    d=d[d.sector.isin(safe)].copy().sort_values(['sector','week_start'])
    v=pd.read_csv(NAT_VALUE,parse_dates=['week_start'])[['week_start','value_thousand_sar']].rename(columns={'value_thousand_sar':'national_value'})
    c=pd.read_csv(NAT_COUNT,parse_dates=['week_start'])[['week_start','transaction_count']].rename(columns={'transaction_count':'national_count'})
    nat=v.merge(c,on='week_start',how='inner').sort_values('week_start')

    # Continuity is mandatory: groupby-shift must mean a calendar week, never the previous available record after a gap.
    continuity={}
    for sector,q in d.groupby('sector'):
        q=q[q.week_start>=pd.Timestamp('2025-05-01')].sort_values('week_start')
        max_gap=int(q.week_start.diff().dt.days.dropna().max()) if len(q)>1 else 999
        continuity[sector]=max_gap
    nat_recent=nat[nat.week_start>=pd.Timestamp('2025-05-01')]
    nat_gap=int(nat_recent.week_start.diff().dt.days.dropna().max()) if len(nat_recent)>1 else 999
    if max(continuity.values())>7 or nat_gap>7:
        raise RuntimeError(f'Weekly continuity gate failed sectors={continuity}, national_gap={nat_gap}')

    x,fcols,targets=build_features(d,nat); complete=x[fcols].notna().all(axis=1)
    origins=sorted(x.loc[x.week_start>=pd.Timestamp('2025-06-01'),'week_start'].unique())
    rows=[]; cache={}
    for oi,origin in enumerate(origins):
        origin=pd.Timestamp(origin); anchor=pd.Timestamp(origins[(oi//4)*4])
        current=x[(x.week_start==origin)&complete].copy()
        if current.empty: continue
        preds={}
        for target in targets:
            horizon=2 if target.endswith('h2') else 1; key=(anchor,target)
            fit=cache.get(key)
            if fit is None:
                known=x.week_start+pd.to_timedelta(7*horizon,unit='D')<=anchor
                tr=x[complete & x[target].notna() & known & (x.week_start<anchor)]
                if len(tr)<1000: raise RuntimeError(f'Insufficient training rows at {anchor}/{target}: {len(tr)}')
                fit=model().fit(tr[fcols],tr[target]); cache[key]=fit
            preds[target]=np.expm1(fit.predict(current[fcols]))
        for j,(_,r) in enumerate(current.iterrows()):
            sector=r.sector
            def actual(h,col):
                z=d[(d.sector==sector)&(d.week_start==origin+pd.Timedelta(days=7*h))]
                return float(z.iloc[0][col]) if len(z) else np.nan
            rows.append({
                'origin_week_start':origin,'sector':sector,
                'forecast_h1_week_start':origin+pd.Timedelta(days=7),'forecast_h2_week_start':origin+pd.Timedelta(days=14),
                'predicted_value_h1':float(preds['value_h1'][j]),'predicted_value_h2':float(preds['value_h2'][j]),
                'predicted_count_h1':float(preds['count_h1'][j]),'predicted_count_h2':float(preds['count_h2'][j]),
                'actual_value_h1':actual(1,'value_thousand_sar'),'actual_value_h2':actual(2,'value_thousand_sar'),
                'actual_count_h1':actual(1,'transaction_count_thousand'),'actual_count_h2':actual(2,'transaction_count_thousand'),
            })
    out=pd.DataFrame(rows).sort_values(['origin_week_start','sector']); out.to_csv(OUT,index=False)
    hold_start=h.week_start.min(); e=out[(out.origin_week_start>=hold_start)&out.actual_value_h1.notna()].copy()
    metrics={}
    for name,a,pc in [('value_h1','actual_value_h1','predicted_value_h1'),('count_h1','actual_count_h1','predicted_count_h1')]:
        q=e[[a,pc]].dropna(); metrics[name]={'rows':int(len(q)),'MAE':float(mean_absolute_error(q[a],q[pc])),'WAPE':wape(q[a],q[pc]),'correlation':float(q[a].corr(q[pc]))}
    report={
        'version':'SAMA-EXTENDED-FORECASTER-2.0','frozen_hyperparameters_from':'SAMA-SECTOR-FORECASTER-1.7',
        'safe_sectors':safe,'holdout_forecast_rows':int(len(e)),'holdout_origin_start':str(e.origin_week_start.min().date()),'holdout_origin_end':str(e.origin_week_start.max().date()),
        'continuity_max_gap_days_by_sector':continuity,'national_max_gap_days':nat_gap,'metrics':metrics,
        'leakage_controls':{'future_actuals_not_features':True,'target_must_be_known_by_batch_anchor':True,'expanding_history_including_only_prior_holdout_weeks_is_operationally_allowed':True,'hyperparameters_not_retuned_on_holdout':True},
    }
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
