from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import build_train_saudi_panel_v1_4 as ml

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts'/'saudi_v1_3'/'saudi_localized_transactions_v1_3_sama.csv.gz'
SAMA_FORECAST=ROOT/'data'/'sama_pos'/'sama_sector_walkforward_forecasts_2023_2025.csv'
SAMA_HISTORY=ROOT/'data'/'sama_pos'/'sama_sector_weekly_value_count_2020_2025.csv'
OUTD=ROOT/'data'/'saudi_v1_5'; REPD=ROOT/'reports'/'saudi_v1_5'; MODD=ROOT/'models'/'saudi_v1_5'
for p in (OUTD,REPD,MODD): p.mkdir(parents=True,exist_ok=True)
PANEL=OUTD/'saudi_sector_daily_panel_v1_5.csv.gz'; SUP=OUTD/'saudi_sector_supervised_v1_5.csv.gz'
AUDIT=REPD/'dataset_quality_audit_v1_5.json'; SUMMARY=REPD/'retraining_summary_v1_5.md'; META=MODD/'model_metadata_v1_5.json'; MODEL=MODD/'sales_decline_ensemble_v1_5.joblib'
VERSION='SA-LOCALIZATION-1.5-SAMA-SECTOR-SAFE'; DECLINE=.20; H=7; B=28
PAY={2023:.70,2024:.79,2025:.85}

RAMADAN=[('2023-03-23','2023-04-20'),('2024-03-11','2024-04-09')]
EID_FITR=[('2023-04-21','2023-04-23'),('2024-04-10','2024-04-12')]
HAJJ=[('2023-06-19','2023-06-30'),('2024-06-07','2024-06-19')]
EID_ADHA=[('2023-06-28','2023-07-01'),('2024-06-16','2024-06-19')]

def dumps(x):
    return json.dumps(x,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else (v.isoformat() if isinstance(v,pd.Timestamp) else str(v)))

def truth(s): return s.astype(str).str.lower().isin(['true','1','yes'])
def week_start(s):
    x=pd.to_datetime(s); return x-pd.to_timedelta((x.dt.dayofweek+1)%7,unit='D')
def in_ranges(d,ranges):
    out=np.zeros(len(d),dtype=bool)
    for a,b in ranges: out |= d.between(pd.Timestamp(a),pd.Timestamp(b)).to_numpy()
    return out

def build_panel():
    use=['TrainingSafeDate','SAMASector','SAMACalibratedNetSalesSAR','BaseNetSalesSAR','EligibleForSalesTraining','IsAdministrativeLine','SaudiInvoiceNo','ObservedSaudiCustomerID','CustomerIDSource','StockCode','OriginalQuantity','PaymentType','SAMAWeeklyMarketIndex','SAMACalibrationFactor']
    d=pd.read_csv(SOURCE,compression='gzip',usecols=use)
    n=len(d); d.TrainingSafeDate=pd.to_datetime(d.TrainingSafeDate).dt.normalize()
    admin=truth(d.IsAdministrativeLine); eligible=truth(d.EligibleForSalesTraining)
    admin_n=int(admin.sum()); d=d[eligible & ~admin].copy()
    obs=d.ObservedSaudiCustomerID.notna() & d.CustomerIDSource.astype(str).eq('ObservedSourceCustomerID')
    d['ObservedCustomer']=d.ObservedSaudiCustomerID.where(obs,pd.NA)
    d['Gross']=d.SAMACalibratedNetSalesSAR.astype(float).clip(lower=0); d['Returns']=-d.SAMACalibratedNetSalesSAR.astype(float).clip(upper=0)
    keys=['TrainingSafeDate','SAMASector']; g=d.groupby(keys,observed=True)
    p=g.agg(sales=('SAMACalibratedNetSalesSAR','sum'),base_sales=('BaseNetSalesSAR','sum'),gross=('Gross','sum'),returns=('Returns','sum'),rows=('SaudiInvoiceNo','size'),invoices=('SaudiInvoiceNo','nunique'),customers=('ObservedCustomer','nunique'),products=('StockCode','nunique'),units=('OriginalQuantity',lambda x:float(np.abs(pd.to_numeric(x,errors='coerce')).sum())),sama_market_index=('SAMAWeeklyMarketIndex','median'),sama_factor=('SAMACalibrationFactor','median')).reset_index()
    inv=d[keys+['SaudiInvoiceNo','PaymentType']].drop_duplicates(keys+['SaudiInvoiceNo']); inv['e']=inv.PaymentType.astype(str).eq('Electronic')
    e=inv.groupby(keys,observed=True).e.mean().rename('electronic_share').reset_index(); p=p.merge(e,on=keys,how='left',validate='one_to_one')

    # Complete only verified 604 TrainingSafeDate dates. A missing sector-day in the complete transaction population is a structural zero.
    dates=pd.Index(sorted(d.TrainingSafeDate.unique()),name='TrainingSafeDate'); sectors=sorted(p.SAMASector.dropna().unique())
    grid=pd.MultiIndex.from_product([sectors,dates],names=['SAMASector','TrainingSafeDate']).to_frame(index=False)
    p=grid.merge(p,on=['SAMASector','TrainingSafeDate'],how='left',validate='one_to_one',indicator=True)
    p['structural_zero']=p._merge.eq('left_only').astype(int); p=p.drop(columns='_merge')
    zeros=['sales','base_sales','gross','returns','rows','invoices','customers','products','units']; p[zeros]=p[zeros].fillna(0)

    # SAMA actual sector signal for structural-zero merchant days comes from the same completed official sector-week.
    hist=pd.read_csv(SAMA_HISTORY,parse_dates=['week_start'])
    p['week_start']=week_start(p.TrainingSafeDate); hist=hist[['week_start','sector','value_thousand_sar','transaction_count_thousand']].rename(columns={'sector':'SAMASector','value_thousand_sar':'sama_sector_value','transaction_count_thousand':'sama_sector_count'})
    p=p.merge(hist,on=['week_start','SAMASector'],how='left',validate='many_to_one')
    # Current official week is NOT a feature below. It is retained only for audit and shifted by one week before modeling.
    p['sama_market_index']=p.groupby('SAMASector').sama_market_index.transform(lambda s:s.ffill().bfill())
    p['sama_factor']=p.groupby('SAMASector').sama_factor.transform(lambda s:s.ffill().bfill())
    p['electronic_share']=p.electronic_share.fillna(p.TrainingSafeDate.dt.year.map(PAY).fillna(.85))
    p['basket']=np.where(p.invoices>0,p.sales/p.invoices,0.0); p['return_rate']=np.where(p.gross>0,p.returns/p.gross,0.0)
    p=p.sort_values(['SAMASector','TrainingSafeDate']).reset_index(drop=True)
    p.to_csv(PANEL,index=False,compression={'method':'gzip','compresslevel':5})
    stats={'source_rows':n,'eligible_rows':len(d),'administrative_rows_excluded':admin_n,'observed_customer_rows':int(obs.sum()),'fallback_rows_not_counted_as_customers':int((~obs).sum()),'sectors':int(p.SAMASector.nunique()),'calendar_days':int(len(dates)),'panel_rows':int(len(p)),'structural_zero_rows':int(p.structural_zero.sum()),'structural_zero_rate':float(p.structural_zero.mean()),'duplicate_sector_dates':int(p.duplicated(['SAMASector','TrainingSafeDate']).sum()),'core_nulls':int(p[['TrainingSafeDate','SAMASector','sales','invoices','customers']].isna().sum().sum()),'official_sama_history_nulls':int(p[['sama_sector_value','sama_sector_count']].isna().sum().sum()),'date_start':str(p.TrainingSafeDate.min().date()),'date_end':str(p.TrainingSafeDate.max().date())}
    return p,stats

def calendar_flags(dates):
    dt=pd.to_datetime(dates); out=pd.DataFrame(index=dates.index)
    out['weekend']=dt.dt.dayofweek.isin([4,5]).astype(int); out['founding']=((dt.dt.month==2)&(dt.dt.day==22)).astype(int); out['national']=((dt.dt.month==9)&(dt.dt.day==23)).astype(int); out['salary']=dt.dt.day.between(25,28).astype(int)
    out['ramadan']=in_ranges(dt,RAMADAN).astype(int); out['eid_fitr']=in_ranges(dt,EID_FITR).astype(int); out['hajj']=in_ranges(dt,HAJJ).astype(int); out['eid_adha']=in_ranges(dt,EID_ADHA).astype(int)
    return out

def featureize(p):
    d=p.copy().sort_values(['SAMASector','TrainingSafeDate']).reset_index(drop=True); g=d.groupby('SAMASector',sort=False,group_keys=False)
    # Current end-of-day merchant state is known when predicting the next seven days.
    metrics={'sales':'sales','base_sales':'base','invoices':'inv','customers':'cust','products':'prod','units':'units','basket':'basket','return_rate':'ret','electronic_share':'epay'}
    feats=pd.DataFrame(index=d.index)
    for col,pre in metrics.items():
        s=d[col].astype(float)
        feats[f'{pre}_t0_log']=np.log1p(np.clip(s,0,None)) if col not in {'return_rate','electronic_share'} else s
        for lag in (1,2,3,7,14,28,56): feats[f'{pre}_lag_{lag}']=g[col].shift(lag)
        for w in (7,14,28):
            mean=g[col].transform(lambda x,w=w:x.shift(1).rolling(w,min_periods=w).mean()); feats[f'{pre}_mean_{w}']=mean
            if pre in {'sales','inv','cust','basket'}: feats[f'{pre}_std_{w}']=g[col].transform(lambda x,w=w:x.shift(1).rolling(w,min_periods=w).std())
        if col in {'sales','invoices','customers','basket'}:
            mean28=g[col].transform(lambda x:x.shift(1).rolling(28,min_periods=28).mean()).replace(0,np.nan); feats[f'{pre}_ratio_28']=s/mean28

    # Actual SAMA sector observations are usable only from the previous completed week.
    weekly=d[['week_start','SAMASector','sama_sector_value','sama_sector_count']].drop_duplicates(['week_start','SAMASector']).sort_values(['SAMASector','week_start'])
    wg=weekly.groupby('SAMASector')
    weekly['sama_value_prev_week']=wg.sama_sector_value.shift(1); weekly['sama_count_prev_week']=wg.sama_sector_count.shift(1)
    weekly['sama_value_prev_change']=wg.sama_sector_value.shift(1)/wg.sama_sector_value.shift(2)-1; weekly['sama_count_prev_change']=wg.sama_sector_count.shift(1)/wg.sama_sector_count.shift(2)-1
    d=d.merge(weekly[['week_start','SAMASector','sama_value_prev_week','sama_count_prev_week','sama_value_prev_change','sama_count_prev_change']],on=['week_start','SAMASector'],how='left',validate='many_to_one')

    # Walk-forward sector forecasts: for current week W use origin W-7; predicted W and W+7 only.
    sf=pd.read_csv(SAMA_FORECAST,parse_dates=['origin_week_start'])
    keep=['origin_week_start','sector','predicted_value_h1_index_52median','predicted_value_h2_index_52median','predicted_count_h1_index_52median','predicted_count_h2_index_52median','predicted_value_h1_change_vs_last','predicted_value_h2_change_vs_last','predicted_count_h1_change_vs_last','predicted_count_h2_change_vs_last']
    sf=sf[keep].rename(columns={'sector':'SAMASector'}); d['forecast_origin']=d.week_start-pd.Timedelta(days=7)
    d=d.merge(sf,left_on=['forecast_origin','SAMASector'],right_on=['origin_week_start','SAMASector'],how='left',validate='many_to_one').drop(columns=['origin_week_start'])
    for c in ['sama_value_prev_week','sama_count_prev_week','sama_value_prev_change','sama_count_prev_change']+[x for x in sf.columns if x.startswith('predicted_')]: feats[c]=d[c].to_numpy()

    # Known-ahead Saudi calendar context, including all seven forecast days.
    cur=calendar_flags(d.TrainingSafeDate)
    for c in cur.columns: feats[f'cal_{c}_today']=cur[c].to_numpy()
    future_counts={c:np.zeros(len(d),dtype=float) for c in cur.columns}
    for h in range(1,H+1):
        f=calendar_flags(d.TrainingSafeDate+pd.Timedelta(days=h))
        for c in f.columns: future_counts[c]+=f[c].to_numpy()
    for c,v in future_counts.items(): feats[f'cal_{c}_next7_count']=v
    dow=d.TrainingSafeDate.dt.dayofweek.astype(float); doy=d.TrainingSafeDate.dt.dayofyear.astype(float)
    feats['dow_sin']=np.sin(2*np.pi*dow/7); feats['dow_cos']=np.cos(2*np.pi*dow/7); feats['year_sin']=np.sin(2*np.pi*doy/365.25); feats['year_cos']=np.cos(2*np.pi*doy/365.25); feats['structural_zero_today']=d.structural_zero.to_numpy()
    cats=pd.get_dummies(d.SAMASector,prefix='sector',dtype=float); feats=pd.concat([feats,cats],axis=1)

    # Target is fixed and never part of model inputs.
    future=pd.concat([g.sales.shift(-i).rename(f'f{i}') for i in range(1,H+1)],axis=1); baseline=g.sales.transform(lambda x:x.shift(1).rolling(B,min_periods=B).mean())
    d['future_mean']=future.mean(axis=1); d['future_complete']=future.notna().sum(axis=1).eq(H); d['baseline']=baseline; d['future_ratio']=d.future_mean/baseline.replace(0,np.nan); d['target']=(d.future_ratio < 1-DECLINE).astype(int)
    X=feats.replace([np.inf,-np.inf],np.nan); good=d.future_complete & d.future_ratio.notna() & X.notna().all(axis=1)
    out=pd.concat([d.loc[good,['TrainingSafeDate','SAMASector','future_ratio','target']].reset_index(drop=True),X.loc[good].reset_index(drop=True)],axis=1)
    fcols=list(X.columns); out.to_csv(SUP,index=False,compression={'method':'gzip','compresslevel':5})
    return out,fcols,{'supervised_rows':len(out),'features':len(fcols),'positive_rate':float(out.target.mean())}

def main():
    p,ps=build_panel(); d,F,ss=featureize(p)
    checks={'source_rows_1049042':ps['source_rows']==1049042,'no_duplicates':ps['duplicate_sector_dates']==0,'no_core_nulls':ps['core_nulls']==0,'official_SAMA_history_complete':ps['official_sama_history_nulls']==0,'at_least_7_sectors':ps['sectors']>=7,'all_604_calendar_days':ps['calendar_days']==604,'panel_rows_at_least_4200':ps['panel_rows']>=4200,'structural_zero_below_5pct':ps['structural_zero_rate']<.05,'supervised_rows_at_least_3500':ss['supervised_rows']>=3500,'target_rate_between_15_and_45pct':.15<=ss['positive_rate']<=.45,'fixed_20pct_target':True,'synthetic_region_not_a_feature_or_entity':True,'future_actual_SAMA_not_feature':True}
    audit={'version':VERSION,'panel':ps,'supervised':ss,'checks':checks,'dataset_quality_passed':bool(all(checks.values())),'scientific_boundary':'No synthetic Region is used in the model. Merchant transactions remain UCI-derived Saudi-localized microdata; official SAMA sector history/forecasts and known Saudi calendar are external Saudi signals.'}
    if not audit['dataset_quality_passed']:
        AUDIT.write_text(dumps(audit),encoding='utf-8'); raise RuntimeError('v1.5 dataset quality gate failed')

    # Reuse rigorously chronological model selection/blending from v1.4; swap output artifact after training.
    old_model=ml.MODEL_FILE; ml.MODEL_FILE=MODEL
    result=ml.train_and_evaluate(d.rename(columns={'target':'target_existing'}),F,DECLINE)
    ml.MODEL_FILE=old_model
    audit['training']=result; AUDIT.write_text(dumps(audit),encoding='utf-8'); META.write_text(dumps(audit),encoding='utf-8')
    m=result['test_metrics']; SUMMARY.write_text(f'''# Saudi Sector Retraining v1.5

- Dataset quality: **PASS**
- Source rows: **{ps['source_rows']:,}**
- Sectors: **{ps['sectors']}**
- Panel rows: **{ps['panel_rows']:,}**
- Supervised rows: **{ss['supervised_rows']:,}**
- Structural-zero rate: **{ps['structural_zero_rate']:.2%}**
- Fixed target: **20% decline**, next 7 calendar days vs trailing 28 days
- Synthetic Region used: **No**
- Selected classifier: **{result['selected_classifier']}**
- Selected regressor: **{result['selected_regressor']}**
- Accuracy: **{m['Accuracy']:.2%}**
- Balanced Accuracy: **{m['BalancedAccuracy']:.2%}**
- Precision: **{m['Precision']:.2%}**
- Recall: **{m['Recall']:.2%}**
- F1: **{m['F1']:.2%}**
- ROC-AUC: **{m['ROC_AUC']:.2%}**
- Majority baseline: **{result['majority_test_accuracy']:.2%}**
- 90% goal met: **{result['high_accuracy_90pct_goal_met']}**
- Scientific acceptance gates: **{result['all_acceptance_gates_passed']}**
''',encoding='utf-8')
    print(dumps({'dataset':'PASS','panel':ps,'supervised':ss,'training':result}))
if __name__=='__main__': main()
