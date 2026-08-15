from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models'/'merchant_total_triage_v4_4'/'merchant_total_triage_v4_4.joblib'
PANEL=ROOT/'data'/'merchant_v4_3'/'merchant_total_feature_panel_v4_3.csv'
OUT=ROOT/'reports'/'merchant_total_triage_v4_4'/'independent_sensitivity_stress.json'
VERSION='MERCHANT-TOTAL-TRIAGE-4.4-INDEPENDENT-SENSITIVITY-1'


def predict(a,X):
    et=a['models']['extra_trees_reg'].predict(X)
    q25=a['models']['hist_gb_q25'].predict(X)
    early=.55*et+.45*q25
    severe=q25
    early_score=-early; severe_score=-severe
    amber=early_score>=float(a['amber_threshold_score'])
    red_t=a.get('red_threshold_score')
    red=np.zeros(len(X),dtype=bool) if red_t is None else severe_score>=float(red_t)
    state=np.where(red,'RED',np.where(amber,'AMBER','GREEN'))
    return pd.DataFrame({'early_pred_ratio':early,'severe_pred_ratio':severe,'early_score':early_score,'severe_score':severe_score,'state':state})


def cols_like(cols,needles):
    return [c for c in cols if any(n in c.lower() for n in needles)]


def main():
    a=joblib.load(MODEL)
    if a.get('version')!='SALES-SENTINEL-MERCHANT-TOTAL-TRIAGE-4.4':
        raise RuntimeError(f'Unexpected artifact {a.get("version")}')
    d=pd.read_csv(PANEL,parse_dates=['date']).sort_values('date').reset_index(drop=True)
    X=d[a['feature_columns']].copy()
    base=predict(a,X)
    green_idx=np.where(base.state.eq('GREEN').to_numpy())[0]
    if len(green_idx)<12: raise RuntimeError(f'Need >=12 GREEN controls, got {len(green_idx)}')
    # Deterministic spread across the historical feature distribution. No label is read by this stress test.
    picks=green_idx[np.linspace(0,len(green_idx)-1,12).astype(int)]
    controls=X.iloc[picks].reset_index(drop=True)
    ctrl_pred=predict(a,controls)

    low=X.quantile(.05,numeric_only=True); high=X.quantile(.95,numeric_only=True)
    patterns={
      'broad_demand_slowdown':{
        'low':['sales_sar_ratio','sales_sar_change','invoice_count_ratio','invoice_count_change','customers_ratio','customers_change','units_ratio','units_change','net_sales_ratio','net_sales_change'],
        'high':['return_rate','cancellation_line_rate']},
      'customer_attrition':{
        'low':['unique_observed_customers','new_customer_share','returning_customer_share','sales_per_customer','observed_customer_count_ratio','observed_customer_count_change','customer_new_rate'],
        'high':['customer_dropout_rate','customer_sales_hhi','customer_top5_share']},
      'product_availability_squeeze':{
        'low':['unique_products_ratio','unique_products_change','products_per_invoice','sku_new_rate'],
        'high':['sku_dropout_rate','sku_sales_hhi','sku_top5_share']},
      'external_market_contraction':{
        'low':['sama_market_index','sama_predicted_value','sama_predicted_count','predicted_value','predicted_count'],
        'high':[]},
    }
    results={}; all_rows=[]
    for name,spec in patterns.items():
        Z=controls.copy()
        low_cols=cols_like(Z.columns,spec['low']); high_cols=cols_like(Z.columns,spec['high'])
        # Use fixed 5th/95th historical feature quantiles; no outcome or threshold feedback changes these values.
        for c in low_cols:
            if c in low.index and np.isfinite(low[c]): Z[c]=float(low[c])
        for c in high_cols:
            if c in high.index and np.isfinite(high[c]): Z[c]=float(high[c])
        p=predict(a,Z)
        delta_early=p.early_pred_ratio.to_numpy()-ctrl_pred.early_pred_ratio.to_numpy()
        delta_severe=p.severe_pred_ratio.to_numpy()-ctrl_pred.severe_pred_ratio.to_numpy()
        item={'n':len(Z),'features_forced_low':len(low_cols),'features_forced_high':len(high_cols),'amber_or_red':int((p.state!='GREEN').sum()),'alert_rate':float((p.state!='GREEN').mean()),'red':int((p.state=='RED').sum()),'median_early_ratio_change':float(np.median(delta_early)),'median_severe_ratio_change':float(np.median(delta_severe)),'mean_early_ratio_change':float(np.mean(delta_early))}
        results[name]=item
        tmp=p.copy();tmp['pattern']=name;tmp['control_early_pred_ratio']=ctrl_pred.early_pred_ratio.to_numpy();tmp['delta_early_ratio']=delta_early;all_rows.append(tmp)
    overall=pd.concat(all_rows,ignore_index=True)
    active_patterns=sum(v['alert_rate']>=.50 for v in results.values())
    worsening_patterns=sum(v['median_early_ratio_change']<=-.02 for v in results.values())
    acceptance={'control_repeat_identical':bool(np.allclose(ctrl_pred.early_pred_ratio,predict(a,controls).early_pred_ratio) and np.array_equal(ctrl_pred.state,predict(a,controls).state)),'overall_alert_rate_ge_60pct':float((overall.state!='GREEN').mean())>=.60,'at_least_3_of_4_patterns_alert_ge_50pct':active_patterns>=3,'at_least_3_of_4_patterns_worsen_ratio_ge_2pct':worsening_patterns>=3,'red_remains_disabled_when_unsupported':bool((overall.state=='RED').sum()==0 and a.get('red_threshold_score') is None)}
    rep={'version':VERSION,'frozen_model':a['version'],'test_type':'feature-space sensitivity stress on new deterioration patterns; not a real-world accuracy/recall estimate','control_rows':len(controls),'scenario_rows':len(overall),'patterns':results,'overall':{'alert_rate':float((overall.state!='GREEN').mean()),'amber':int((overall.state=='AMBER').sum()),'red':int((overall.state=='RED').sum()),'green':int((overall.state=='GREEN').sum()),'median_early_ratio_change':float(np.median(overall.delta_early_ratio))},'acceptance':acceptance,'all_sensitivity_gates_passed':bool(all(acceptance.values())),'scientific_boundary':'No model fitting, threshold selection, or outcome labels are used here. Feature perturbations test directional sensitivity only. External real merchant validation is still required for accuracy claims.'}
    OUT.write_text(json.dumps(rep,indent=2),encoding='utf-8');print(json.dumps(rep,indent=2))

if __name__=='__main__':main()
