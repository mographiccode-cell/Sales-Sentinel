from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import train_merchant_total_hybrid_v4_3 as v3
import train_merchant_total_triage_v4_4 as v44

ROOT=Path(__file__).resolve().parents[1]
VERSION='SALES-SENTINEL-MERCHANT-TOTAL-TRIAGE-4.5-OPERATIONAL-GUARDRAIL'
SRC=ROOT/'data'/'merchant_v4_3'/'merchant_total_feature_panel_v4_3.csv'
OUT=ROOT/'reports'/'merchant_total_triage_v4_5';MOD=ROOT/'models'/'merchant_total_triage_v4_5'
OUT.mkdir(parents=True,exist_ok=True);MOD.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'development_report.json';SUMMARY=OUT/'development_summary.md';MODEL=MOD/'merchant_total_triage_v4_5.joblib'

MERCHANT_RATIO_COLS=[
 'sama_calibrated_net_sales_sar_ratio_7','invoice_count_ratio_7','unique_observed_customers_ratio_7',
 'returning_observed_customers_ratio_7','unique_products_ratio_7','units_ratio_7','transaction_rows_ratio_7'
]
MERCHANT_CHANGE_COLS=[
 'sama_calibrated_net_sales_sar_change_7','invoice_count_change_7','unique_observed_customers_change_7',
 'returning_observed_customers_change_7','unique_products_change_7','units_change_7','transaction_rows_change_7'
]


def groups(columns):
    cat_ratio=[c for c in columns if c.startswith('cat__') and any(x in c for x in ['net_sales_ratio_mean_7','invoice_count_ratio_mean_7','observed_customer_count_ratio_mean_7','unique_products_ratio_mean_7'])]
    cat_change=[c for c in columns if c.startswith('cat__') and any(x in c for x in ['net_sales_change_7','invoice_count_change_7','observed_customer_count_change_7','unique_products_change_7'])]
    rich_ratio=[c for c in columns if c.startswith('rich__') and any(x in c for x in ['observed_customer_count_ratio_4','unique_products_ratio_4','avg_skus_per_invoice_ratio_4','avg_lines_per_invoice_ratio_4'])]
    rich_dropout=[c for c in columns if c.startswith('rich__') and any(x in c for x in ['customer_dropout_rate','sku_dropout_rate'])]
    rich_conc=[c for c in columns if c.startswith('rich__') and any(x in c for x in ['customer_sales_hhi','customer_top5_share','sku_sales_hhi','sku_top5_share'])]
    return cat_ratio,cat_change,rich_ratio,rich_dropout,rich_conc


def build_reference(Xtrain,columns):
    _,_,_,drop,conc=groups(columns)
    high_cols=drop+conc
    return {c:float(Xtrain[c].quantile(.80)) for c in high_cols if c in Xtrain}


def operational_score(X,reference):
    cols=list(X.columns);cat_ratio,cat_change,rich_ratio,rich_dropout,rich_conc=groups(cols)
    mr=[c for c in MERCHANT_RATIO_COLS if c in X];mc=[c for c in MERCHANT_CHANGE_COLS if c in X]
    merchant=np.zeros(len(X),float);den=0
    for c in mr: merchant+=(X[c].to_numpy(float)<.95);den+=1
    for c in mc: merchant+=(X[c].to_numpy(float)<-.05);den+=1
    merchant/=max(den,1)
    adverse=[]
    for c in cat_ratio: adverse.append(X[c].to_numpy(float)<.92)
    for c in cat_change: adverse.append(X[c].to_numpy(float)<-.08)
    category=np.column_stack(adverse).mean(axis=1) if adverse else np.zeros(len(X))
    radv=[]
    for c in rich_ratio: radv.append(X[c].to_numpy(float)<.90)
    for c in rich_dropout+rich_conc:
        if c in reference: radv.append(X[c].to_numpy(float)>float(reference[c]))
    rich=np.column_stack(radv).mean(axis=1) if radv else np.zeros(len(X))
    score=.45*merchant+.35*category+.20*rich
    return score,{'merchant_evidence':merchant,'category_breadth':category,'rich_evidence':rich}


def metric(y,pred):
    y=np.asarray(y,int);p=np.asarray(pred,bool)
    tn=int(((y==0)&(~p)).sum());fn=int(((y==1)&(~p)).sum())
    return {'precision':float(precision_score(y,p,zero_division=0)),'recall':float(recall_score(y,p,zero_division=0)),'f1':float(f1_score(y,p,zero_division=0)),'alert_rate':float(p.mean()),'green_npv':tn/max(tn+fn,1),'tp':int(((y==1)&p).sum()),'fp':int(((y==0)&p).sum()),'fn':fn,'tn':tn}


def main():
    d=pd.read_csv(SRC,parse_dates=['date']).sort_values('date').reset_index(drop=True);meta=d[['date','future_ratio']].copy();X=d.drop(columns=['date','future_ratio','target'])
    fs=v3.folds(meta.assign(target=(meta.future_ratio<.8).astype(int)))
    if len(fs)<5:raise RuntimeError('Need five rolling folds')
    oof_early=np.full(len(meta),np.nan);oof_severe=np.full(len(meta),np.nan);oof_op=np.full(len(meta),np.nan);foldid=np.full(len(meta),-1,int);valmask=np.zeros(len(meta),bool);foldmeta=[]
    for fid,(st,en,tr,va) in enumerate(fs):
        y=meta.loc[tr,'future_ratio'].clip(0,2.5)
        et=clone(v44.factories()['extra_trees_reg']).fit(X.loc[tr],y);q25=clone(v44.factories()['hist_gb_q25']).fit(X.loc[tr],y)
        pe=.55*et.predict(X.loc[va])+.45*q25.predict(X.loc[va]);ps=q25.predict(X.loc[va])
        ref=build_reference(X.loc[tr],X.columns);op,_=operational_score(X.loc[va],ref)
        ii=np.where(va.to_numpy())[0];oof_early[ii]=-pe;oof_severe[ii]=-ps;oof_op[ii]=op;foldid[ii]=fid;valmask[ii]=True
        foldmeta.append({'fold_id':fid,'start':str(st.date()),'end':str(en.date()),'train_rows':int(tr.sum()),'validation_rows':int(va.sum())})
    idx=np.where(valmask)[0];actual=meta.future_ratio.to_numpy(float)[idx];y15=(actual<.85).astype(int);y20=(actual<.80).astype(int);early=oof_early[idx];severe=oof_severe[idx];op=oof_op[idx]
    # Reproduce v4.4 thresholds from OOF; v4.5 is allowed to use historical development only.
    amber_t,base15,_=v44.select_amber(y15,early);base_alert=early>=amber_t
    red_t,red20,nred=v44.select_red(y20,severe,base_alert);red=np.zeros(len(idx),bool) if not np.isfinite(red_t) else severe>=red_t
    candidates=np.unique(np.r_[np.linspace(0.08,.80,145),np.quantile(op,np.linspace(.20,.995,120))])
    rows=[]
    for t in candidates:
        structural=op>=float(t);combined=base_alert|structural|red;m=metric(y15,combined);inc=structural&(~base_alert)&(~red);incneg=float((inc&(y15==0)).sum()/max(int(((y15==0)&(~base_alert)&(~red)).sum()),1))
        if m['recall']>=.78 and m['alert_rate']<=.50 and m['green_npv']>=.94 and m['precision']>=.27 and incneg<=.15:
            rows.append((float(t),m,incneg,int(inc.sum()),int((inc&(y15==1)).sum())))
    if rows:
        rows.sort(key=lambda z:(z[1]['recall'],z[1]['green_npv'],z[1]['precision'],-z[1]['alert_rate'],-z[2]),reverse=True);op_t,combined15,incneg,incalerts,inctp=rows[0];supported=True
    else:
        # Fail closed: keep guardrail disabled rather than weakening the contract.
        op_t=float('inf');combined15=metric(y15,base_alert|red);incneg=0.;incalerts=inctp=0;supported=False
    combined=(base_alert|(op>=op_t)|red);combined20=metric(y20,combined)
    per=[]
    for fid in sorted(set(foldid[idx])):
        m=foldid[idx]==fid;per.append({'fold_id':int(fid),'positives15':int(y15[m].sum()),'baseline':metric(y15[m],base_alert[m]),'combined':metric(y15[m],combined[m]),'operational_alert_rate':float((op[m]>=op_t).mean()) if supported else 0.})
    # Freeze final models and full-development operational reference.
    yf=meta.future_ratio.clip(0,2.5);final={'extra_trees_reg':clone(v44.factories()['extra_trees_reg']).fit(X,yf),'hist_gb_q25':clone(v44.factories()['hist_gb_q25']).fit(X,yf)};ref=build_reference(X,X.columns)
    artifact={'version':VERSION,'status':'DEVELOPMENT_FROZEN_PENDING_NEW_STRESS','feature_columns':list(X.columns),'models':final,'early_model_formula':{'extra_trees_reg':.55,'hist_gb_q25':.45},'amber_threshold_score':float(amber_t),'red_threshold_score':None if not np.isfinite(red_t) else float(red_t),'operational_guardrail_supported':supported,'operational_threshold':None if not supported else float(op_t),'operational_reference':ref,'operational_rule_constants':{'merchant_ratio_lt':.95,'merchant_change_lt':-.05,'category_ratio_lt':.92,'category_change_lt':-.08,'rich_ratio_lt':.90,'rich_high_quantile':.80,'weights':{'merchant':.45,'category':.35,'rich':.20}},'states':{'GREEN':'no model or operational warning','AMBER':'model early downside or transparent operational deterioration','RED':'severe high-precision state only if historical support exists'}}
    joblib.dump(artifact,MODEL)
    contract={'combined15_recall_min':.78,'combined15_green_npv_min':.94,'combined15_alert_rate_max':.50,'combined15_precision_min':.27,'incremental_negative_rate_max':.15}
    gates={'rolling_origin_past_only':True,'operational_channel_current_past_only':True,'base_dense_model_frozen_design':True,'combined_recall':combined15['recall']>=contract['combined15_recall_min'],'combined_green_npv':combined15['green_npv']>=contract['combined15_green_npv_min'],'combined_alert_rate':combined15['alert_rate']<=contract['combined15_alert_rate_max'],'combined_precision':combined15['precision']>=contract['combined15_precision_min'],'incremental_negative_rate':incneg<=contract['incremental_negative_rate_max'],'guardrail_supported':supported}
    rep={'version':VERSION,'status':artifact['status'],'scientific_boundary':'v4.4 sensitivity stress is development feedback for this new architecture and is NOT reused as independent evidence. v4.5 guardrail threshold is selected from historical rolling OOF labels only; a different stress suite is required next.','rows':len(meta),'oof_rows':len(idx),'feature_count':X.shape[1],'folds':foldmeta,'base_v4_4':{'amber_threshold':float(amber_t),'metrics15':base15},'operational_guardrail':{'supported':supported,'threshold':None if not supported else float(op_t),'feasible_candidates':len(rows),'incremental_alerts':incalerts,'incremental_true15':inctp,'incremental_negative_rate':incneg},'combined15':combined15,'combined20':combined20,'red20':red20,'red_supported':bool(np.isfinite(red_t)),'per_fold':per,'contract':contract,'gates':gates,'all_development_gates_passed':bool(all(gates.values())),'next_required_evidence':'New frozen sensitivity stress patterns that differ from v4.4 stress, followed by real merchant external validation.'}
    REPORT.write_text(json.dumps(rep,indent=2),encoding='utf-8');SUMMARY.write_text('# Sales Sentinel v4.5 — Operational Guardrail\n\n'+f"- Guardrail supported **{supported}**\n- Base early recall **{base15['recall']:.2%}**\n- Combined early recall **{combined15['recall']:.2%}**\n- Combined precision **{combined15['precision']:.2%}**\n- GREEN NPV **{combined15['green_npv']:.2%}**\n- Alert rate **{combined15['alert_rate']:.2%}**\n- Incremental negative rate **{incneg:.2%}**\n- Development gates **{all(gates.values())}**\n",encoding='utf-8');print(json.dumps(rep,indent=2))

if __name__=='__main__':main()
