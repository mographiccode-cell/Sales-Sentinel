from __future__ import annotations

import json,re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import train_merchant_error_corrector_v7_5 as v75

ROOT=Path(__file__).resolve().parents[1]
VERSION='SALES-SENTINEL-V12-SECTOR-BREADTH-PREQUENTIAL-VERIFIER'
SECTOR=ROOT/'data/saudi_v1_5/saudi_sector_daily_panel_v1_5.csv.gz'
BASE=ROOT/'reports/merchant_multihorizon_v11/oof_predictions.csv'
V92=ROOT/'reports/merchant_meta_verifier_v9_2/oof_predictions.csv'
DIAG=ROOT/'reports/merchant_market_fusion_v6_1/oof_policy_diagnostics.csv'
OUT=ROOT/'reports/merchant_sector_breadth_v12'; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'development_report.json'; SUMMARY=OUT/'development_summary.md'; OOF=OUT/'oof_predictions.csv'; FEATURE_FILE=OUT/'sector_daily_features.csv'
SEED=42


def sector_features():
    s=pd.read_csv(SECTOR,parse_dates=['TrainingSafeDate']).sort_values(['SAMASector','TrainingSafeDate']).copy()
    sectors=sorted(s.SAMASector.astype(str).unique().tolist()); sid={v:i for i,v in enumerate(sectors)}; s['sid']=s.SAMASector.astype(str).map(sid)
    metrics=[c for c in ['sales','invoices','customers','products'] if c in s.columns]
    for c in metrics:
        g=s.groupby('sid')[c]
        ma7=g.transform(lambda x:x.rolling(7,min_periods=4).mean()); ma28=g.transform(lambda x:x.rolling(28,min_periods=14).mean())
        s[f'{c}_r7_28']=ma7/ma28.replace(0,np.nan)
        s[f'{c}_ch7']=g.pct_change(7,fill_method=None).replace([np.inf,-np.inf],np.nan)
    for c in [x for x in ['sama_sector_value','sama_sector_count','sama_factor'] if x in s.columns]:
        g=s.groupby('sid')[c]; s[f'{c}_ch7']=g.pct_change(7,fill_method=None).replace([np.inf,-np.inf],np.nan)
    if 'sales_ch7' in s and 'sama_sector_value_ch7' in s: s['sales_minus_sama_ch7']=s.sales_ch7-s.sama_sector_value_ch7
    feature_cols=[c for c in s.columns if c.endswith('_r7_28') or c.endswith('_ch7') or c=='sales_minus_sama_ch7']
    rows=[]
    for dt,g in s.groupby('TrainingSafeDate'):
        row={'date':dt}
        if 'sales' in g:
            w=np.clip(g.sales.to_numpy(float),0,None); total=w.sum(); sh=w/total if total>0 else np.repeat(1/len(w),len(w)); row['sector__sales_hhi']=float(np.sum(sh**2)); row['sector__max_sales_share']=float(sh.max())
        if 'sales_r7_28' in g:
            a=g.sales_r7_28.to_numpy(float); row['sector__decline_breadth_100']=float(np.nanmean(a<1.0)); row['sector__decline_breadth_90']=float(np.nanmean(a<.90)); row['sector__decline_breadth_80']=float(np.nanmean(a<.80)); row['sector__rebound_breadth_105']=float(np.nanmean(a>1.05))
        for c in feature_cols:
            a=pd.to_numeric(g[c],errors='coerce').to_numpy(float)
            if np.isfinite(a).any():
                row[f'sector__{c}__mean']=float(np.nanmean(a)); row[f'sector__{c}__std']=float(np.nanstd(a)); row[f'sector__{c}__min']=float(np.nanmin(a)); row[f'sector__{c}__max']=float(np.nanmax(a))
        # retain each sector sales/invoice/customer regime individually
        for _,r in g.iterrows():
            i=int(r.sid)
            for c in [x for x in ['sales_r7_28','sales_ch7','invoices_r7_28','customers_r7_28','products_r7_28','sama_sector_value_ch7','sales_minus_sama_ch7'] if x in g.columns]:
                val=pd.to_numeric(pd.Series([r[c]]),errors='coerce').iloc[0]; row[f'sector{i}__{c}']=float(val) if pd.notna(val) else np.nan
        rows.append(row)
    f=pd.DataFrame(rows).sort_values('date').reset_index(drop=True); f.to_csv(FEATURE_FILE,index=False); return f,sectors


def fit_predict(A,y,B,C):
    A=A.replace([np.inf,-np.inf],np.nan); B=B.replace([np.inf,-np.inf],np.nan); med=A.median().fillna(0); A=A.fillna(med); B=B.fillna(med)
    sc=StandardScaler(); X=sc.fit_transform(A); Z=sc.transform(B); m=LogisticRegression(C=C,class_weight='balanced',max_iter=3000,solver='liblinear',random_state=SEED); m.fit(X,y); return m.predict_proba(X)[:,1],m.predict_proba(Z)[:,1]


def candidate(y,folds,base,x,strong,market,consensus,cfg):
    final=base.copy(); details=[]; sector=[c for c in x.columns if c.startswith('sector')]; core=[c for c in x.columns if not c.startswith('sector')]; cols=sector if cfg['features']=='sector' else core+sector
    for f in sorted(np.unique(folds)):
        cur=folds==f
        if f==0: details.append({'fold_id':int(f),'mode':'v11_bootstrap'}); continue
        hist=folds<f; ha=hist&base
        if ha.sum()<20 or len(np.unique(y[ha]))<2: details.append({'fold_id':int(f),'mode':'insufficient_history'}); continue
        ptr,pcur=fit_predict(x.loc[ha,cols],y[ha],x.loc[cur,cols],cfg['C']); tp=ptr[y[ha]==1]; thr=float(max(0,np.quantile(tp,cfg['tp_quantile'])-cfg['margin']))
        scope=(market[cur]&(~strong[cur])) if cfg['scope']=='market' else ((~strong[cur]) if cfg['scope']=='nonstrong' else np.ones(cur.sum(),bool)); curbase=base[cur].copy(); guard=np.ones(cur.sum(),bool) if cfg['guard']>1 else consensus[cur]<cfg['guard']; veto=curbase&scope&(pcur<thr)&guard; final[cur]=curbase&(~veto); details.append({'fold_id':int(f),'threshold':thr,'vetoes':int(veto.sum()),'history_alerts':int(ha.sum()),'history_tp':int(y[ha].sum()),'mode':'verified'})
    return final,details


def main():
    b=pd.read_csv(BASE).sort_values(['fold_id','date']).reset_index(drop=True); v=pd.read_csv(V92).sort_values(['fold_id','date']).reset_index(drop=True); d=pd.read_csv(DIAG); y=b.y.to_numpy(int); folds=b.fold_id.to_numpy(int); base=b.v11_pred.to_numpy(bool)
    if len(b)!=381 or not np.array_equal(y,v.y.to_numpy(int)) or not np.array_equal(y,d.y.to_numpy(int)): raise RuntimeError('OOF alignment mismatch')
    sf,sectors=sector_features(); dates=pd.DataFrame({'date':pd.to_datetime(b.date)}); x=dates.merge(sf,on='date',how='left').drop(columns=['date'])
    x['risk3']=b.risk3.to_numpy(float); x['risk14']=b.risk14.to_numpy(float); x['v9_risk']=v.v9_risk.to_numpy(float); x['v76_score']=v.v76_score.to_numpy(float); x['risk_consensus']=(x.risk3+x.risk14+x.v9_risk+x.v76_score)/4
    for c in ['merchant_mean','merchant_disagreement','market_v3__risk_mean','market_v3__risk_p90','market_v3__risk_share_25','market_v3__precursor_mean']:
        if c in d.columns: x[c]=pd.to_numeric(d[c],errors='coerce').fillna(0).to_numpy()
    strong,_,market,_=v75.base_components(d); consensus=x.risk_consensus.to_numpy(float); bm=v75.metrics(y,base,folds)
    configs=[]
    for features,C,tpq,margin,scope,guard in product(['sector','combined'],[.05,.1,.5],[0,.1,.2],[.01,.03],['market','nonstrong','any'],[.5,.65,1.01]): configs.append({'features':features,'C':C,'tp_quantile':tpq,'margin':margin,'scope':scope,'guard':guard})
    rows=[]; preds=[]
    for i,cfg in enumerate(configs):
        p,details=candidate(y,folds,base,x,strong,market,consensus,cfg); m=v75.metrics(y,p,folds); adopt=bool(m['recall']>=bm['recall'] and m['green_npv']>=bm['green_npv'] and m['precision']>bm['precision'] and m['f1']>bm['f1'] and m['fp']<bm['fp'] and m['worst_fold_recall']>=bm['worst_fold_recall'] and m['alert_rate']<=bm['alert_rate']); rows.append({'config_id':i,'config':cfg,'metrics':m,'strictly_dominates_v11':adopt,'details':details}); preds.append(p)
    def key(r): m=r['metrics']; return (int(r['strictly_dominates_v11']),m['f1'],m['precision'],-m['fp'],m['recall'],m['green_npv'],-m['alert_rate'])
    sel=max(rows,key=key); pred=preds[sel['config_id']]; m=sel['metrics']; pd.DataFrame({'date':b.date,'y':y,'fold_id':folds,'v11_pred':base.astype(int),'risk_consensus':consensus,'v12_pred':pred.astype(int)}).to_csv(OOF,index=False)
    report={'version':VERSION,'status':'DEVELOPMENT_BEST' if sel['strictly_dominates_v11'] else 'EXPERIMENTAL_V11_REMAINS_BEST','scientific_boundary':'V12 sector-shape features are computed from same-day and trailing history only on the 8-sector Saudi-localized/SAMA-calibrated panel. Alert-verifier training is earlier-fold-only. Configuration selection remains development evidence; external real Saudi merchant validation is required.','sector_count':len(sectors),'sector_feature_count':int(sum(c.startswith('sector') for c in x.columns)),'candidate_count':len(rows),'v11':bm,'selected':sel,'top_candidates':sorted(rows,key=key,reverse=True)[:10],'red_supported':False}; REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Sales Sentinel V12 — Sector Breadth Verifier','',f'- Status: **{report["status"]}**',f'- Sectors: **{len(sectors)}**',f'- Sector features: **{report["sector_feature_count"]}**',f'- Candidates: **{len(rows)}**','',f'- Precision: V11 **{bm["precision"]:.2%}** -> V12 **{m["precision"]:.2%}**',f'- Recall: V11 **{bm["recall"]:.2%}** -> V12 **{m["recall"]:.2%}**',f'- F1: V11 **{bm["f1"]:.2%}** -> V12 **{m["f1"]:.2%}**',f'- NPV: V11 **{bm["green_npv"]:.2%}** -> V12 **{m["green_npv"]:.2%}**',f'- Alert rate: V11 **{bm["alert_rate"]:.2%}** -> V12 **{m["alert_rate"]:.2%}**',f'- TP/FP/FN/TN: **{m["tp"]}/{m["fp"]}/{m["fn"]}/{m["tn"]}**',f'- Worst-fold recall: **{m["worst_fold_recall"]:.2%}**',f'- Strictly dominates V11: **{sel["strictly_dominates_v11"]}**','','Scientific boundary: development evidence only; external Saudi merchant validation remains pending.']; SUMMARY.write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
