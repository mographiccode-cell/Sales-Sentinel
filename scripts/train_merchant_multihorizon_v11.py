from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import train_merchant_error_corrector_v7_5 as v75
import train_merchant_target_refinement_v8 as v8

ROOT=Path(__file__).resolve().parents[1]
VERSION="SALES-SENTINEL-V11-MULTI-HORIZON-PREQUENTIAL-VERIFIER"
PANEL=ROOT/'data/merchant_v7_1/merchant_feature_panel_v7_1.csv'
SECTOR=ROOT/'data/saudi_v1_5/saudi_sector_daily_panel_v1_5.csv.gz'
BASE_OOF=ROOT/'reports/merchant_meta_verifier_v9_2/oof_predictions.csv'
DIAG=ROOT/'reports/merchant_market_fusion_v6_1/oof_policy_diagnostics.csv'
OUT=ROOT/'reports/merchant_multihorizon_v11'; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'development_report.json'; SUMMARY=OUT/'development_summary.md'; OOF=OUT/'oof_predictions.csv'; AUX=OUT/'oof_auxiliary_scores.csv'
META={'date','future_ratio','future7_sales','baseline28_daily','target','future3_ratio','future14_ratio','target3','target14'}
SEED=42


def build_labels(d):
    s=pd.read_csv(SECTOR,parse_dates=['TrainingSafeDate'])
    daily=s.groupby('TrainingSafeDate')['sales'].sum().sort_index().to_frame('sales')
    daily=daily.reindex(pd.date_range(daily.index.min(),daily.index.max(),freq='D')).fillna(0.0)
    daily['baseline28']=daily.sales.rolling(28,min_periods=28).mean()
    for h in [3,7,14]:
        daily[f'future{h}']=sum(daily.sales.shift(-k) for k in range(1,h+1))
        daily[f'ratio{h}']=daily[f'future{h}']/(h*daily.baseline28.replace(0,np.nan))
    z=d[['date']].merge(daily[['ratio3','ratio7','ratio14']],left_on='date',right_index=True,how='left')
    if np.nanmax(np.abs(z.ratio7.to_numpy(float)-d.future_ratio.to_numpy(float)))>1e-8:
        raise RuntimeError('Reconstructed 7-day target does not match frozen target')
    d=d.copy(); d['future3_ratio']=z.ratio3; d['future14_ratio']=z.ratio14
    d['target3']=np.where(d.future3_ratio.notna(),(d.future3_ratio<.85).astype(float),np.nan)
    d['target14']=np.where(d.future14_ratio.notna(),(d.future14_ratio<.85).astype(float),np.nan)
    return d


def horizon_windows(d,h):
    periods=[('2023-07-08','2023-09-30'),('2023-10-08','2023-12-31'),('2024-01-08','2024-03-31'),('2024-04-08','2024-06-30'),('2024-07-08','2024-08-19')]
    out=[]
    for fid,(a,b) in enumerate(periods):
        a=pd.Timestamp(a); b=pd.Timestamp(b)
        tr=(d.date<=a-pd.Timedelta(days=h+1))
        va=d.date.between(a,b)
        out.append((fid,tr,va))
    return out


def make_model(kind,y):
    y=np.asarray(y,int); pos=max((y==1).sum(),1); neg=max((y==0).sum(),1)
    if kind=='catboost':
        return CatBoostClassifier(iterations=550,depth=4,learning_rate=.022,l2_leaf_reg=16,random_seed=SEED,verbose=False,allow_writing_files=False,loss_function='Logloss',auto_class_weights='Balanced',random_strength=1.0)
    return XGBClassifier(n_estimators=520,max_depth=2,learning_rate=.022,min_child_weight=8,subsample=.86,colsample_bytree=.72,reg_alpha=2.0,reg_lambda=15.0,gamma=.18,objective='binary:logistic',eval_metric='logloss',random_state=SEED,n_jobs=2,scale_pos_weight=float(neg/pos))


def auxiliary_oof(d,h,kind,topk):
    target=f'target{h}'; features=[c for c in d.columns if c not in META]
    parts=[]; stats=[]
    for fid,tr,va in horizon_windows(d,h):
        tr=tr & d[target].notna()
        Xtr0,Xva0=v75.prepare(d.loc[tr,features],d.loc[va,features]); ytr=d.loc[tr,target].astype(int)
        cols=v75.stable_top(Xtr0,ytr,topk); Xtr=Xtr0[cols]; Xva=Xva0[cols]
        m=make_model(kind,ytr); m.fit(Xtr,ytr)
        score=m.predict_proba(Xva)[:,1]
        yy=d.loc[va,target].to_numpy(float); valid=np.isfinite(yy)
        st={'fold_id':fid,'rows':int(va.sum()),'labeled':int(valid.sum()),'positives':int(np.nansum(yy)),'feature_count':len(cols)}
        if valid.sum()>5 and len(np.unique(yy[valid].astype(int)))==2:
            st['roc_auc']=float(roc_auc_score(yy[valid].astype(int),score[valid])); st['pr_auc']=float(average_precision_score(yy[valid].astype(int),score[valid]))
        else: st['roc_auc']=None; st['pr_auc']=None
        stats.append(st)
        parts.append(pd.DataFrame({'date':d.loc[va,'date'].to_numpy(),'fold_id':fid,f'risk{h}':score}))
    o=pd.concat(parts,ignore_index=True).sort_values(['fold_id','date']).reset_index(drop=True)
    aucs=[x['roc_auc'] for x in stats if x['roc_auc'] is not None]; prs=[x['pr_auc'] for x in stats if x['pr_auc'] is not None]
    return o,{'mean_fold_auc':float(np.mean(aucs)),'min_fold_auc':float(np.min(aucs)),'mean_fold_pr':float(np.mean(prs)),'folds':stats}


def choose_aux(d,h):
    rows=[]; oo=[]
    for kind,topk in product(['catboost','xgb'],[64,96,128]):
        o,m=auxiliary_oof(d,h,kind,topk); rows.append({'config':{'model':kind,'topk':topk},'metrics':m}); oo.append(o)
    def key(x): return (x['metrics']['mean_fold_auc'],x['metrics']['mean_fold_pr'],x['metrics']['min_fold_auc'])
    ix=max(range(len(rows)),key=lambda i:key(rows[i])); return rows[ix],oo[ix],rows


def fit_predict(A,y,B,C):
    sc=StandardScaler(); X=sc.fit_transform(A); Z=sc.transform(B)
    m=LogisticRegression(C=C,class_weight='balanced',max_iter=3000,solver='liblinear',random_state=SEED); m.fit(X,y)
    return m.predict_proba(X)[:,1],m.predict_proba(Z)[:,1]


def meta_features(base,r3,r14,diag,panel_oof):
    x=pd.DataFrame(index=np.arange(len(base)))
    x['risk3']=r3; x['risk14']=r14; x['risk_min']=np.minimum(r3,r14); x['risk_max']=np.maximum(r3,r14); x['risk_mean']=(r3+r14)/2; x['horizon_gap']=np.abs(r3-r14)
    if 'v9_risk' in base.columns: x['v9_risk']=base.v9_risk.to_numpy(float)
    if 'v76_score' in base.columns: x['v76_score']=base.v76_score.to_numpy(float)
    for c in ['merchant_logreg','merchant_extra','merchant_mean','merchant_disagreement','market_v3__risk_mean','market_v3__risk_max','market_v3__risk_p90','market_v3__risk_share_25','market_v3__precursor_mean']:
        if c in diag.columns: x[c]=pd.to_numeric(diag[c],errors='coerce').fillna(0).to_numpy()
    strong,quiet,market,_=v75.base_components(diag); x['branch_strong']=strong.astype(float); x['branch_quiet']=quiet.astype(float); x['branch_market']=market.astype(float)
    for c in [c for c in panel_oof.columns if c.startswith('v8__')]: x[c]=pd.to_numeric(panel_oof[c],errors='coerce').fillna(0).to_numpy()
    return x,strong,market


def causal_candidate(y,folds,base_pred,x,strong,market,cfg):
    final=base_pred.copy(); details=[]; core=[c for c in x.columns if not c.startswith('v8__')]; hard=[c for c in x.columns if c.startswith('v8__')]; cols=core if cfg['features']=='core' else core+hard
    for f in sorted(np.unique(folds)):
        cur=folds==f
        if f==0: details.append({'fold_id':int(f),'mode':'v9_2_bootstrap'}); continue
        hist=folds<f; ha=hist&base_pred
        if ha.sum()<20 or len(np.unique(y[ha]))<2: details.append({'fold_id':int(f),'mode':'insufficient_history'}); continue
        ptr,pcur=fit_predict(x.loc[ha,cols],y[ha],x.loc[cur,cols],cfg['C']); tp=ptr[y[ha]==1]
        thr=float(max(0,np.quantile(tp,cfg['tp_quantile'])-cfg['margin']))
        scope=(market[cur]&(~strong[cur])) if cfg['scope']=='market' else ((~strong[cur]) if cfg['scope']=='nonstrong' else np.ones(cur.sum(),bool))
        cur_base=base_pred[cur].copy(); veto=cur_base&scope&(pcur<thr)&(x.loc[cur,'risk3'].to_numpy()<cfg['guard3'])&(x.loc[cur,'risk14'].to_numpy()<cfg['guard14'])
        final[cur]=cur_base&(~veto); details.append({'fold_id':int(f),'threshold':thr,'vetoes':int(veto.sum()),'history_alerts':int(ha.sum()),'history_tp':int(y[ha].sum()),'mode':'verified'})
    return final,details


def main():
    d=pd.read_csv(PANEL,parse_dates=['date']).sort_values('date').reset_index(drop=True); d=v8.add_hard_negative_features(d); d=build_labels(d)
    base=pd.read_csv(BASE_OOF).sort_values(['fold_id','date']).reset_index(drop=True); diag=pd.read_csv(DIAG); y=diag.y.to_numpy(int); folds=diag.fold_id.to_numpy(int); bp=base.v9_2_pred.to_numpy(bool)
    if len(base)!=381 or not np.array_equal(base.y.to_numpy(int),y): raise RuntimeError('Base alignment mismatch')
    # Align panel rows to the five validation folds for hard-negative meta features.
    pp=[]
    for fid,_,va in v75.windows(d): q=d.loc[va].copy(); q['fold_id']=fid; pp.append(q)
    po=pd.concat(pp,ignore_index=True).sort_values(['fold_id','date']).reset_index(drop=True)
    a3,o3,all3=choose_aux(d,3); a14,o14,all14=choose_aux(d,14)
    if len(o3)!=381 or len(o14)!=381: raise RuntimeError('Aux OOF length mismatch')
    r3=o3.risk3.to_numpy(float); r14=o14.risk14.to_numpy(float); x,strong,market=meta_features(base,r3,r14,diag,po); bm=v75.metrics(y,bp,folds)
    configs=[]
    for features,C,tpq,margin,scope,g3,g14 in product(['core','hard'],[.05,.1,.5],[0,.1,.2],[.01,.03],['market','nonstrong','any'],[.4,.5,.6],[.4,.5,.6]): configs.append({'features':features,'C':C,'tp_quantile':tpq,'margin':margin,'scope':scope,'guard3':g3,'guard14':g14})
    rows=[]; preds=[]
    for i,cfg in enumerate(configs):
        p,details=causal_candidate(y,folds,bp,x,strong,market,cfg); m=v75.metrics(y,p,folds); adopt=bool(m['recall']>=bm['recall'] and m['green_npv']>=bm['green_npv'] and m['precision']>bm['precision'] and m['f1']>bm['f1'] and m['fp']<bm['fp'] and m['worst_fold_recall']>=bm['worst_fold_recall'] and m['alert_rate']<=bm['alert_rate']); rows.append({'config_id':i,'config':cfg,'metrics':m,'strictly_dominates_v9_2':adopt,'details':details}); preds.append(p)
    def key(r): m=r['metrics']; return (int(r['strictly_dominates_v9_2']),m['f1'],m['precision'],-m['fp'],m['recall'],m['green_npv'],-m['alert_rate'])
    sel=max(rows,key=key); pred=preds[sel['config_id']]; m=sel['metrics']
    pd.DataFrame({'date':base.date,'y':y,'fold_id':folds,'v9_2_pred':bp.astype(int),'risk3':r3,'risk14':r14,'v11_pred':pred.astype(int)}).to_csv(OOF,index=False)
    pd.DataFrame({'date':base.date,'fold_id':folds,'risk3':r3,'risk14':r14}).to_csv(AUX,index=False)
    report={'version':VERSION,'status':'DEVELOPMENT_BEST' if sel['strictly_dominates_v9_2'] else 'EXPERIMENTAL_V9_2_REMAINS_BEST','scientific_boundary':'3-day and 14-day labels are reconstructed from the same sector-daily sales source that exactly reproduces the frozen 7-day target. Horizon models use horizon-specific purges and the final verifier is fitted only on earlier folds. Configuration selection remains development evidence; external Saudi merchant validation is required.','target_reconstruction_exact':True,'aux3_selected':a3,'aux14_selected':a14,'aux3_candidates':all3,'aux14_candidates':all14,'meta_candidate_count':len(rows),'v9_2':bm,'selected':sel,'top_candidates':sorted(rows,key=key,reverse=True)[:10],'red_supported':False}
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Sales Sentinel V11 — Multi-Horizon Prequential Verifier','',f'- Status: **{report["status"]}**','- Frozen 7-day target reconstruction: **Exact**',f'- 3-day auxiliary: **{a3["config"]}**, mean fold AUC **{a3["metrics"]["mean_fold_auc"]:.2%}**',f'- 14-day auxiliary: **{a14["config"]}**, mean fold AUC **{a14["metrics"]["mean_fold_auc"]:.2%}**',f'- Meta candidates: **{len(rows)}**','',f'- Precision: V9.2 **{bm["precision"]:.2%}** -> V11 **{m["precision"]:.2%}**',f'- Recall: V9.2 **{bm["recall"]:.2%}** -> V11 **{m["recall"]:.2%}**',f'- F1: V9.2 **{bm["f1"]:.2%}** -> V11 **{m["f1"]:.2%}**',f'- NPV: V9.2 **{bm["green_npv"]:.2%}** -> V11 **{m["green_npv"]:.2%}**',f'- Alert rate: V9.2 **{bm["alert_rate"]:.2%}** -> V11 **{m["alert_rate"]:.2%}**',f'- TP/FP/FN/TN: **{m["tp"]}/{m["fp"]}/{m["fn"]}/{m["tn"]}**',f'- Worst-fold recall: **{m["worst_fold_recall"]:.2%}**',f'- Strictly dominates V9.2: **{sel["strictly_dominates_v9_2"]}**','','Scientific boundary: development evidence only; fresh external Saudi merchant validation remains pending.']
    SUMMARY.write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
