from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_merchant_total_hybrid_v4_3 as v3

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data'/'merchant_v4_3'/'merchant_total_feature_panel_v4_3.csv'
OUT=ROOT/'reports'/'merchant_total_hybrid_v4_3'/'dense_diagnostic.json'
SEED=42


def models():
    return {
        'ridge':make_pipeline(StandardScaler(),Ridge(alpha=20.0)),
        'huber':make_pipeline(StandardScaler(),HuberRegressor(epsilon=1.5,alpha=.01,max_iter=2000)),
        'extra_trees_reg':ExtraTreesRegressor(n_estimators=900,max_depth=7,min_samples_leaf=7,max_features=.50,random_state=SEED,n_jobs=-1),
        'hist_gb_mean':HistGradientBoostingRegressor(max_iter=320,learning_rate=.025,max_leaf_nodes=12,min_samples_leaf=24,l2_regularization=10.,random_state=SEED),
        'hist_gb_q25':HistGradientBoostingRegressor(loss='quantile',quantile=.25,max_iter=320,learning_rate=.025,max_leaf_nodes=12,min_samples_leaf=24,l2_regularization=10.,random_state=SEED+1),
    }


def main():
    d=pd.read_csv(SRC,parse_dates=['date']).sort_values('date').reset_index(drop=True)
    meta=d[['date','future_ratio']].copy(); X=d.drop(columns=['date','future_ratio','target'])
    fs=v3.folds(meta.assign(target=(meta.future_ratio<.8).astype(int)))
    if len(fs)<5:raise RuntimeError(f'Expected 5 folds, got {len(fs)}')
    preds={n:np.full(len(meta),np.nan) for n in models()}; valmask=np.zeros(len(meta),bool); foldmeta=[]
    for fid,(st,en,tr,va) in enumerate(fs):
        y=meta.loc[tr,'future_ratio'].clip(0,2.5); valmask|=va.to_numpy(); foldmeta.append({'fold_id':fid,'start':str(st.date()),'end':str(en.date()),'train_rows':int(tr.sum()),'validation_rows':int(va.sum())})
        for n,m in models().items():
            z=clone(m).fit(X.loc[tr],y); preds[n][va.to_numpy()]=z.predict(X.loc[va])
    idx=np.where(valmask)[0]; actual=meta.future_ratio.to_numpy(float)[idx]
    base={}
    for n,p in preds.items():
        pp=p[idx]; base[n]={'mae_ratio':float(mean_absolute_error(actual,pp)),'spearman':float(spearmanr(actual,pp).statistic)}
    # Blends are defined before inspecting threshold-specific labels.
    blend_defs={
        'mean_tree_gb':.5*preds['extra_trees_reg']+.5*preds['hist_gb_mean'],
        'mean_q25_mix':.55*preds['hist_gb_mean']+.45*preds['hist_gb_q25'],
        'tree_q25_mix':.55*preds['extra_trees_reg']+.45*preds['hist_gb_q25'],
    }
    for n,p in blend_defs.items():
        pp=p[idx];base[n]={'mae_ratio':float(mean_absolute_error(actual,pp)),'spearman':float(spearmanr(actual,pp).statistic)}
    allpred={**preds,**blend_defs}; thresholds={}
    for decline in (.10,.15,.20):
        cutoff=1-decline;y=(actual<cutoff).astype(int); item={'positive_rate':float(y.mean()),'positives':int(y.sum()),'rows':len(y),'models':{}}
        for n,p in allpred.items():
            pp=p[idx]; item['models'][n]={'roc_auc':float(roc_auc_score(y,-pp)) if len(np.unique(y))==2 else None}
        best=max(item['models'],key=lambda n:item['models'][n]['roc_auc'] if item['models'][n]['roc_auc'] is not None else -1)
        item['best_model']=best;item['best_auc']=item['models'][best]['roc_auc'];thresholds[f'decline_{int(decline*100)}pct']=item
    rep={'version':'MERCHANT-TOTAL-DENSE-DIAGNOSTIC-1','scientific_boundary':'Post-v4.3 development diagnostic only; not independent validation. Uses rolling-origin past-only fits and dense future-ratio target.','rows':len(meta),'feature_count':X.shape[1],'folds':foldmeta,'regression':base,'decline_thresholds':thresholds}
    OUT.write_text(json.dumps(rep,indent=2),encoding='utf-8');print(json.dumps(rep,indent=2))

if __name__=='__main__':main()
