from __future__ import annotations
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED=42
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'sama_pos'/'sama_pos_2020_2025_normalized.csv'
OUT=ROOT/'models'/'sama_panel_v2_0'; REP=ROOT/'reports'/'sama_panel_v2_0'; OUT.mkdir(parents=True,exist_ok=True); REP.mkdir(parents=True,exist_ok=True)
TARGET_CANDIDATES=[.05,.10,.15,.20,.25]

def metric(y,p,t):
    z=(p>=t).astype(int); return {'Accuracy':float(accuracy_score(y,z)),'BalancedAccuracy':float(balanced_accuracy_score(y,z)),'Precision':float(precision_score(y,z,zero_division=0)),'Recall':float(recall_score(y,z,zero_division=0)),'F1':float(f1_score(y,z,zero_division=0)),'ROC_AUC':float(roc_auc_score(y,p)),'ConfusionMatrix':confusion_matrix(y,z,labels=[0,1]).tolist()}
def score(m): return .30*m['BalancedAccuracy']+.25*m['F1']+.20*m['Accuracy']+.15*m['ROC_AUC']+.10*m['Recall']
def models(): return {'LogisticRegression':Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.5,max_iter=5000,class_weight='balanced',random_state=SEED))]),'RandomForest':RandomForestClassifier(n_estimators=900,max_depth=12,min_samples_leaf=3,max_features=.65,class_weight='balanced_subsample',random_state=SEED,n_jobs=-1),'ExtraTrees':ExtraTreesClassifier(n_estimators=1200,max_depth=14,min_samples_leaf=2,max_features=.70,class_weight='balanced',random_state=SEED,n_jobs=-1),'HistGradientBoosting':HistGradientBoostingClassifier(learning_rate=.035,max_iter=450,max_leaf_nodes=24,min_samples_leaf=20,l2_regularization=2,random_state=SEED)}

def prepare():
    raw=pd.read_csv(DATA,parse_dates=['week_start','week_end'])
    ind=raw['indicator'].astype(str).str.lower()
    vm=ind.str.contains('value')&ind.str.contains('transaction')&~ind.str.contains('change')
    nm=ind.str.contains('number')&ind.str.contains('transaction')&~ind.str.contains('change')
    def sub(mask,name): return raw.loc[mask,['week_start','city','sector','value']].rename(columns={'value':name}).drop_duplicates(['week_start','city','sector'],keep='last')
    d=sub(vm,'pos_value_thousand_sar').merge(sub(nm,'transaction_count'),on=['week_start','city','sector'],how='inner')
    d=d[~d.city.astype(str).str.strip().str.lower().eq('total') & ~d.sector.astype(str).str.strip().str.lower().eq('total')].copy()
    d['pos_value_thousand_sar']=pd.to_numeric(d['pos_value_thousand_sar'],errors='coerce'); d['transaction_count']=pd.to_numeric(d['transaction_count'],errors='coerce')
    d=d.dropna(); d=d[(d.pos_value_thousand_sar>0)&(d.transaction_count>0)].sort_values(['city','sector','week_start']).reset_index(drop=True)
    counts=d.groupby(['city','sector']).size(); keep=set(counts[counts>=80].index); d=d[d.set_index(['city','sector']).index.isin(keep)].sort_values(['city','sector','week_start']).reset_index(drop=True)
    g=d.groupby(['city','sector'],sort=False)
    d['week_gap_days']=g.week_start.diff().dt.days
    for col,prefix in [('pos_value_thousand_sar','value'),('transaction_count','count')]:
        s=d[col].astype(float); d[f'log_{prefix}_t0']=np.log1p(s)
        for lag in (1,2,3,4,8,13,26,52): d[f'log_{prefix}_lag_{lag}']=np.log1p(g[col].shift(lag))
        for w in (4,8,13,26,52):
            d[f'{prefix}_mean_{w}']=g[col].transform(lambda x:x.rolling(w).mean()); d[f'{prefix}_std_{w}']=g[col].transform(lambda x:x.rolling(w).std())
        d[f'{prefix}_ratio_mean4']=s/d[f'{prefix}_mean_4'].replace(0,np.nan); d[f'{prefix}_ratio_mean13']=s/d[f'{prefix}_mean_13'].replace(0,np.nan); d[f'{prefix}_ratio_mean52']=s/d[f'{prefix}_mean_52'].replace(0,np.nan)
        d[f'{prefix}_pct_change1']=g[col].pct_change(1); d[f'{prefix}_pct_change4']=g[col].pct_change(4); d[f'{prefix}_pct_change13']=g[col].pct_change(13)
    nat_v=sub(vm,'national_value'); nat_v=nat_v[nat_v.city.astype(str).str.strip().str.lower().eq('total') & nat_v.sector.astype(str).str.strip().str.lower().eq('total')][['week_start','national_value']]
    nat_n=sub(nm,'national_count'); nat_n=nat_n[nat_n.city.astype(str).str.strip().str.lower().eq('total') & nat_n.sector.astype(str).str.strip().str.lower().eq('total')][['week_start','national_count']]
    nat=nat_v.merge(nat_n,on='week_start',how='inner').sort_values('week_start').reset_index(drop=True)
    for col in ('national_value','national_count'):
        nat[f'{col}_pct1']=nat[col].pct_change(1); nat[f'{col}_pct4']=nat[col].pct_change(4); nat[f'{col}_mean13']=nat[col].rolling(13).mean(); nat[f'{col}_ratio13']=nat[col]/nat[f'{col}_mean13'].replace(0,np.nan)
    d=d.merge(nat,on='week_start',how='left').sort_values(['city','sector','week_start']).reset_index(drop=True)
    week=d.week_start.dt.isocalendar().week.astype(float); d['week_sin']=np.sin(2*np.pi*week/52.18); d['week_cos']=np.cos(2*np.pi*week/52.18); d['year_trend']=d.week_start.dt.year+(week/52.18)
    # IMPORTANT: merge rebuilt the index; recreate groups before target construction.
    g2=d.groupby(['city','sector'],sort=False)
    d['baseline4']=g2.pos_value_thousand_sar.transform(lambda x:x.rolling(4).mean()); d['next_value']=g2.pos_value_thousand_sar.shift(-1); d['next_ratio']=d.next_value/d.baseline4.replace(0,np.nan)
    d['next_week_start']=g2.week_start.shift(-1); d['next_gap_days']=(d.next_week_start-d.week_start).dt.days
    cat=pd.get_dummies(d[['city','sector']],prefix=['city','sector'],dtype=float); d=pd.concat([d,cat],axis=1)
    exclude={'week_start','next_week_start','next_gap_days','city','sector','pos_value_thousand_sar','transaction_count','next_value','next_ratio'}
    features=[c for c in d.columns if c not in exclude and not c.startswith('target_')]
    d=d[(d.next_gap_days==7)&((d.week_gap_days==7)|d.week_gap_days.isna())].replace([np.inf,-np.inf],np.nan).dropna(subset=features+['next_ratio']).reset_index(drop=True)
    return d,features

def choose_target(d):
    dev=d[d.week_start<'2024-01-01'].copy(); rows=[]
    for t in TARGET_CANDIDATES:
        y=(dev.next_ratio<(1-t)).astype(int); seg=[]
        for year in sorted(dev.week_start.dt.year.unique()):
            yy=y[dev.week_start.dt.year==year]; seg.append({'year':int(year),'rows':int(len(yy)),'positive_rate':float(yy.mean()),'positives':int(yy.sum())})
        rows.append({'threshold':t,'rows':int(len(y)),'positive_rate':float(y.mean()),'positive_count':int(y.sum()),'year_rates':seg,'eligible':bool(.15<=y.mean()<=.40 and all(s['positives']>=20 for s in seg if s['rows']>=100))})
    eligible=[r for r in rows if r['eligible']]; chosen=max(eligible,key=lambda r:r['threshold']) if eligible else min(rows,key=lambda r:abs(r['positive_rate']-.25)); return chosen,rows

def main():
    d,F=prepare(); chosen,target_diag=choose_target(d); threshold_decline=float(chosen['threshold']); d['target']=(d.next_ratio<(1-threshold_decline)).astype(int)
    train=d[d.week_start<'2024-01-01']; val=d[(d.week_start>='2024-01-07')&(d.week_start<'2025-01-01')]; test=d[d.week_start>='2025-01-05']
    print(json.dumps({'prepared_rows':len(d),'entities':int(d[['city','sector']].drop_duplicates().shape[0]),'train':len(train),'validation':len(val),'test':len(test),'features':len(F)},indent=2))
    if min(len(train),len(val),len(test))<200: raise RuntimeError(f'insufficient chronological splits {len(train)}/{len(val)}/{len(test)}')
    selection={}
    for n,s in models().items():
        q=clone(s).fit(train[F],train.target); p=q.predict_proba(val[F])[:,1]; best=None
        for pt in np.arange(.05,.951,.005):
            m=metric(val.target.to_numpy(),p,float(pt)); penalty=max(0,.70-m['Recall']); c=(score(m)-.20*penalty,m['BalancedAccuracy'],m['F1'],m['Accuracy'],-abs(pt-.5),float(pt),m)
            if best is None or c[:5]>best[:5]: best=c
        selection[n]={'probability_threshold':best[5],'validation_metrics':best[6],'score':best[0]}
    selected=max(selection,key=lambda n:(selection[n]['score'],selection[n]['validation_metrics']['BalancedAccuracy'],selection[n]['validation_metrics']['F1'])); pt=float(selection[selected]['probability_threshold'])
    fit=pd.concat([train,val]).sort_values('week_start'); model=clone(models()[selected]).fit(fit[F],fit.target); prob=model.predict_proba(test[F])[:,1]; tm=metric(test.target.to_numpy(),prob,pt); majority=max(float(test.target.mean()),1-float(test.target.mean()))
    gates={'accuracy_at_least_90pct':tm['Accuracy']>=.90,'balanced_accuracy_at_least_80pct':tm['BalancedAccuracy']>=.80,'recall_at_least_75pct':tm['Recall']>=.75,'f1_at_least_70pct':tm['F1']>=.70,'roc_auc_at_least_85pct':tm['ROC_AUC']>=.85,'beats_majority_accuracy':tm['Accuracy']>majority}
    artifact={'version':'SAMA-PANEL-DECLINE-2.0','source':'Official Saudi Central Bank (SAMA) Point of Sale Transactions by Sector and City','scientific_scope':'Saudi market segment risk by city-sector-week; not merchant-level observed transactions','panel':{'rows':int(len(d)),'entities':int(d[['city','sector']].drop_duplicates().shape[0]),'cities':int(d.city.nunique()),'sectors':int(d.sector.nunique()),'start':str(d.week_start.min().date()),'end':str(d.week_start.max().date())},'target_selection':{'development_only_before_2024':True,'chosen_decline_threshold':threshold_decline,'chosen':chosen,'candidates':target_diag},'splits':{'train_rows':int(len(train)),'validation_rows':int(len(val)),'test_rows':int(len(test)),'train_end':'2023-12-31','validation_year':'2024','test_start':'2025-01-05'},'features':F,'selection':selection,'selected_model':selected,'selected_probability_threshold':pt,'test_positive_rate':float(test.target.mean()),'test_metrics':tm,'majority_test_accuracy':majority,'acceptance_gates':gates,'all_acceptance_gates_passed':bool(all(gates.values())),'leakage_controls':{'shuffle':False,'target_threshold_selected_without_2024_or_2025':True,'model_and_probability_threshold_selected_on_2024_only':True,'2025_test_used_only_once_after_selection':True,'next_gap_days_excluded_from_features':True,'all_features_current_or_lagged_at_origin_week':True}}
    joblib.dump({'model':model,'features':F,'probability_threshold':pt,'decline_threshold':threshold_decline,'version':artifact['version']},OUT/'sama_market_decline_classifier_v2_0.joblib'); (OUT/'model_metadata_v2_0.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8'); (REP/'sama_panel_decline_v2_0_report.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8'); (REP/'sama_panel_decline_v2_0_summary.md').write_text(f"# SAMA Panel Decline v2.0\n\n- Real SAMA city-sector weekly panel rows: **{len(d):,}**\n- Entities: **{artifact['panel']['entities']}**\n- Cities: **{artifact['panel']['cities']}**\n- Sectors: **{artifact['panel']['sectors']}**\n- Target decline threshold: **{threshold_decline:.0%}**\n- 2025 test rows: **{len(test):,}**\n- Test positive rate: **{test.target.mean():.2%}**\n- Accuracy: **{tm['Accuracy']:.2%}**\n- Balanced Accuracy: **{tm['BalancedAccuracy']:.2%}**\n- Precision: **{tm['Precision']:.2%}**\n- Recall: **{tm['Recall']:.2%}**\n- F1: **{tm['F1']:.2%}**\n- ROC-AUC: **{tm['ROC_AUC']:.2%}**\n- Majority baseline: **{majority:.2%}**\n- Selected: **{selected}**, probability threshold **{pt:.3f}**\n- All acceptance gates passed: **{all(gates.values())}**\n",encoding='utf-8'); print(json.dumps({'panel':artifact['panel'],'target':chosen,'selected':selected,'test':tm,'majority':majority,'gates':gates,'all_passed':all(gates.values())},indent=2))
if __name__=='__main__': main()
