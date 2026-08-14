from __future__ import annotations

import json
from collections import Counter
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_sama_panel_decline_v2_0 import prepare

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'models' / 'sama_panel_v2_2'
REP = ROOT / 'reports' / 'sama_panel_v2_2'
OUT.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)
DECLINE_RATIO = 0.90
VAL_WEEKS = 13
TEST_WEEKS = 13
PURGE_WEEKS = 1

# Historical Saudi calendar windows used only as known-ahead calendar features.
RAMADAN = [
    ('2020-04-24','2020-05-23'),('2021-04-13','2021-05-12'),('2022-04-02','2022-05-01'),
    ('2023-03-23','2023-04-20'),('2024-03-11','2024-04-09'),('2025-03-01','2025-03-29')
]
EID_FITR = [
    ('2020-05-24','2020-05-27'),('2021-05-13','2021-05-16'),('2022-05-02','2022-05-05'),
    ('2023-04-21','2023-04-24'),('2024-04-10','2024-04-13'),('2025-03-30','2025-04-02')
]
EID_ADHA = [
    ('2020-07-31','2020-08-03'),('2021-07-20','2021-07-23'),('2022-07-09','2022-07-12'),
    ('2023-06-28','2023-07-01'),('2024-06-16','2024-06-19'),('2025-06-06','2025-06-09')
]


def week_overlaps(series: pd.Series, windows) -> np.ndarray:
    starts = pd.to_datetime(series).to_numpy(dtype='datetime64[D]')
    ends = starts + np.timedelta64(6, 'D')
    out = np.zeros(len(series), dtype=float)
    for a, b in windows:
        aa = np.datetime64(a, 'D'); bb = np.datetime64(b, 'D')
        out |= ((starts <= bb) & (ends >= aa))
    return out.astype(float)


def augment(d: pd.DataFrame, base_features: list[str]):
    z = d.copy().sort_values(['entity_id','week_start']).reset_index(drop=True)
    g = z.groupby('entity_id', sort=False)

    # Unit economics / average ticket. Both inputs are known at origin week.
    z['avg_ticket'] = z['pos_value_thousand_sar'] / z['transaction_count'].replace(0, np.nan)
    z['log_avg_ticket'] = np.log1p(z['avg_ticket'])
    for lag in (1, 2, 4, 13, 52):
        z[f'avg_ticket_lag_{lag}'] = g['avg_ticket'].shift(lag)
    for w in (4, 13, 26):
        z[f'avg_ticket_mean_{w}'] = g['avg_ticket'].transform(lambda x: x.rolling(w).mean())
        z[f'avg_ticket_ratio_{w}'] = z['avg_ticket'] / z[f'avg_ticket_mean_{w}'].replace(0, np.nan)
    z['avg_ticket_pct1'] = g['avg_ticket'].pct_change(1)
    z['avg_ticket_pct4'] = g['avg_ticket'].pct_change(4)
    z['avg_ticket_pct13'] = g['avg_ticket'].pct_change(13)

    # Same-week-last-year / medium-term momentum and acceleration.
    z['value_yoy_ratio'] = z['pos_value_thousand_sar'] / np.expm1(z['log_value_lag_52']).replace(0, np.nan)
    z['count_yoy_ratio'] = z['transaction_count'] / np.expm1(z['log_count_lag_52']).replace(0, np.nan)
    z['value_momentum_4_13'] = z['value_pct_change4'] - z['value_pct_change13']
    z['count_momentum_4_13'] = z['count_pct_change4'] - z['count_pct_change13']
    z['value_acceleration'] = z['value_pct_change1'] - z['value_pct_change4'] / 4.0
    z['count_acceleration'] = z['count_pct_change1'] - z['count_pct_change4'] / 4.0
    z['value_cv_13'] = z['value_std_13'] / z['value_mean_13'].replace(0, np.nan)
    z['count_cv_13'] = z['count_std_13'] / z['count_mean_13'].replace(0, np.nan)

    # National market spread: does this entity weaken more than the Saudi market?
    z['value_vs_national_1'] = z['value_pct_change1'] - z['national_value_pct1']
    z['value_vs_national_4'] = z['value_pct_change4'] - z['national_value_pct4']
    z['count_vs_national_1'] = z['count_pct_change1'] - z['national_count_pct1']
    z['count_vs_national_4'] = z['count_pct_change4'] - z['national_count_pct4']

    # National average-ticket trend, mapped once per week to avoid repeated-panel pct_change errors.
    n = z[['week_start','national_value','national_count']].drop_duplicates('week_start').sort_values('week_start').copy()
    n['national_ticket'] = n['national_value'] / n['national_count'].replace(0, np.nan)
    n['national_ticket_pct1'] = n['national_ticket'].pct_change(1)
    n['national_ticket_pct4'] = n['national_ticket'].pct_change(4)
    nt1 = n.set_index('week_start')['national_ticket_pct1']; nt4 = n.set_index('week_start')['national_ticket_pct4']
    z['national_ticket_pct1'] = z['week_start'].map(nt1)
    z['national_ticket_pct4'] = z['week_start'].map(nt4)
    z['ticket_vs_national_1'] = z['avg_ticket_pct1'] - z['national_ticket_pct1']
    z['ticket_vs_national_4'] = z['avg_ticket_pct4'] - z['national_ticket_pct4']

    # Cross-sectional market breadth and relative strength, separated by City/Sector panels.
    panel = z.groupby(['entity_type','week_start'], sort=False)
    for c in ('value_pct_change1','value_pct_change4','count_pct_change1','count_pct_change4','avg_ticket_pct1'):
        mean = panel[c].transform('mean')
        median = panel[c].transform('median')
        std = panel[c].transform('std').replace(0, np.nan)
        z[f'{c}_panel_mean'] = mean
        z[f'{c}_panel_median'] = median
        z[f'{c}_panel_z'] = (z[c] - mean) / std
        down_col = f'__{c}_down'
        z[down_col] = (z[c] < 0).astype(float)
        z[f'{c}_breadth_down'] = z.groupby(['entity_type','week_start'], sort=False)[down_col].transform('mean')
        z.drop(columns=[down_col], inplace=True)

    z['panel_value_sum'] = panel['pos_value_thousand_sar'].transform('sum')
    z['panel_count_sum'] = panel['transaction_count'].transform('sum')
    z['entity_value_share'] = z['pos_value_thousand_sar'] / z['panel_value_sum'].replace(0, np.nan)
    z['entity_count_share'] = z['transaction_count'] / z['panel_count_sum'].replace(0, np.nan)
    z['entity_value_share_lag1'] = g['entity_value_share'].shift(1)
    z['entity_count_share_lag1'] = g['entity_count_share'].shift(1)
    z['entity_value_share_change'] = z['entity_value_share'] - z['entity_value_share_lag1']
    z['entity_count_share_change'] = z['entity_count_share'] - z['entity_count_share_lag1']

    # Multi-harmonic annual seasonality; the original model only had harmonic 1.
    week = z['week_start'].dt.isocalendar().week.astype(float)
    for k in range(2, 7):
        z[f'week_sin_{k}'] = np.sin(2*np.pi*k*week/52.18)
        z[f'week_cos_{k}'] = np.cos(2*np.pi*k*week/52.18)
    z['quarter'] = z['week_start'].dt.quarter.astype(float)
    z['month'] = z['week_start'].dt.month.astype(float)
    z['is_year_end'] = z['week_start'].dt.month.isin([11,12]).astype(float)
    z['is_summer'] = z['week_start'].dt.month.isin([6,7,8]).astype(float)
    z['is_ramadan_week'] = week_overlaps(z['week_start'], RAMADAN)
    z['is_eid_fitr_week'] = week_overlaps(z['week_start'], EID_FITR)
    z['is_eid_adha_week'] = week_overlaps(z['week_start'], EID_ADHA)
    z['is_national_day_week'] = (((z['week_start'] <= pd.to_datetime(z['week_start'].dt.year.astype(str)+'-09-23')) & ((z['week_start'] + pd.Timedelta(days=6)) >= pd.to_datetime(z['week_start'].dt.year.astype(str)+'-09-23')))).astype(float)
    foundation = pd.to_datetime(z['week_start'].dt.year.astype(str)+'-02-22')
    z['is_foundation_day_week'] = ((z['week_start'].dt.year >= 2022) & (z['week_start'] <= foundation) & ((z['week_start'] + pd.Timedelta(days=6)) >= foundation)).astype(float)
    # Salary is normally due near day 27; this is a deterministic known-ahead calendar proxy.
    month_start = z['week_start'].dt.to_period('M').dt.start_time
    pay_this = month_start + pd.to_timedelta(26, unit='D')
    next_month_start = (z['week_start'] + pd.offsets.MonthBegin(1)).dt.to_period('M').dt.start_time
    pay_next = next_month_start + pd.to_timedelta(26, unit='D')
    week_end = z['week_start'] + pd.Timedelta(days=6)
    z['is_salary_week'] = (((z['week_start'] <= pay_this) & (week_end >= pay_this)) | ((z['week_start'] <= pay_next) & (week_end >= pay_next))).astype(float)

    excluded = {'week_start','next_week_start','next_gap_days','city','sector','entity_type','entity_name','entity_id','pos_value_thousand_sar','transaction_count','next_value','next_ratio','target','panel_value_sum','panel_count_sum'}
    all_features = [c for c in z.columns if c not in excluded and pd.api.types.is_numeric_dtype(z[c])]
    z = z.replace([np.inf,-np.inf], np.nan).dropna(subset=all_features + ['next_ratio']).reset_index(drop=True)
    return z, all_features


def specs():
    return {
        'LogisticRegression': Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(C=.35, max_iter=7000, class_weight='balanced', random_state=SEED))]),
        'RandomForest': RandomForestClassifier(n_estimators=1000, max_depth=13, min_samples_leaf=3, max_features=.55, class_weight='balanced_subsample', random_state=SEED, n_jobs=-1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=1200, max_depth=16, min_samples_leaf=2, max_features=.65, class_weight='balanced', random_state=SEED, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingClassifier(learning_rate=.03, max_iter=500, max_leaf_nodes=24, min_samples_leaf=18, l2_regularization=3.0, random_state=SEED),
        'XGBoost': xgb.XGBClassifier(n_estimators=850, max_depth=5, learning_rate=.022, min_child_weight=7, subsample=.85, colsample_bytree=.72, reg_alpha=.25, reg_lambda=5.0, objective='binary:logistic', eval_metric='logloss', random_state=SEED, n_jobs=-1),
    }


def cls_metrics(y, p, threshold):
    z = (p >= threshold).astype(int)
    return {
        'Accuracy': float(accuracy_score(y,z)), 'BalancedAccuracy': float(balanced_accuracy_score(y,z)),
        'Precision': float(precision_score(y,z,zero_division=0)), 'Recall': float(recall_score(y,z,zero_division=0)),
        'F1': float(f1_score(y,z,zero_division=0)), 'ROC_AUC': float(roc_auc_score(y,p)),
        'ConfusionMatrix': confusion_matrix(y,z,labels=[0,1]).tolist()
    }


def metric_score(m):
    return .24*m['BalancedAccuracy'] + .22*m['F1'] + .20*m['Accuracy'] + .20*m['ROC_AUC'] + .14*m['Recall']


def selective_metrics(y, p, lo, hi):
    mask = (p <= lo) | (p >= hi)
    if mask.sum() == 0:
        return {'Coverage':0.0,'DecisionAccuracy':0.0,'BalancedAccuracy':0.0,'F1':0.0,'DeclinePrecision':0.0,'DeclineRecallOnAll':0.0,'Decisions':0,'DeclineDecisions':0,'StableDecisions':0}
    pred = np.where(p[mask] >= hi, 1, 0)
    yy = y[mask]
    ba = balanced_accuracy_score(yy,pred) if len(np.unique(yy)) == 2 else 0.5
    f1 = f1_score(yy,pred,zero_division=0)
    precision = precision_score(yy,pred,zero_division=0)
    decline_recall_all = float(((p >= hi) & (y == 1)).sum() / max(1,(y == 1).sum()))
    return {
        'Coverage': float(mask.mean()), 'DecisionAccuracy': float(accuracy_score(yy,pred)),
        'BalancedAccuracy': float(ba), 'F1': float(f1), 'DeclinePrecision': float(precision),
        'DeclineRecallOnAll': decline_recall_all, 'Decisions': int(mask.sum()),
        'DeclineDecisions': int((pred==1).sum()), 'StableDecisions': int((pred==0).sum())
    }


def fit_probs(train, val, features):
    fitted = {}; probs = {}
    pos = float(train.target.mean())
    for name, spec in specs().items():
        q = clone(spec)
        if name == 'XGBoost':
            q.set_params(scale_pos_weight=max(1.0,(1-pos)/max(pos,1e-6)))
        q.fit(train[features], train.target)
        fitted[name] = q
        probs[name] = q.predict_proba(val[features])[:,1]
    return fitted, probs


def blend_candidates(names):
    # Individual models + coarse convex blends in 0.25 increments.
    out=[]; n=len(names)
    for units in product(range(5), repeat=n):
        if sum(units) != 4: continue
        w=np.asarray(units,dtype=float)/4.0
        out.append(w)
    return out


def choose(train, val, features):
    fitted, probs = fit_probs(train,val,features); names=list(probs)
    matrix=np.column_stack([probs[n] for n in names])
    best=None
    for w in blend_candidates(names):
        p=matrix@w
        for t in np.arange(.20,.801,.01):
            m=cls_metrics(val.target.to_numpy(),p,float(t))
            # Penalize solutions that simply sacrifice decline recall.
            penalty=max(0,.68-m['Recall'])*.22
            cand=(metric_score(m)-penalty,m['BalancedAccuracy'],m['F1'],m['Accuracy'],m['ROC_AUC'],-abs(t-.5),w,float(t),m)
            if best is None or cand[:6]>best[:6]: best=cand
    w=best[6]; threshold=best[7]; p=matrix@w

    # High-confidence selective mode: maximize coverage subject to >=90% validation decision accuracy.
    sbest=None
    for lo in np.arange(.05,.451,.025):
        for hi in np.arange(.55,.951,.025):
            if lo >= hi: continue
            sm=selective_metrics(val.target.to_numpy(),p,float(lo),float(hi))
            if sm['Decisions'] < 60 or sm['DeclineDecisions'] < 10 or sm['StableDecisions'] < 20: continue
            if sm['DecisionAccuracy'] < .90: continue
            cand=(sm['Coverage'],sm['BalancedAccuracy'],sm['F1'],sm['DecisionAccuracy'],-(hi-lo),float(lo),float(hi),sm)
            if sbest is None or cand[:5]>sbest[:5]: sbest=cand
    if sbest is None:
        # Fall back to best-accuracy selective band; still report honestly.
        for lo in np.arange(.05,.451,.025):
            for hi in np.arange(.55,.951,.025):
                sm=selective_metrics(val.target.to_numpy(),p,float(lo),float(hi))
                if sm['Decisions'] < 40 or sm['DeclineDecisions'] < 5 or sm['StableDecisions'] < 15: continue
                cand=(sm['DecisionAccuracy'],sm['Coverage'],sm['BalancedAccuracy'],sm['F1'],float(lo),float(hi),sm)
                if sbest is None or cand[:4]>sbest[:4]: sbest=('fallback',)+cand
        if sbest and sbest[0]=='fallback':
            lo,hi,sm=sbest[5],sbest[6],sbest[7]
        else:
            lo,hi,sm=.25,.75,selective_metrics(val.target.to_numpy(),p,.25,.75)
    else:
        lo,hi,sm=sbest[5],sbest[6],sbest[7]
    return {'names':names,'weights':w.tolist(),'threshold':threshold,'validation':best[8],'selective_lo':lo,'selective_hi':hi,'selective_validation':sm}


def predict_blend(train, test, features, selection):
    names=selection['names']; weights=np.asarray(selection['weights']); pos=float(train.target.mean())
    cols=[]; fitted={}
    for name in names:
        q=clone(specs()[name])
        if name=='XGBoost': q.set_params(scale_pos_weight=max(1.0,(1-pos)/max(pos,1e-6)))
        q.fit(train[features],train.target); fitted[name]=q; cols.append(q.predict_proba(test[features])[:,1])
    return np.column_stack(cols)@weights, fitted


def main():
    raw, base_features = prepare()
    d, features = augment(raw, base_features)
    d['target']=(d.next_ratio < DECLINE_RATIO).astype(int)
    starts=[pd.Timestamp('2024-01-07'),pd.Timestamp('2024-04-07'),pd.Timestamp('2024-07-07'),pd.Timestamp('2024-10-06'),pd.Timestamp('2025-01-05'),pd.Timestamp('2025-04-06')]
    all_y=[]; all_p=[]; all_z=[]; selective_rows=[]; folds=[]; model_weight_counter=Counter()

    for i,start in enumerate(starts,1):
        end=start+pd.Timedelta(weeks=TEST_WEEKS)
        test=d[(d.week_start>=start)&(d.week_start<end)]
        if len(test)<150: continue
        val_end=start-pd.Timedelta(weeks=PURGE_WEEKS); val_start=val_end-pd.Timedelta(weeks=VAL_WEEKS)
        train=d[d.week_start < val_start-pd.Timedelta(weeks=PURGE_WEEKS)]
        val=d[(d.week_start>=val_start)&(d.week_start<val_end)]
        fit=d[d.week_start < start-pd.Timedelta(weeks=PURGE_WEEKS)]
        if len(train)<900 or len(val)<150: continue
        sel=choose(train,val,features)
        p,_=predict_blend(fit,test,features,sel); y=test.target.to_numpy(); z=(p>=sel['threshold']).astype(int)
        all_y.extend(y.tolist()); all_p.extend(p.tolist()); all_z.extend(z.tolist())
        sm=selective_metrics(y,p,sel['selective_lo'],sel['selective_hi'])
        selective_rows.append({'y':y,'p':p,'lo':sel['selective_lo'],'hi':sel['selective_hi']})
        for n,w in zip(sel['names'],sel['weights']): model_weight_counter[n]+=float(w)
        folds.append({'fold':i,'test_start':str(start.date()),'test_end':str((end-pd.Timedelta(days=1)).date()),'train_rows':int(len(fit)),'validation_rows':int(len(val)),'test_rows':int(len(test)),'selection':sel,'test_metrics':cls_metrics(y,p,sel['threshold']),'selective_test':sm})

    y=np.asarray(all_y,dtype=int); p=np.asarray(all_p); z=np.asarray(all_z,dtype=int)
    agg={'Accuracy':float(accuracy_score(y,z)),'BalancedAccuracy':float(balanced_accuracy_score(y,z)),'Precision':float(precision_score(y,z,zero_division=0)),'Recall':float(recall_score(y,z,zero_division=0)),'F1':float(f1_score(y,z,zero_division=0)),'ROC_AUC':float(roc_auc_score(y,p)),'ConfusionMatrix':confusion_matrix(y,z,labels=[0,1]).tolist()}
    # Aggregate selective decisions using the fold-specific validation-selected confidence bands.
    sy=[]; sz=[]; all_total=0; decline_hits=0; decline_total=int((y==1).sum())
    for r in selective_rows:
        yy=r['y']; pp=r['p']; mask=(pp<=r['lo'])|(pp>=r['hi']); pred=np.where(pp[mask]>=r['hi'],1,0); sy.extend(yy[mask].tolist()); sz.extend(pred.tolist()); all_total+=len(yy); decline_hits+=int(((pp>=r['hi'])&(yy==1)).sum())
    sy=np.asarray(sy,dtype=int); sz=np.asarray(sz,dtype=int)
    selective={'Coverage':float(len(sy)/max(1,all_total)),'DecisionAccuracy':float(accuracy_score(sy,sz)) if len(sy) else 0.0,'BalancedAccuracy':float(balanced_accuracy_score(sy,sz)) if len(sy) and len(np.unique(sy))==2 else 0.5,'Precision':float(precision_score(sy,sz,zero_division=0)) if len(sy) else 0.0,'RecallWithinDecisions':float(recall_score(sy,sz,zero_division=0)) if len(sy) else 0.0,'F1':float(f1_score(sy,sz,zero_division=0)) if len(sy) else 0.0,'DeclineRecallOnAll':float(decline_hits/max(1,decline_total)),'Decisions':int(len(sy)),'TotalRows':int(all_total)}
    positive=float(y.mean()); majority=max(positive,1-positive)
    gates={'full_accuracy_at_least_85pct':agg['Accuracy']>=.85,'full_balanced_accuracy_at_least_80pct':agg['BalancedAccuracy']>=.80,'full_roc_auc_at_least_85pct':agg['ROC_AUC']>=.85,'beats_majority_accuracy':agg['Accuracy']>majority,'high_confidence_accuracy_at_least_90pct':selective['DecisionAccuracy']>=.90,'high_confidence_coverage_at_least_30pct':selective['Coverage']>=.30}

    # Deployment selection from the latest completed validation block.
    latest=d.week_start.max()+pd.Timedelta(days=7); val_end=latest-pd.Timedelta(weeks=PURGE_WEEKS); val_start=val_end-pd.Timedelta(weeks=VAL_WEEKS)
    tr=d[d.week_start < val_start-pd.Timedelta(weeks=PURGE_WEEKS)]; va=d[(d.week_start>=val_start)&(d.week_start<val_end)]; fit=d[d.week_start<val_end]
    selection=choose(tr,va,features); _, fitted=predict_blend(fit,fit.iloc[:1],features,selection)
    bundle={'models':fitted,'features':features,'selection':selection,'true_decline_ratio':DECLINE_RATIO,'version':'SAMA-PANEL-ENHANCED-2.2'}
    joblib.dump(bundle,OUT/'sama_panel_enhanced_ensemble_v2_2.joblib')

    report={'version':'SAMA-PANEL-ENHANCED-2.2','source':'Official SAMA city totals + national sector totals','scientific_scope':'One-week-ahead Saudi market segment decline risk. High-confidence mode may abstain instead of fabricating certainty.','target':'next-week POS value / trailing 4-week mean < 0.90','evaluation':'six expanding chronological backtests; each fold blend, standard threshold, and selective confidence band selected on prior validation only','panel_rows':int(len(d)),'entities':int(d.entity_id.nunique()),'feature_count':int(len(features)),'base_feature_count':int(len(base_features)),'xgboost_version':xgb.__version__,'folds':folds,'aggregate_full_coverage':agg,'aggregate_high_confidence':selective,'positive_rate':positive,'majority_accuracy':majority,'aggregate_model_weights':dict(model_weight_counter),'acceptance_gates':gates,'deployment_selection':selection,'leakage_controls':{'shuffle':False,'future_target_columns_excluded':True,'cross_sectional_features_use_origin_week_only':True,'calendar_features_known_ahead':True,'fold_thresholds_selected_before_test':True,'uncertain_cases_not_counted_as correct in full_coverage_metrics':True}}
    (REP/'sama_panel_enhanced_v2_2_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'model_metadata_v2_2.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    summary=f"""# SAMA Panel Enhanced v2.2\n\n- Backtest rows: **{len(y):,}**\n- Features: **{len(features)}** (base {len(base_features)})\n- Positive rate: **{positive:.2%}**\n- Full-coverage Accuracy: **{agg['Accuracy']:.2%}**\n- Full-coverage Balanced Accuracy: **{agg['BalancedAccuracy']:.2%}**\n- Full-coverage F1: **{agg['F1']:.2%}**\n- Full-coverage ROC-AUC: **{agg['ROC_AUC']:.2%}**\n- Majority baseline: **{majority:.2%}**\n\n## High-confidence selective mode\n- Decision Accuracy: **{selective['DecisionAccuracy']:.2%}**\n- Coverage: **{selective['Coverage']:.2%}**\n- Balanced Accuracy: **{selective['BalancedAccuracy']:.2%}**\n- Precision: **{selective['Precision']:.2%}**\n- F1: **{selective['F1']:.2%}**\n- Decline recall across all rows: **{selective['DeclineRecallOnAll']:.2%}**\n- Decisions: **{selective['Decisions']:,}/{selective['TotalRows']:,}**\n\n- Gates: **{gates}**\n"""
    (REP/'sama_panel_enhanced_v2_2_summary.md').write_text(summary,encoding='utf-8')
    print(json.dumps({'full':agg,'selective':selective,'positive_rate':positive,'majority':majority,'features':len(features),'gates':gates,'deployment':selection},indent=2))

if __name__=='__main__':
    main()
