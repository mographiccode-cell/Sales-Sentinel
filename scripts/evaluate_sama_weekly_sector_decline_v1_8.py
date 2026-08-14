from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / 'data' / 'sama_pos' / 'sama_sector_weekly_value_count_2020_2025.csv'
FORECAST = ROOT / 'data' / 'sama_pos' / 'sama_sector_walkforward_forecasts_2023_2025.csv'
OUT = ROOT / 'reports' / 'sama_sector_decline_v1_8'
MODELS = ROOT / 'models' / 'sama_sector_decline_v1_8'
OUT.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
REPORT = OUT / 'report.json'; SUMMARY = OUT / 'summary.md'; MODEL = MODELS / 'market_decline_classifier.joblib'
DECLINE = 0.20


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    return {
        'Accuracy': float(accuracy_score(y, pred)),
        'BalancedAccuracy': float(balanced_accuracy_score(y, pred)),
        'Precision': float(precision_score(y, pred, zero_division=0)),
        'Recall': float(recall_score(y, pred, zero_division=0)),
        'F1': float(f1_score(y, pred, zero_division=0)),
        'ROC_AUC': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan'),
    }


def choose_threshold(y, p):
    best = None
    for t in np.linspace(0.05, 0.95, 181):
        m = metrics(y, p, float(t))
        # prioritize balanced accuracy/F1 while requiring useful decline recall
        score = 0.45*m['BalancedAccuracy'] + 0.30*m['F1'] + 0.25*m['ROC_AUC'] - 0.35*max(0.0, 0.70-m['Recall'])
        cand = (score, m['BalancedAccuracy'], m['F1'], m['Recall'], -abs(t-.5), float(t), m)
        if best is None or cand[:5] > best[:5]: best = cand
    return best[5], best[6], best[0]


def main():
    h = pd.read_csv(HISTORY, parse_dates=['week_start']).sort_values(['sector','week_start'])
    f = pd.read_csv(FORECAST, parse_dates=['origin_week_start']).sort_values(['sector','origin_week_start'])
    # keep official SAMA sector totals only; no merchant synthetic regions are involved.
    g = h.groupby('sector', group_keys=False)
    h['value_mean4'] = g['value_thousand_sar'].transform(lambda s: s.rolling(4, min_periods=4).mean())
    h['count_mean4'] = g['transaction_count_thousand'].transform(lambda s: s.rolling(4, min_periods=4).mean())
    h['value_prev_change'] = g['value_thousand_sar'].pct_change()
    h['count_prev_change'] = g['transaction_count_thousand'].pct_change()

    base = h[['sector','week_start','value_thousand_sar','transaction_count_thousand','value_mean4','count_mean4','value_prev_change','count_prev_change']].rename(columns={'week_start':'origin_week_start','value_thousand_sar':'origin_value','transaction_count_thousand':'origin_count'})
    d = f.merge(base, on=['sector','origin_week_start'], how='inner', validate='one_to_one')
    nxt = h[['sector','week_start','value_thousand_sar','transaction_count_thousand']].copy()
    nxt['origin_week_start'] = nxt['week_start'] - pd.Timedelta(days=7)
    nxt = nxt.rename(columns={'value_thousand_sar':'actual_next_value','transaction_count_thousand':'actual_next_count'})[['sector','origin_week_start','actual_next_value','actual_next_count']]
    d = d.merge(nxt, on=['sector','origin_week_start'], how='left', validate='one_to_one')
    d['actual_ratio'] = d['actual_next_value'] / d['value_mean4']
    d['target'] = (d['actual_ratio'] < 1.0-DECLINE).astype(int)
    d['predicted_value_ratio'] = d['predicted_value_h1'] / d['value_mean4']
    d['predicted_count_ratio'] = d['predicted_count_h1'] / d['count_mean4']
    d['predicted_value_vs_origin'] = d['predicted_value_h1'] / d['origin_value']
    d['predicted_count_vs_origin'] = d['predicted_count_h1'] / d['origin_count']
    week = d['origin_week_start'].dt.isocalendar().week.astype(float)
    d['week_sin'] = np.sin(2*np.pi*week/52.18); d['week_cos'] = np.cos(2*np.pi*week/52.18)
    cats = pd.get_dummies(d['sector'], prefix='sector', dtype=float)
    X = pd.concat([d[['predicted_value_ratio','predicted_count_ratio','predicted_value_vs_origin','predicted_count_vs_origin','predicted_value_h1_change_vs_last','predicted_count_h1_change_vs_last','value_prev_change','count_prev_change','week_sin','week_cos']].reset_index(drop=True), cats.reset_index(drop=True)], axis=1)
    good = d['actual_next_value'].notna() & d['value_mean4'].gt(0) & d['count_mean4'].gt(0) & X.notna().all(axis=1)
    d = d.loc[good].reset_index(drop=True); X = X.loc[good].reset_index(drop=True)

    train_mask = d['origin_week_start'] <= pd.Timestamp('2023-12-31')
    val_mask = d['origin_week_start'].between(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-04-30'))
    test_mask = d['origin_week_start'] >= pd.Timestamp('2024-05-01')
    train, val, test = d[train_mask], d[val_mask], d[test_mask]
    Xtr, Xv, Xt = X[train_mask], X[val_mask], X[test_mask]
    if min(len(train),len(val),len(test)) < 150: raise RuntimeError(f'insufficient splits {len(train)}/{len(val)}/{len(test)}')
    if min(train.target.sum(), val.target.sum(), test.target.sum()) < 10: raise RuntimeError('too few decline positives in one split')

    models = {
        'LogisticRegression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42)),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=900, max_depth=12, min_samples_leaf=4, max_features=.7, class_weight='balanced', random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=350, learning_rate=.04, max_leaf_nodes=24, min_samples_leaf=20, l2_regularization=2.0, random_state=42),
    }
    val_results = {}; fitted = {}
    for name, model in models.items():
        fit = clone(model).fit(Xtr, train.target)
        pv = fit.predict_proba(Xv)[:,1]
        t,m,s = choose_threshold(val.target.to_numpy(), pv)
        val_results[name] = {'threshold':t,'metrics':m,'selection_score':s}; fitted[name]=fit
    best = max(val_results, key=lambda n:(val_results[n]['selection_score'],val_results[n]['metrics']['ROC_AUC']))
    threshold = val_results[best]['threshold']

    trainval_mask = train_mask | val_mask
    final = clone(models[best]).fit(X[trainval_mask], d.loc[trainval_mask,'target'])
    pt = final.predict_proba(Xt)[:,1]
    tm = metrics(test.target.to_numpy(), pt, threshold)
    majority = max(float(test.target.mean()), 1-float(test.target.mean()))

    # Also report the pure regression/forecast rule with no learned classifier.
    direct_score = 1.0 - np.clip(test['predicted_value_ratio'].to_numpy()/max(1.0-DECLINE,1e-9),0,2)
    direct_pred = (test['predicted_value_ratio'].to_numpy() < 1.0-DECLINE).astype(int)
    direct = {
        'Accuracy': float(accuracy_score(test.target, direct_pred)),
        'BalancedAccuracy': float(balanced_accuracy_score(test.target, direct_pred)),
        'Precision': float(precision_score(test.target,direct_pred,zero_division=0)),
        'Recall': float(recall_score(test.target,direct_pred,zero_division=0)),
        'F1': float(f1_score(test.target,direct_pred,zero_division=0)),
    }

    report = {
        'version':'SAMA-SECTOR-DECLINE-1.8',
        'source':'Saudi Central Bank (SAMA) official weekly POS sector aggregates',
        'target':'next official sector week value is >=20% below trailing four official weeks mean',
        'split':{'train_rows':len(train),'val_rows':len(val),'test_rows':len(test),'train_end':'2023-12-31','validation':'2024-01-01..2024-04-30','test_start':'2024-05-01','shuffle':False},
        'positive_rates':{'train':float(train.target.mean()),'validation':float(val.target.mean()),'test':float(test.target.mean())},
        'validation_candidates':val_results,
        'selected_model':best,'selected_threshold':threshold,
        'test_metrics':tm,'majority_test_accuracy':majority,'direct_forecast_rule_test':direct,
        'accuracy_90_goal_met':bool(tm['Accuracy']>=.90),
        'scientific_gates':{
            'beats_majority_by_3pp':tm['Accuracy']>=majority+.03,
            'balanced_accuracy_75':tm['BalancedAccuracy']>=.75,
            'recall_70':tm['Recall']>=.70,
            'f1_65':tm['F1']>=.65,
            'roc_auc_82':tm['ROC_AUC']>=.82,
        },
        'leakage_controls':{'walkforward_forecasts_only':True,'future_actual_SAMA_not_feature':True,'test_not_used_for_model_or_threshold_selection':True,'synthetic_merchant_region_not_used':True}
    }
    report['all_scientific_gates_passed']=bool(all(report['scientific_gates'].values()))
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    SUMMARY.write_text(f"""# SAMA Weekly Sector Decline v1.8\n\n- Selected model: **{best}**\n- Train/Validation/Test: **{len(train)}/{len(val)}/{len(test)}**\n- Test decline rate: **{float(test.target.mean()):.2%}**\n- Accuracy: **{tm['Accuracy']:.2%}**\n- Balanced Accuracy: **{tm['BalancedAccuracy']:.2%}**\n- Recall: **{tm['Recall']:.2%}**\n- F1: **{tm['F1']:.2%}**\n- ROC-AUC: **{tm['ROC_AUC']:.2%}**\n- Majority baseline: **{majority:.2%}**\n- 90% goal met: **{report['accuracy_90_goal_met']}**\n- All scientific gates: **{report['all_scientific_gates_passed']}**\n""",encoding='utf-8')
    joblib.dump({'model':final,'features':list(X.columns),'threshold':threshold,'decline_threshold':DECLINE,'version':report['version']},MODEL)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
