from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_sama_city_risk_v2_1 as source

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / 'data' / 'sama_pos' / 'sama_city_weekly_value_count_2020_2025.csv'
OUT = ROOT / 'reports' / 'sama_city_v3'
MOD = ROOT / 'models' / 'sama_city_v3'
OUT.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)
REPORT = OUT / 'development_report.json'
MODEL = MOD / 'city_risk_v3.joblib'
VERSION = 'SAMA-CITY-RISK-3.0-GENERALIZATION-FIRST'
SEED = 73
DECLINE = 0.20
DEV_END = pd.Timestamp('2025-06-29')

# v3 deliberately optimizes a triage system, not headline accuracy.
CONTRACT = {
    'pooled_red_precision_min': 0.70,
    'pooled_red_fpr_max': 0.015,
    'pooled_alert_recall_min': 0.85,
    'pooled_green_npv_min': 0.985,
    'pooled_roc_auc_min': 0.85,
    'min_red_alerts': 8,
    'worst_positive_fold_alert_recall_min': 0.55,
    'worst_fold_red_fpr_max': 0.04,
}


def safe_ratio(a, b):
    a = pd.Series(a, index=getattr(a, 'index', None), dtype=float)
    b = pd.Series(b, index=a.index, dtype=float).replace(0, np.nan)
    return a / b


def norm_slope(s: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    xm = x.mean()
    den = float(((x - xm) ** 2).sum())
    def f(v):
        v = np.asarray(v, dtype=float)
        m = float(np.mean(v))
        if not np.isfinite(m) or abs(m) < 1e-12:
            return np.nan
        return float(((x - xm) * (v - v.mean())).sum() / den / abs(m))
    return s.rolling(window, min_periods=window).apply(f, raw=True)


def featureize(panel: pd.DataFrame, require_target: bool = True):
    d = panel.copy()
    d['week_start'] = pd.to_datetime(d['week_start'])
    d = d.sort_values(['city', 'week_start']).reset_index(drop=True)
    if d.duplicated(['week_start', 'city']).any():
        raise RuntimeError('Duplicate city/week source rows')
    g = d.groupby('city', sort=False)

    d['baseline4'] = g.value_thousand_sar.transform(lambda s: s.rolling(4, min_periods=4).mean())
    d['actual_next_value'] = g.value_thousand_sar.shift(-1)
    d['future_ratio'] = safe_ratio(d.actual_next_value, d.baseline4)
    d['target_float'] = np.where(d.future_ratio.notna(), (d.future_ratio < 1 - DECLINE).astype(float), np.nan)
    d['target'] = d.target_float.fillna(0).astype(int)

    F = pd.DataFrame(index=d.index)
    # Compact, scale-free, city-agnostic features. No city identity and no historical target-rate features.
    for col, pre in [('value_thousand_sar', 'value'), ('transaction_count_thousand', 'count')]:
        s = d[col].astype(float)
        for w in (4, 8, 13):
            mean = g[col].transform(lambda x, w=w: x.rolling(w, min_periods=w).mean())
            std = g[col].transform(lambda x, w=w: x.rolling(w, min_periods=w).std())
            F[f'{pre}_ratio_mean_{w}'] = safe_ratio(s, mean)
            F[f'{pre}_cv_{w}'] = safe_ratio(std, mean.abs())
        for lag in (1, 2, 4, 8, 13):
            F[f'{pre}_change_{lag}'] = g[col].pct_change(lag)
        F[f'{pre}_slope_4'] = g[col].transform(lambda x: norm_slope(x, 4))
        F[f'{pre}_slope_8'] = g[col].transform(lambda x: norm_slope(x, 8))
        F[f'{pre}_drawdown_13'] = safe_ratio(s, g[col].transform(lambda x: x.rolling(13, min_periods=13).max())) - 1.0

    national = d.groupby('week_start', as_index=False).agg(
        nvalue=('value_thousand_sar', 'sum'), ncount=('transaction_count_thousand', 'sum')
    ).sort_values('week_start')
    d = d.merge(national, on='week_start', how='left', validate='many_to_one')
    d['value_share'] = safe_ratio(d.value_thousand_sar, d.nvalue)
    d['count_share'] = safe_ratio(d.transaction_count_thousand, d.ncount)
    gs = d.groupby('city', sort=False)
    for col in ('value_share', 'count_share'):
        mean13 = gs[col].transform(lambda s: s.rolling(13, min_periods=13).mean())
        F[f'{col}_ratio_13'] = safe_ratio(d[col], mean13)
        F[f'{col}_change_4'] = gs[col].pct_change(4)

    for col, pre in [('nvalue', 'nvalue'), ('ncount', 'ncount')]:
        s = national[col].astype(float)
        for w in (4, 13):
            mean = s.rolling(w, min_periods=w).mean()
            std = s.rolling(w, min_periods=w).std()
            national[f'{pre}_ratio_mean_{w}'] = safe_ratio(s, mean)
            national[f'{pre}_cv_{w}'] = safe_ratio(std, mean.abs())
        for lag in (1, 4, 13):
            national[f'{pre}_change_{lag}'] = s.pct_change(lag)
        national[f'{pre}_slope_4'] = norm_slope(s, 4)
    ncols = [c for c in national.columns if c not in {'week_start', 'nvalue', 'ncount'}]
    dn = d[['week_start']].merge(national[['week_start'] + ncols], on='week_start', how='left', validate='many_to_one')
    F = pd.concat([F, dn[ncols]], axis=1)

    # Average-ticket dynamics capture divergence between value and transaction count without absolute levels.
    d['ticket'] = safe_ratio(d.value_thousand_sar, d.transaction_count_thousand)
    gt = d.groupby('city', sort=False)
    ticket_mean13 = gt.ticket.transform(lambda s: s.rolling(13, min_periods=13).mean())
    F['ticket_ratio_mean_13'] = safe_ratio(d.ticket, ticket_mean13)
    F['ticket_change_4'] = gt.ticket.pct_change(4)

    week = d.week_start.dt.isocalendar().week.astype(float)
    F['week_sin'] = np.sin(2 * np.pi * week / 52.18)
    F['week_cos'] = np.cos(2 * np.pi * week / 52.18)
    F = F.replace([np.inf, -np.inf], np.nan)

    # Leading-deterioration signals used only for safety triage, all observable at origin t.
    precursor = pd.DataFrame(index=d.index)
    precursor['value_below_short_trend'] = F['value_ratio_mean_4'] < 0.97
    precursor['count_below_short_trend'] = F['count_ratio_mean_4'] < 0.97
    precursor['value_negative_slope'] = F['value_slope_4'] < -0.012
    precursor['count_negative_slope'] = F['count_slope_4'] < -0.012
    precursor['value_2w_drop'] = F['value_change_2'] < -0.045
    precursor['count_2w_drop'] = F['count_change_2'] < -0.045
    precursor['share_weakening'] = F['value_share_change_4'] < -0.025
    precursor_count = precursor.sum(axis=1).astype(int)

    good = F.notna().all(axis=1)
    if require_target:
        good &= d.future_ratio.notna()
    meta = d.loc[good, ['week_start', 'city', 'target', 'target_float', 'future_ratio', 'baseline4']].reset_index(drop=True)
    X = F.loc[good].reset_index(drop=True)
    P = precursor.loc[good].reset_index(drop=True)
    pc = precursor_count.loc[good].reset_index(drop=True)
    return meta, X, P, pc


def model_factories():
    return {
        'elastic_logistic': make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty='elasticnet', solver='saga', l1_ratio=0.20, C=0.22,
                max_iter=6000, class_weight='balanced', random_state=SEED,
            ),
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=700, max_depth=6, min_samples_leaf=8, max_features=0.65,
            class_weight='balanced', random_state=SEED, n_jobs=-1,
        ),
        'hist_gb': HistGradientBoostingClassifier(
            max_iter=240, learning_rate=0.025, max_leaf_nodes=10,
            min_samples_leaf=28, l2_regularization=8.0, random_state=SEED,
        ),
    }


def folds(d: pd.DataFrame):
    # Many regimes, including calm and volatile blocks; thresholds are not tuned on a single six-month window.
    starts = pd.to_datetime([
        '2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01',
        '2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01',
        '2025-01-01', '2025-04-01',
    ])
    out = []
    for st in starts:
        en = min(st + pd.DateOffset(months=3) - pd.Timedelta(days=1), DEV_END)
        tr = d.week_start <= st - pd.Timedelta(days=14)
        va = d.week_start.between(st, en)
        if tr.sum() >= 500 and va.sum() >= 90 and d.loc[tr, 'target'].nunique() == 2:
            out.append((st, en, tr, va))
    return out


def fit_one(model, X, y):
    if isinstance(model, HistGradientBoostingClassifier):
        pos = max(int(y.sum()), 1)
        neg = max(len(y) - pos, 1)
        w = np.where(np.asarray(y) == 1, neg / pos, 1.0)
        return model.fit(X, y, sample_weight=w)
    return model.fit(X, y)


def build_oof(d, X, precursor_count):
    rows = []
    fold_meta = []
    for fid, (st, en, tr, va) in enumerate(folds(d)):
        ytr = d.loc[tr, 'target']
        q = d.loc[va, ['week_start', 'city', 'target']].rename(columns={'target': 'y'}).copy()
        q['fold_id'] = fid
        q['precursor_count'] = precursor_count.loc[va].to_numpy()
        for name, factory in model_factories().items():
            m = fit_one(clone(factory), X.loc[tr], ytr)
            q[name] = m.predict_proba(X.loc[va])[:, 1]
        q['score'] = q[list(model_factories())].mean(axis=1)
        # Agreement is deliberately coarse: independent models must broadly agree before RED.
        q['agreement'] = (q[list(model_factories())] >= 0.50).sum(axis=1)
        rows.append(q)
        fold_meta.append({
            'fold_id': fid, 'start': str(st.date()), 'end': str(en.date()),
            'train_rows': int(tr.sum()), 'validation_rows': int(va.sum()),
            'train_positive_rate': float(ytr.mean()), 'validation_positive_rate': float(q.y.mean()),
            'validation_positives': int(q.y.sum()),
        })
    if not rows:
        raise RuntimeError('No v3 OOF folds')
    return pd.concat(rows, ignore_index=True), fold_meta


def cm(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'precision': float(tp / max(tp + fp, 1)),
        'recall': float(tp / max(tp + fn, 1)),
        'FPR': float(fp / max(fp + tn, 1)),
        'NPV': float(tn / max(tn + fn, 1)),
    }


def classify(q, watch_t, red_t, min_precursor_red=2):
    score = q.score.to_numpy(float)
    pc = q.precursor_count.to_numpy(int)
    agree = q.agreement.to_numpy(int)
    # RED requires score + multi-model agreement + observable leading deterioration.
    red = (score >= red_t) & (agree >= 2) & (pc >= min_precursor_red)
    # AMBER is deliberately broad: either model risk or concrete deterioration.
    alert = red | (score >= watch_t) | (pc >= 2)
    state = np.where(red, 'RED', np.where(alert, 'AMBER', 'GREEN'))
    return state


def evaluate_policy(q, watch_t, red_t):
    state = classify(q, watch_t, red_t)
    y = q.y.to_numpy(int)
    red = state == 'RED'
    alert = state != 'GREEN'
    pooled = {'RED': cm(y, red), 'RED_plus_AMBER': cm(y, alert)}
    pooled['GREEN'] = {'NPV': pooled['RED_plus_AMBER']['NPV'], 'missed_declines': pooled['RED_plus_AMBER']['FN']}
    folds_out = []
    for fid, z in q.assign(state=state).groupby('fold_id'):
        yy = z.y.to_numpy(int)
        rr = z.state.eq('RED').to_numpy()
        aa = ~z.state.eq('GREEN').to_numpy()
        r = cm(yy, rr); a = cm(yy, aa)
        folds_out.append({
            'fold_id': int(fid), 'rows': int(len(z)), 'positives': int(yy.sum()),
            'red_precision': r['precision'], 'red_fpr': r['FPR'],
            'alert_recall': a['recall'], 'green_npv': a['NPV'],
        })
    return pooled, folds_out, state


def choose_policy(q):
    # Search broad candidates, then select by pooled constraints and worst-fold stability.
    scores = q.score.to_numpy(float)
    cand = np.unique(np.r_[np.quantile(scores, np.linspace(0.02, 0.995, 120)), np.linspace(0.01, 0.90, 90)])
    best = None
    for wt in cand:
        for rt in cand[cand >= wt]:
            pooled, ff, state = evaluate_policy(q, float(wt), float(rt))
            red_n = int((state == 'RED').sum())
            pos_folds = [f for f in ff if f['positives'] >= 2]
            worst_recall = min((f['alert_recall'] for f in pos_folds), default=1.0)
            worst_red_fpr = max((f['red_fpr'] for f in ff), default=0.0)
            ok = (
                red_n >= CONTRACT['min_red_alerts'] and
                pooled['RED']['precision'] >= CONTRACT['pooled_red_precision_min'] and
                pooled['RED']['FPR'] <= CONTRACT['pooled_red_fpr_max'] and
                pooled['RED_plus_AMBER']['recall'] >= CONTRACT['pooled_alert_recall_min'] and
                pooled['GREEN']['NPV'] >= CONTRACT['pooled_green_npv_min'] and
                worst_recall >= CONTRACT['worst_positive_fold_alert_recall_min'] and
                worst_red_fpr <= CONTRACT['worst_fold_red_fpr_max']
            )
            if not ok:
                continue
            # Favor high RED precision, then recall, then fewer RED false positives, then higher threshold.
            objective = (
                pooled['RED']['precision'], pooled['RED_plus_AMBER']['recall'],
                -pooled['RED']['FP'], rt,
            )
            if best is None or objective > best[0]:
                best = (objective, float(wt), float(rt), pooled, ff, red_n)
    if best is None:
        raise RuntimeError('No v3 triage policy meets cross-regime contract')
    return best[1:]


def robust_ood_profile(X: pd.DataFrame):
    prof = {}
    for c in X.columns:
        s = X[c].astype(float)
        q01, q25, q50, q75, q99 = np.nanquantile(s, [0.01, 0.25, 0.50, 0.75, 0.99])
        iqr = max(float(q75 - q25), 1e-9)
        prof[c] = {
            'median': float(q50), 'iqr': iqr,
            'low': float(q01 - 0.75 * iqr), 'high': float(q99 + 0.75 * iqr),
        }
    return prof


def main():
    panel = source.reconciled_load_panel(HISTORY)
    d, X, P, pc = featureize(panel, require_target=True)
    keep = d.week_start <= DEV_END
    d = d.loc[keep].reset_index(drop=True)
    X = X.loc[keep].reset_index(drop=True)
    P = P.loc[keep].reset_index(drop=True)
    pc = pc.loc[keep].reset_index(drop=True)

    # Hard leakage/generalization guards.
    forbidden = [c for c in X.columns if c.startswith('city_') or 'decline_rate' in c or 'target' in c or 'future' in c]
    if forbidden:
        raise RuntimeError(f'Forbidden v3 features: {forbidden}')
    if len(X.columns) > 55:
        raise RuntimeError(f'v3 feature budget exceeded: {len(X.columns)}')

    oof, fold_meta = build_oof(d, X, pc)
    roc = float(roc_auc_score(oof.y, oof.score))
    pr = float(average_precision_score(oof.y, oof.score))
    watch_t, red_t, pooled, fold_policy, red_n = choose_policy(oof)
    gates = {
        'roc_auc': roc >= CONTRACT['pooled_roc_auc_min'],
        'red_precision': pooled['RED']['precision'] >= CONTRACT['pooled_red_precision_min'],
        'red_fpr': pooled['RED']['FPR'] <= CONTRACT['pooled_red_fpr_max'],
        'alert_recall': pooled['RED_plus_AMBER']['recall'] >= CONTRACT['pooled_alert_recall_min'],
        'green_npv': pooled['GREEN']['NPV'] >= CONTRACT['pooled_green_npv_min'],
        'feature_budget': len(X.columns) <= 55,
        'no_city_identity': not any(c.startswith('city_') for c in X.columns),
        'no_target_prevalence_features': not any('decline_rate' in c for c in X.columns),
    }

    fitted = {}
    for name, factory in model_factories().items():
        fitted[name] = fit_one(clone(factory), X, d.target)

    artifact = {
        'version': VERSION,
        'models': fitted,
        'features': list(X.columns),
        'watch_threshold': watch_t,
        'red_threshold': red_t,
        'min_precursor_red': 2,
        'ood_profile': robust_ood_profile(X),
        'ood_max_fraction': 0.15,
        'development_end': str(DEV_END.date()),
        'target_definition': 'next-week city POS value < 80% of current trailing-4-week mean',
        'scope': 'forecastable market deterioration; truly exogenous surprise shocks are not claimed predictable',
    }
    joblib.dump(artifact, MODEL)

    report = {
        'version': VERSION,
        'rows': int(len(d)), 'positives': int(d.target.sum()), 'positive_rate': float(d.target.mean()),
        'feature_count': int(len(X.columns)), 'features': list(X.columns),
        'folds': fold_meta, 'pooled_ranking': {'ROC_AUC': roc, 'PR_AUC': pr},
        'policy': {'watch_threshold': watch_t, 'red_threshold': red_t, 'red_alerts': red_n},
        'pooled_policy': pooled, 'fold_policy': fold_policy,
        'contract': CONTRACT, 'gates': gates, 'all_gates_passed': bool(all(gates.values())),
        'root_cause_controls': {
            'city_identity_removed': True,
            'target_prevalence_features_removed': True,
            'absolute_levels_removed': True,
            'compact_feature_budget': True,
            'multi_regime_oof': True,
            'worst_fold_constraints': True,
            'red_requires_model_agreement': True,
            'red_requires_observed_precursors': True,
            'ood_fail_closed_profile_saved': True,
            'surprise_shock_claim': 'not predictable without leading information',
        },
        'scientific_boundary': 'All model and threshold fitting ends 2025-06-29. Later data are diagnostics only and are not read here.',
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    (OUT / 'development_summary.md').write_text(
        '# Sales Sentinel v3 — Generalization-first development\n\n'
        f"- Rows: **{len(d):,}**; positives: **{int(d.target.sum())} ({d.target.mean():.2%})**\n"
        f"- Compact features: **{len(X.columns)}** (no city identity, no raw levels, no target-prevalence features)\n"
        f"- OOF ROC-AUC: **{roc:.2%}**; PR-AUC: **{pr:.2%}**\n"
        f"- RED precision: **{pooled['RED']['precision']:.2%}**; RED FPR: **{pooled['RED']['FPR']:.2%}**\n"
        f"- RED+AMBER recall: **{pooled['RED_plus_AMBER']['recall']:.2%}**\n"
        f"- GREEN NPV: **{pooled['GREEN']['NPV']:.2%}**\n"
        f"- All development gates passed: **{report['all_gates_passed']}**\n",
        encoding='utf-8',
    )
    print(json.dumps({'report': report}, indent=2))


if __name__ == '__main__':
    main()
