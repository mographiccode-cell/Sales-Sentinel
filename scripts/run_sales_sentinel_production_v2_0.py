from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

import train_sales_sentinel_production_v2_0 as p
import train_sama_sector_decline_v1_9 as v19


def main():
    d, X, _ = v19.prepare()
    d, X = p.add_reliability_features(d, X)

    # Preserve exact row alignment while applying the pre-declared development cutoff.
    keep = d.origin_week_start <= p.DEVELOPMENT_END
    d = d.loc[keep].reset_index(drop=True)
    X = X.loc[keep].reset_index(drop=True)
    if len(X) != len(d) or not X.index.equals(d.index):
        raise RuntimeError('Feature/label alignment failed after development cutoff')

    oof, folds = p.oof_scores(d, X)
    selection = oof[oof.origin_week_start <= p.SELECTION_END].copy()
    policy = oof[oof.origin_week_start.between(p.POLICY_START, p.POLICY_END)].copy()
    if len(selection) < 300 or len(policy) < 200 or selection.y.sum() < 20 or policy.y.sum() < 15:
        raise RuntimeError(
            f'Insufficient separated development windows: selection={len(selection)}/{selection.y.sum()}+, '
            f'policy={len(policy)}/{policy.y.sum()}+'
        )

    candidate_names = [c for c in oof.columns if c not in {'idx','origin_week_start','y'}]
    candidate_metrics = {c: p.ranking_metrics(selection.y, selection[c]) for c in candidate_names}
    selected = max(candidate_names, key=lambda c: (candidate_metrics[c]['PR_AUC'], candidate_metrics[c]['ROC_AUC']))

    # Calibration is fitted only on 2024 OOF; policy thresholds are chosen only on 2025-H1 OOF.
    calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=p.SEED).fit(selection[[selected]], selection.y)
    selection_p = calibrator.predict_proba(selection[[selected]])[:, 1]
    policy_p = calibrator.predict_proba(policy[[selected]])[:, 1]
    policy_contract = p.choose_policy(policy.y, policy_p)
    watch_t = policy_contract['watch']['threshold']
    red_t = policy_contract['red']['threshold']

    dev_triage = p.triage_metrics(policy.y, policy_p, watch_t, red_t)
    dev_rank = p.ranking_metrics(policy.y, policy_p)
    dev_cal = p.calibration_metrics(policy.y, policy_p)
    dev_gates = {
        'red_precision': dev_triage['RED']['precision'] >= p.ACCEPTANCE['red_precision_min'],
        'red_false_positive_rate': dev_triage['RED']['false_positive_rate'] <= p.ACCEPTANCE['red_false_positive_rate_max'],
        'alert_recall': dev_triage['RED_plus_AMBER']['recall'] >= p.ACCEPTANCE['alert_recall_min'],
        'green_npv': dev_triage['GREEN']['NPV'] >= p.ACCEPTANCE['green_npv_min'],
        'roc_auc': dev_rank['ROC_AUC'] >= p.ACCEPTANCE['roc_auc_min'],
        'pr_auc': dev_rank['PR_AUC'] >= p.ACCEPTANCE['pr_auc_min'],
    }

    # Final development fit. The fresh 2025-2026 holdout is not read anywhere in this script.
    y = d.target
    pos = int(y.sum()); neg = len(y) - pos
    final_models = {}
    available = p.models_for(neg / max(pos, 1))
    if selected in available:
        final_models[selected] = clone(available[selected]).fit(X, y)
    elif selected == 'RobustMedian':
        for name in ('LogisticRegression','ExtraTrees','XGBoost'):
            final_models[name] = clone(available[name]).fit(X, y)

    artifact = {
        'version': p.VERSION,
        'selected_score': selected,
        'models': final_models,
        'calibrator': calibrator,
        'features': list(X.columns),
        'watch_threshold': watch_t,
        'red_threshold': red_t,
        'decline_threshold': p.DECLINE,
        'acceptance_contract': p.ACCEPTANCE,
        'development_end': str(p.DEVELOPMENT_END.date()),
        'inference_policy': {
            'GREEN': f'p < {watch_t:.6f}',
            'AMBER': f'{watch_t:.6f} <= p < {red_t:.6f}',
            'RED': f'p >= {red_t:.6f}',
        },
    }
    joblib.dump(artifact, p.MODEL)

    report = {
        'version': p.VERSION,
        'scientific_boundary': 'All model choice/calibration/policy thresholds use SAMA data ending 2025-07-06. Newly acquired 2025-2026 PDFs are not read by development training.',
        'target': 'next official SAMA sector week POS value is >=20% below trailing four completed official weeks mean',
        'development_rows': int(len(d)),
        'development_positive_rate': float(d.target.mean()),
        'folds': folds,
        'separation': {
            'candidate_selection_end': str(p.SELECTION_END.date()),
            'policy_threshold_window': f'{p.POLICY_START.date()}..{p.POLICY_END.date()}',
            'fresh_holdout_not_used': True,
        },
        'candidate_selection_metrics_2024': candidate_metrics,
        'selected_score': selected,
        'calibration_selection_window': p.calibration_metrics(selection.y, selection_p),
        'policy_contract': policy_contract,
        'policy_window_ranking': dev_rank,
        'policy_window_calibration': dev_cal,
        'policy_window_triage': dev_triage,
        'acceptance_contract': p.ACCEPTANCE,
        'development_gates': dev_gates,
        'development_all_gates_passed': bool(all(dev_gates.values())),
        'leakage_controls': {
            'expanding_time_folds': True,
            'one_week_purge_before_each_validation_fold': True,
            'future_actual_SAMA_as_feature': False,
            'forecast_residual_features_shifted_before_use': True,
            'candidate_selection_and_policy_threshold_windows_separated': True,
            'fresh_2025_2026_holdout_used_for_training_or_thresholds': False,
            'shuffle': False,
        },
    }
    p.REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    p.SUMMARY.write_text(f'''# Sales Sentinel Production v2.0 — Frozen Development\n\n- Selected score: **{selected}**\n- Development rows: **{len(d):,}**\n- Development decline rate: **{d.target.mean():.2%}**\n- Candidate selection: **through 2024-12-31**\n- Policy thresholds: **2025-01-01 through 2025-06-29 only**\n- Fresh 2025-2026 holdout used: **No**\n- RED threshold: **{red_t:.4f}**\n- WATCH threshold: **{watch_t:.4f}**\n- RED precision: **{dev_triage['RED']['precision']:.2%}**\n- RED false-positive rate: **{dev_triage['RED']['false_positive_rate']:.2%}**\n- RED+AMBER recall: **{dev_triage['RED_plus_AMBER']['recall']:.2%}**\n- GREEN NPV: **{dev_triage['GREEN']['NPV']:.2%}**\n- PR-AUC: **{dev_rank['PR_AUC']:.2%}**\n- ROC-AUC: **{dev_rank['ROC_AUC']:.2%}**\n- Brier: **{dev_cal['Brier']:.4f}**\n- Development contract passed: **{report['development_all_gates_passed']}**\n''', encoding='utf-8')
    print(json.dumps({
        'selected': selected,
        'rows': len(d),
        'positive_rate': float(d.target.mean()),
        'thresholds': {'watch': watch_t, 'red': red_t},
        'triage': dev_triage,
        'ranking': dev_rank,
        'calibration': dev_cal,
        'gates': dev_gates,
        'all_gates': report['development_all_gates_passed'],
    }, indent=2))


if __name__ == '__main__':
    main()
