from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_sama_city_risk_v2_1 as source
import train_sama_city_risk_v2_2 as v22
import production_city_risk_engine_v2_4 as engine

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / 'data' / 'sama_pos' / 'sama_city_weekly_value_count_2020_2025.csv'
EXTENDED = ROOT / 'data' / 'sama_pos' / 'sama_city_weekly_value_count_2020_2026_extended.csv'
BASE = ROOT / 'models' / 'sama_city_v2_2' / 'city_market_risk_v2_2.joblib'
POLICY = ROOT / 'models' / 'sama_city_v2_4' / 'prior_shift_policy_v2_4.joblib'
DEV_REPORT = ROOT / 'reports' / 'sama_city_v2_2' / 'development_report.json'
POLICY_REPORT = ROOT / 'reports' / 'sama_city_v2_4' / 'policy_development_report.json'
STRESS_REPORT = ROOT / 'reports' / 'sama_city_v2_4' / 'post_diagnosis_prequential_stress.json'
OUT_DIR = ROOT / 'reports' / 'final_model_audit_v2_4'
FIX_DIR = ROOT / 'tests' / 'fixtures' / 'sales_sentinel_v2_4'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIX_DIR.mkdir(parents=True, exist_ok=True)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / den
    return max(0.0, center - half), min(1.0, center + half)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def future_perturbation_invariance(panel: pd.DataFrame, base_features: list[str]) -> dict:
    origins = [pd.Timestamp('2024-06-30'), pd.Timestamp('2025-01-05'), pd.Timestamp('2025-06-29')]
    checks = []
    rng = np.random.default_rng(20260814)
    for requested in origins:
        available = sorted(pd.to_datetime(panel.week_start.unique()))
        origin = max([w for w in available if w <= requested])
        truncated = panel[panel.week_start <= origin].copy()
        mutated = panel.copy()
        fut = mutated.week_start > origin
        # Deliberately destroy every future value/count. Causal features at the origin must not change.
        mutated.loc[fut, 'value_thousand_sar'] *= rng.uniform(0.05, 8.0, fut.sum())
        mutated.loc[fut, 'transaction_count_thousand'] *= rng.uniform(0.05, 8.0, fut.sum())
        _, ft = engine.build_inference_features(truncated)
        dm, fm = engine.build_inference_features(mutated)
        dt, _ = engine.build_inference_features(truncated)
        mt = dt.week_start.eq(origin)
        mm = dm.week_start.eq(origin)
        a = ft.loc[mt, base_features].sort_index(axis=1).reset_index(drop=True)
        b = fm.loc[mm, base_features].sort_index(axis=1).reset_index(drop=True)
        same = a.shape == b.shape and np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-10, atol=1e-12, equal_nan=True)
        checks.append({'origin': str(origin.date()), 'rows': int(len(a)), 'features_identical_after_destroying_future': bool(same)})
    return {'cases': checks, 'passed': all(x['features_identical_after_destroying_future'] for x in checks)}


def fold_overlap_audit(d: pd.DataFrame) -> dict:
    details = []
    ok = True
    for st, en, tr, va in v22.folds(d):
        tri = set(d.index[tr].tolist())
        vai = set(d.index[va].tolist())
        overlap = len(tri & vai)
        train_max = pd.Timestamp(d.loc[tr, 'week_start'].max())
        val_min = pd.Timestamp(d.loc[va, 'week_start'].min())
        gap_days = int((val_min - train_max).days)
        this_ok = overlap == 0 and gap_days >= 14
        ok &= this_ok
        details.append({'validation_start': str(st.date()), 'validation_end': str(en.date()), 'index_overlap': overlap, 'train_max_origin': str(train_max.date()), 'validation_min_origin': str(val_min.date()), 'purge_gap_days': gap_days, 'passed': this_ok})
    return {'folds': details, 'passed': bool(ok)}


def permutation_sanity(d: pd.DataFrame, X: pd.DataFrame, n_perm: int = 8) -> dict:
    rng = np.random.default_rng(20260814)
    aucs = []
    # A deliberately simple model keeps this sanity test fast and interpretable.
    for _ in range(n_perm):
        yp = pd.Series(rng.permutation(d.target.to_numpy()), index=d.index)
        pred = []
        truth = []
        for st, en, tr, va in v22.folds(d):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=.35, max_iter=2000, class_weight='balanced', random_state=42))
            model.fit(X.loc[tr], yp.loc[tr])
            pred.extend(model.predict_proba(X.loc[va])[:, 1].tolist())
            truth.extend(d.loc[va, 'target'].tolist())
        aucs.append(float(roc_auc_score(truth, pred)))
    return {'n_permutations': n_perm, 'mean_auc_against_true_labels': float(np.mean(aucs)), 'max_auc_against_true_labels': float(np.max(aucs)), 'all_aucs': aucs, 'passed': bool(np.mean(aucs) < 0.58 and np.max(aucs) < 0.65)}


def build_app_fixture(extended: pd.DataFrame) -> dict:
    weeks = sorted(pd.to_datetime(extended.week_start.unique()))
    keep_weeks = weeks[-130:]
    fixture = extended[extended.week_start.isin(keep_weeks)][['week_start', 'week_end', 'city', 'value_thousand_sar', 'transaction_count_thousand']].copy()
    fixture = fixture.sort_values(['city', 'week_start']).reset_index(drop=True)
    input_path = FIX_DIR / 'app_test_input_sama_real_130_weeks.csv'
    fixture.to_csv(input_path, index=False, date_format='%Y-%m-%d')
    prediction = engine.predict_latest(fixture, BASE, POLICY)
    expected_path = FIX_DIR / 'app_test_expected_predictions.json'
    expected_path.write_text(json.dumps(prediction, indent=2, default=str), encoding='utf-8')
    manifest = {
        'purpose': 'Functional/regression test input for the Sales Sentinel v2.4 production engine. Not an independent scientific holdout.',
        'source': 'Official SAMA city-total weekly POS panel already versioned in this repository.',
        'input_has_target_or_future_label': False,
        'rows': int(len(fixture)),
        'cities': int(fixture.city.nunique()),
        'weeks': int(fixture.week_start.nunique()),
        'period_start': str(pd.Timestamp(fixture.week_start.min()).date()),
        'period_end': str(pd.Timestamp(fixture.week_start.max()).date()),
        'columns': fixture.columns.tolist(),
        'input_sha256': sha256(input_path),
        'expected_sha256': sha256(expected_path),
        'expected_engine_status': prediction.get('status'),
        'expected_prediction_rows': len(prediction.get('predictions', [])),
    }
    manifest_path = FIX_DIR / 'app_test_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return {'input_path': str(input_path.relative_to(ROOT)), 'expected_path': str(expected_path.relative_to(ROOT)), 'manifest_path': str(manifest_path.relative_to(ROOT)), **manifest}


def main():
    base = joblib.load(BASE)
    dev = json.loads(DEV_REPORT.read_text(encoding='utf-8'))
    pol = json.loads(POLICY_REPORT.read_text(encoding='utf-8'))
    stress = json.loads(STRESS_REPORT.read_text(encoding='utf-8'))

    raw_hist = source.reconciled_load_panel(HISTORY)
    raw_hist['week_start'] = pd.to_datetime(raw_hist.week_start)
    d, X = v22.featureize(raw_hist)
    keep = d.week_start <= v22.DEV_END
    d = d.loc[keep].reset_index(drop=True)
    X = X.loc[keep].reset_index(drop=True)
    extended = pd.read_csv(EXTENDED, parse_dates=['week_start', 'week_end'])

    forbidden_tokens = ('future', 'actual_next', 'target')
    forbidden_features = [c for c in base['features'] if any(t in c.lower() for t in forbidden_tokens)]
    source_dup = int(raw_hist.duplicated(['week_start', 'city']).sum())
    feature_dup = int(pd.DataFrame(X).duplicated().sum())
    exact_schema = list(X.columns) == list(base['features'])
    fold_audit = fold_overlap_audit(d)
    causal = future_perturbation_invariance(raw_hist.copy(), list(base['features']))
    perm = permutation_sanity(d, X)

    hred = pol['historical_RED']
    hgreen_tn = pol['historical_RED_plus_AMBER']['TN']
    hgreen_fn = pol['historical_RED_plus_AMBER']['FN']
    sred = stress['metrics']['RED']
    sgreen = stress['metrics']['GREEN']
    hred_ci = wilson(hred['TP'], hred['TP'] + hred['FP'])
    hgreen_ci = wilson(hgreen_tn, hgreen_tn + hgreen_fn)
    sred_ci = wilson(sred['TP'], sred['TP'] + sred['FP'])
    sgreen_ci = wilson(sgreen['TN'], sgreen['TN'] + sgreen['FN'])

    statistical = {
        'historical_red_precision': {'point': hred['precision'], 'n_alerts': hred['TP'] + hred['FP'], 'wilson95': hred_ci},
        'historical_green_npv': {'point': pol['historical_RED_plus_AMBER']['NPV'], 'n_green': hgreen_tn + hgreen_fn, 'wilson95': hgreen_ci},
        'stress_red_precision': {'point': sred['precision'], 'n_alerts': sred['TP'] + sred['FP'], 'wilson95': sred_ci},
        'stress_green_npv': {'point': sgreen['NPV'], 'n_green': sgreen['TN'] + sgreen['FN'], 'wilson95': sgreen_ci},
        'historical_red_precision_ci_lower_meets_70pct': hred_ci[0] >= .70,
        'historical_green_npv_ci_lower_meets_99pct': hgreen_ci[0] >= .99,
        'stress_red_precision_meets_contract': sred['precision'] >= .70,
        'stress_red_fpr_meets_contract': sred['FPR'] <= .0075,
        'stress_alert_recall_meets_contract': stress['metrics']['RED_plus_AMBER']['recall'] >= .92,
        'stress_green_npv_meets_contract': sgreen['NPV'] >= .99,
    }

    fixture = build_app_fixture(extended)

    leakage_gates = {
        'no_forbidden_target_future_feature_names': len(forbidden_features) == 0,
        'source_week_city_unique': source_dup == 0,
        'frozen_feature_schema_exact': exact_schema,
        'chronological_folds_have_no_overlap_and_purge': fold_audit['passed'],
        'future_perturbation_does_not_change_origin_features': causal['passed'],
        'permuted_label_sanity_collapses_signal': perm['passed'],
        'trainer_declares_no_fresh_2025_2026_use': dev['leakage_controls']['fresh_2025_2026_used'] is False,
        'policy_declares_no_fresh_parameter_fitting': pol['controls']['no_fresh_2025_2026_labels_used'] is True,
    }
    generalization_gates = {
        'stress_red_precision': statistical['stress_red_precision_meets_contract'],
        'stress_red_fpr': statistical['stress_red_fpr_meets_contract'],
        'stress_alert_recall': statistical['stress_alert_recall_meets_contract'],
        'stress_green_npv': statistical['stress_green_npv_meets_contract'],
        'stress_roc_auc': stress['ranking']['ROC_AUC'] >= .87,
        'stress_pr_auc': stress['ranking']['PR_AUC'] >= .45,
        'independent_post_design_holdout_exists': False,
        'historical_green_npv_99pct_supported_by_95pct_ci': statistical['historical_green_npv_ci_lower_meets_99pct'],
    }

    final_approval = bool(all(leakage_gates.values()) and all(generalization_gates.values()))
    report = {
        'version': 'SALES-SENTINEL-V2.4-FINAL-AUDIT-1',
        'decision': 'APPROVED_FINAL' if final_approval else 'NOT_FINAL_APPROVED',
        'reason': 'Leakage controls are audited separately from generalization. A high development score alone is not sufficient for final approval.',
        'development_point_metrics': {'red_precision': hred['precision'], 'red_fpr': hred['FPR'], 'alert_recall': pol['historical_RED_plus_AMBER']['recall'], 'green_npv': pol['historical_RED_plus_AMBER']['NPV']},
        'stress_point_metrics': {'red_precision': sred['precision'], 'red_fpr': sred['FPR'], 'alert_recall': stress['metrics']['RED_plus_AMBER']['recall'], 'green_npv': sgreen['NPV'], 'roc_auc': stress['ranking']['ROC_AUC'], 'pr_auc': stress['ranking']['PR_AUC']},
        'leakage_audit': {'forbidden_features': forbidden_features, 'source_duplicate_week_city_rows': source_dup, 'exact_duplicate_feature_rows': feature_dup, 'fold_audit': fold_audit, 'future_perturbation_invariance': causal, 'permutation_sanity': perm, 'gates': leakage_gates, 'all_passed': bool(all(leakage_gates.values()))},
        'statistical_reliability': statistical,
        'generalization_gates': generalization_gates,
        'all_generalization_gates_passed': bool(all(generalization_gates.values())),
        'final_approval': final_approval,
        'app_test_fixture': fixture,
        'required_next_evidence': 'Prospective outcomes published after the v2.4 architecture/policy freeze must be evaluated without changing model weights, features, policy parameters, or thresholds.',
    }
    report_path = OUT_DIR / 'final_audit.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    md = f"""# Sales Sentinel v2.4 — Final Model Audit\n\n- Final decision: **{report['decision']}**\n- Leakage audit: **{'PASS' if report['leakage_audit']['all_passed'] else 'FAIL'}**\n- Generalization gates: **{'PASS' if report['all_generalization_gates_passed'] else 'FAIL'}**\n- Development RED precision: **{hred['precision']:.2%}** ({hred['TP']} TP / {hred['FP']} FP), Wilson 95% lower **{hred_ci[0]:.2%}**\n- Development GREEN NPV: **{pol['historical_RED_plus_AMBER']['NPV']:.2%}**, Wilson 95% lower **{hgreen_ci[0]:.2%}**\n- Stress RED precision: **{sred['precision']:.2%}** ({sred['TP']} TP / {sred['FP']} FP)\n- Stress RED FPR: **{sred['FPR']:.2%}**\n- Stress RED+AMBER recall: **{stress['metrics']['RED_plus_AMBER']['recall']:.2%}**\n- Stress GREEN NPV: **{sgreen['NPV']:.2%}**\n- Stress ROC-AUC: **{stress['ranking']['ROC_AUC']:.2%}**\n- Stress PR-AUC: **{stress['ranking']['PR_AUC']:.2%}**\n- App test input: `{fixture['input_path']}`\n- Expected output: `{fixture['expected_path']}`\n\nThe model is not finally approved until a post-freeze prospective validation passes the frozen production contract.\n"""
    (OUT_DIR / 'final_audit.md').write_text(md, encoding='utf-8')
    print(json.dumps({'decision': report['decision'], 'leakage_all_passed': report['leakage_audit']['all_passed'], 'generalization_all_passed': report['all_generalization_gates_passed'], 'fixture': fixture}, indent=2))
    # CI should succeed so evidence is persisted even when scientific approval is false.


if __name__ == '__main__':
    main()
