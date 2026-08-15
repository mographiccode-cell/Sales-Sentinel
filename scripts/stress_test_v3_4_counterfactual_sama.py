from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from production_city_risk_engine_v3_4 import MODEL, predict_latest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'sama_pos' / 'sama_city_weekly_value_count_2020_2026_extended.csv'
OUT = ROOT / 'reports' / 'sama_city_v3_4' / 'counterfactual_stress'
OUT.mkdir(parents=True, exist_ok=True)
EXPECTED_MODEL_VERSION = 'SAMA-CITY-RISK-3.4-STRUCTURAL-HYBRID'
CITIES = ['ABHA','BURAIDAH','DAMMAM','HAIL','JEDDAH','KHOBAR','MADINA','MAKKAH','OTHER','RIYADH','TABOUK']

# Three NEW patterns, fixed before scoring and different from the v3.3 stress test.
PATTERNS = {
    'balanced_gradual': {
        'value': {-2: 0.97, -1: 0.92, 0: 0.86},
        'count': {-2: 0.97, -1: 0.92, 0: 0.86},
        'event_ratio': 0.70,
    },
    'value_led': {
        'value': {-2: 0.96, -1: 0.90, 0: 0.82},
        'count': {-2: 0.99, -1: 0.95, 0: 0.90},
        'event_ratio': 0.72,
    },
    'count_led': {
        'value': {-2: 0.99, -1: 0.95, 0: 0.90},
        'count': {-2: 0.96, -1: 0.90, 0: 0.82},
        'event_ratio': 0.68,
    },
}


def prediction_map(result: dict) -> dict[str, dict]:
    if result.get('status') != 'OK':
        raise RuntimeError(json.dumps(result))
    if result.get('model_version') != EXPECTED_MODEL_VERSION:
        raise RuntimeError(f"Serving/model mismatch {result.get('model_version')}")
    return {str(x['city']): x for x in result['predictions']}


def inject(panel: pd.DataFrame, city: str, origin: pd.Timestamp, pattern_name: str):
    cfg = PATTERNS[pattern_name]
    d = panel.copy()
    weeks = sorted(pd.to_datetime(d.week_start.unique()))
    idx = weeks.index(pd.Timestamp(origin))
    if idx < 3 or idx + 1 >= len(weeks):
        raise RuntimeError('origin lacks required history/event week')
    event_week = pd.Timestamp(weeks[idx + 1])

    for offset in (-2, -1, 0):
        w = pd.Timestamp(weeks[idx + offset])
        m = d.week_start.eq(w) & d.city.eq(city)
        if int(m.sum()) != 1:
            raise RuntimeError(f'missing {city} {w.date()}')
        d.loc[m, 'value_thousand_sar'] *= cfg['value'][offset]
        d.loc[m, 'transaction_count_thousand'] *= cfg['count'][offset]

    # Define the unseen event by the target itself, after precursor modification.
    cr = d[d.city.eq(city)].sort_values('week_start').reset_index(drop=True)
    pos = int(cr.index[cr.week_start.eq(origin)][0])
    trailing4_value = float(cr.loc[pos-3:pos, 'value_thousand_sar'].mean())
    trailing4_count = float(cr.loc[pos-3:pos, 'transaction_count_thousand'].mean())
    event_mask = d.week_start.eq(event_week) & d.city.eq(city)
    d.loc[event_mask, 'value_thousand_sar'] = trailing4_value * float(cfg['event_ratio'])
    d.loc[event_mask, 'transaction_count_thousand'] = trailing4_count * float(cfg['event_ratio'])

    return d, {
        'city': city, 'origin': str(origin.date()), 'event_week': str(event_week.date()),
        'pattern': pattern_name, 'event_ratio': float(cfg['event_ratio']),
    }


def main():
    artifact = joblib.load(MODEL)
    if artifact.get('version') != EXPECTED_MODEL_VERSION:
        raise RuntimeError(f'wrong artifact {artifact.get("version")}')

    panel = pd.read_csv(DATA, parse_dates=['week_start','week_end']).sort_values(['week_start','city']).reset_index(drop=True)
    weeks = [pd.Timestamp(x) for x in sorted(panel.week_start.unique())]
    eligible = [w for w in weeks if pd.Timestamp('2025-08-17') <= w <= pd.Timestamp('2026-06-14')]
    if len(eligible) < 36:
        raise RuntimeError(f'not enough eligible weeks {len(eligible)}')

    # 33 deterministic origins selected only from calendar positions; no outcome labels used.
    pos = np.linspace(3, len(eligible)-3, 33, dtype=int)
    origins = [eligible[i] for i in pos]
    pattern_names = list(PATTERNS)
    scenarios = []
    for i in range(33):
        scenarios.append((CITIES[i % len(CITIES)], origins[i], pattern_names[i // len(CITIES)]))

    event_rows = []
    control_rows = []
    for sid, (city, origin, pattern) in enumerate(scenarios, start=1):
        base_hist = panel[panel.week_start <= origin][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']].copy()
        bmap = prediction_map(predict_latest(base_hist))

        injected, truth = inject(panel, city, origin, pattern)
        inj_hist = injected[injected.week_start <= origin][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']].copy()
        imap = prediction_map(predict_latest(inj_hist))

        b = bmap[city]; p = imap[city]
        event_rows.append({
            'scenario_id': sid, **truth,
            'baseline_state': b['state'], 'baseline_score': b['risk_score'],
            'injected_state': p['state'], 'injected_reason': p['reason'], 'injected_score': p['risk_score'],
            'structural_core_count': p.get('structural_core_count'), 'precursor_count': p.get('precursor_count'),
            'ood_fraction': p['ood_fraction'], 'score_lift': p['risk_score'] - b['risk_score'],
            'alerted': int(p['state'] in {'RED','AMBER'}), 'red': int(p['state'] == 'RED'),
            'ood_abstain': int(p['reason'] == 'OOD_ABSTAIN'),
        })

        for other in CITIES:
            if other == city:
                continue
            bp = bmap[other]; ip = imap[other]
            control_rows.append({
                'scenario_id': sid, 'origin': str(origin.date()), 'pattern': pattern,
                'injected_city': city, 'control_city': other,
                'baseline_state': bp['state'], 'injected_state': ip['state'],
                'baseline_score': bp['risk_score'], 'injected_score': ip['risk_score'],
                'new_red': int(bp['state'] != 'RED' and ip['state'] == 'RED'),
                'new_alert': int(bp['state'] == 'GREEN' and ip['state'] in {'RED','AMBER'}),
                'score_change': ip['risk_score'] - bp['risk_score'],
            })

    e = pd.DataFrame(event_rows)
    c = pd.DataFrame(control_rows)
    pattern_stats = {}
    for pat, z in e.groupby('pattern'):
        pattern_stats[pat] = {
            'n': int(len(z)), 'alerted': int(z.alerted.sum()),
            'recall': float(z.alerted.mean()), 'red': int(z.red.sum()),
            'median_score_lift': float(z.score_lift.median()),
            'structural_warnings': int(z.injected_reason.eq('STRUCTURAL_TREND_WARNING').sum()),
            'ood_abstentions': int(z.ood_abstain.sum()),
        }

    overall_recall = float(e.alerted.mean())
    control_new_red_rate = float(c.new_red.mean())
    control_new_alert_rate = float(c.new_alert.mean())
    max_control_score_change = float(c.score_change.abs().max())
    ood_rate = float(e.ood_abstain.mean())

    acceptance = {
        'overall_injected_alert_recall_ge_90pct': overall_recall >= 0.90,
        'each_pattern_recall_ge_80pct': all(x['recall'] >= 0.80 for x in pattern_stats.values()),
        'control_new_red_rate_le_1pct': control_new_red_rate <= 0.01,
        'control_new_alert_rate_le_5pct': control_new_alert_rate <= 0.05,
        'injected_ood_abstain_rate_le_30pct': ood_rate <= 0.30,
        'all_events_are_gt20pct_declines': bool((e.event_ratio < 0.80).all()),
        'serving_exact_v3_4_artifact': artifact.get('version') == EXPECTED_MODEL_VERSION,
    }

    report = {
        'version': 'SAMA-CITY-V3.4-MULTIPATTERN-COUNTERFACTUAL-1',
        'frozen_model': EXPECTED_MODEL_VERSION,
        'model_development_end': artifact.get('development_end'),
        'test_type': 'new multi-pattern counterfactual stress test on real SAMA-shaped covariates',
        'scenario_count': int(len(e)), 'control_rows': int(len(c)),
        'patterns': PATTERNS, 'pattern_results': pattern_stats,
        'overall': {
            'alerted': int(e.alerted.sum()), 'recall': overall_recall,
            'RED': int(e.red.sum()), 'AMBER': int(e.injected_state.eq('AMBER').sum()),
            'GREEN': int(e.injected_state.eq('GREEN').sum()),
            'median_score_lift': float(e.score_lift.median()),
            'structural_warnings': int(e.injected_reason.eq('STRUCTURAL_TREND_WARNING').sum()),
            'model_warnings': int(e.injected_reason.eq('MODEL_EARLY_WARNING').sum()),
            'ood_abstentions': int(e.ood_abstain.sum()), 'ood_rate': ood_rate,
        },
        'controls': {
            'new_red': int(c.new_red.sum()), 'new_red_rate': control_new_red_rate,
            'new_alert': int(c.new_alert.sum()), 'new_alert_rate': control_new_alert_rate,
            'max_abs_score_change': max_control_score_change,
        },
        'acceptance': acceptance, 'all_acceptance_passed': bool(all(acceptance.values())),
        'scientific_boundary': 'No model weight, feature, threshold, structural rule, or policy parameter is fit in this script. These three injection patterns were not used by v3.4 training.',
    }
    e.to_csv(OUT / 'injected_scenarios.csv', index=False)
    c.to_csv(OUT / 'unaffected_controls.csv', index=False)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (OUT / 'summary.md').write_text(
        '# v3.4 Multi-Pattern Counterfactual Stress\n\n'
        f'- Overall injected recall: **{overall_recall:.2%}** ({int(e.alerted.sum())}/{len(e)})\n'
        + ''.join(f'- {k}: **{v["recall"]:.2%}** ({v["alerted"]}/{v["n"]})\n' for k,v in pattern_stats.items())
        + f'- Control new RED: **{int(c.new_red.sum())}/{len(c)} ({control_new_red_rate:.2%})**\n'
        + f'- Control new alert: **{int(c.new_alert.sum())}/{len(c)} ({control_new_alert_rate:.2%})**\n'
        + f'- All gates: **{report["all_acceptance_passed"]}**\n', encoding='utf-8'
    )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
