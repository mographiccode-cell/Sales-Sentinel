from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from production_city_risk_engine_v3 import MODEL, predict_latest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'sama_pos' / 'sama_city_weekly_value_count_2020_2026_extended.csv'
OUT = ROOT / 'reports' / 'sama_city_v3_3' / 'counterfactual_stress'
OUT.mkdir(parents=True, exist_ok=True)

# Fixed before scoring. The event happens one week AFTER the prediction origin.
# The model is allowed to see only the three completed precursor weeks.
PRECURSOR_FACTORS = {-2: 0.95, -1: 0.90, 0: 0.84}
EVENT_TARGET_RATIO = 0.65
EXPECTED_MODEL_VERSION = 'SAMA-CITY-RISK-3.3-DUAL-CHANNEL'
CITIES = ['ABHA','BURAIDAH','DAMMAM','HAIL','JEDDAH','KHOBAR','MADINA','MAKKAH','OTHER','RIYADH','TABOUK']


def inject_one(panel: pd.DataFrame, city: str, origin: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    d = panel.copy()
    weeks = sorted(pd.to_datetime(d.week_start.unique()))
    idx = weeks.index(pd.Timestamp(origin))
    if idx < 3 or idx + 1 >= len(weeks):
        raise RuntimeError('Origin does not have required history/event week')
    event_week = pd.Timestamp(weeks[idx + 1])

    # Apply three visible precursor weeks ending at origin.
    for offset, factor in PRECURSOR_FACTORS.items():
        w = pd.Timestamp(weeks[idx + offset])
        m = d.week_start.eq(w) & d.city.eq(city)
        if int(m.sum()) != 1:
            raise RuntimeError(f'Missing city-week for {city} {w.date()}')
        d.loc[m, 'value_thousand_sar'] *= factor
        d.loc[m, 'transaction_count_thousand'] *= factor

    # Calculate the target from the visible history ONLY, then define the synthetic next week.
    # This avoids a calendar-specific future spike weakening the intended >20% decline.
    city_hist = d[(d.city.eq(city)) & (d.week_start <= origin)].sort_values('week_start')
    if len(city_hist) < 4:
        raise RuntimeError('Insufficient city history for injected target')
    trailing4 = float(city_hist.tail(4).value_thousand_sar.mean())
    if not np.isfinite(trailing4) or trailing4 <= 0:
        raise RuntimeError('Invalid trailing4 for injected target')

    m = d.week_start.eq(event_week) & d.city.eq(city)
    if int(m.sum()) != 1:
        raise RuntimeError(f'Missing event city-week for {city} {event_week.date()}')
    original_event_value = float(d.loc[m, 'value_thousand_sar'].iloc[0])
    original_event_count = float(d.loc[m, 'transaction_count_thousand'].iloc[0])
    desired_value = EVENT_TARGET_RATIO * trailing4
    scale = desired_value / max(original_event_value, 1e-12)
    d.loc[m, 'value_thousand_sar'] = desired_value
    d.loc[m, 'transaction_count_thousand'] = original_event_count * scale

    city_rows = d[d.city.eq(city)].sort_values('week_start').reset_index(drop=True)
    pos = int(city_rows.index[city_rows.week_start.eq(origin)][0])
    next_value = float(city_rows.loc[pos+1, 'value_thousand_sar'])
    ratio = next_value / trailing4
    if not np.isfinite(ratio) or abs(ratio - EVENT_TARGET_RATIO) > 1e-9:
        raise RuntimeError(f'Injected ratio mismatch: {city} {origin.date()} ratio={ratio}')
    if ratio >= 0.80:
        raise RuntimeError(f'Injected scenario did not create >20% decline: {city} {origin.date()} ratio={ratio}')
    return d, {
        'city': city, 'origin': str(origin.date()), 'event_week': str(event_week.date()),
        'trailing4_value': trailing4, 'event_value': next_value, 'next_ratio': ratio,
    }


def prediction_map(result: dict) -> dict[str, dict]:
    if result.get('status') != 'OK':
        raise RuntimeError(json.dumps(result))
    if result.get('model_version') != EXPECTED_MODEL_VERSION:
        raise RuntimeError(f"Serving/model mismatch: {result.get('model_version')} != {EXPECTED_MODEL_VERSION}")
    return {str(x['city']): x for x in result['predictions']}


def main():
    artifact = joblib.load(MODEL)
    if artifact.get('version') != EXPECTED_MODEL_VERSION:
        raise RuntimeError(f'Wrong frozen artifact: {artifact.get("version")}')

    panel = pd.read_csv(DATA, parse_dates=['week_start','week_end']).sort_values(['week_start','city']).reset_index(drop=True)
    weeks = sorted(pd.to_datetime(panel.week_start.unique()))
    eligible = [w for w in weeks if pd.Timestamp('2025-07-13') <= w <= pd.Timestamp('2026-07-05')]
    if len(eligible) < 30:
        raise RuntimeError(f'Not enough post-development real-shape weeks: {len(eligible)}')

    # Deterministic, label-free origin selection from calendar positions only.
    # 22 tests = two separate injected scenarios for every city.
    candidate_positions = np.linspace(4, len(eligible)-3, 22, dtype=int)
    origins = [pd.Timestamp(eligible[i]) for i in candidate_positions]
    scenarios = [(CITIES[i % len(CITIES)], origins[i]) for i in range(22)]

    rows = []
    unaffected_rows = []
    for sid, (city, origin) in enumerate(scenarios, start=1):
        base_hist = panel[panel.week_start <= origin][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']].copy()
        base_map = prediction_map(predict_latest(base_hist))

        injected, truth = inject_one(panel, city, origin)
        inj_hist = injected[injected.week_start <= origin][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']].copy()
        inj_map = prediction_map(predict_latest(inj_hist))

        p = inj_map[city]
        b = base_map[city]
        rows.append({
            'scenario_id': sid, **truth,
            'baseline_state': b['state'], 'baseline_score': b['risk_score'],
            'injected_state': p['state'], 'injected_reason': p['reason'],
            'injected_score': p['risk_score'], 'precursor_count': p['precursor_count'],
            'ood_fraction': p['ood_fraction'],
            'score_lift': p['risk_score'] - b['risk_score'],
            'alerted': int(p['state'] in {'RED','AMBER'}), 'red': int(p['state'] == 'RED'),
        })

        # Other 10 cities were not modified; any new RED/AMBER is a spillover diagnostic.
        for other in CITIES:
            if other == city:
                continue
            bp = base_map[other]; ip = inj_map[other]
            unaffected_rows.append({
                'scenario_id': sid, 'origin': str(origin.date()), 'injected_city': city, 'control_city': other,
                'baseline_state': bp['state'], 'injected_state': ip['state'],
                'baseline_score': bp['risk_score'], 'injected_score': ip['risk_score'],
                'new_red': int(bp['state'] != 'RED' and ip['state'] == 'RED'),
                'new_alert': int(bp['state'] == 'GREEN' and ip['state'] in {'RED','AMBER'}),
                'score_change': ip['risk_score'] - bp['risk_score'],
            })

    s = pd.DataFrame(rows)
    c = pd.DataFrame(unaffected_rows)
    injected_recall = float(s.alerted.mean())
    red_rate = float(s.red.mean())
    median_score_lift = float(s.score_lift.median())
    new_control_red_rate = float(c.new_red.mean())
    new_control_alert_rate = float(c.new_alert.mean())

    # Acceptance contract was fixed before the first result and remains unchanged.
    acceptance = {
        'injected_decline_alert_recall_ge_90pct': injected_recall >= 0.90,
        'injected_median_score_lift_positive': median_score_lift > 0.0,
        'control_new_red_rate_le_1pct': new_control_red_rate <= 0.01,
        'control_new_alert_rate_le_5pct': new_control_alert_rate <= 0.05,
        'all_injected_targets_verified_gt20pct': bool((s.next_ratio < 0.80).all()),
        'serving_uses_exact_v3_3_artifact': artifact.get('version') == EXPECTED_MODEL_VERSION,
    }

    report = {
        'version': 'SAMA-CITY-V3.3-COUNTERFACTUAL-STRESS-1',
        'test_type': 'counterfactual stress test on real SAMA-shaped covariates; injected labels are synthetic and fixed before scoring',
        'frozen_model': EXPECTED_MODEL_VERSION,
        'model_development_end': artifact.get('development_end'),
        'scenario_count': int(len(s)), 'control_rows': int(len(c)),
        'injection_contract': {'precursor_factors': {str(k):v for k,v in PRECURSOR_FACTORS.items()}, 'event_target_ratio': EVENT_TARGET_RATIO},
        'injected_declines': {
            'alerted': int(s.alerted.sum()), 'RED': int(s.red.sum()),
            'alert_recall': injected_recall, 'red_rate': red_rate,
            'median_score_lift': median_score_lift,
            'median_next_ratio': float(s.next_ratio.median()),
            'min_next_ratio': float(s.next_ratio.min()), 'max_next_ratio': float(s.next_ratio.max()),
            'states': {k:int(v) for k,v in s.injected_state.value_counts().to_dict().items()},
            'reasons': {k:int(v) for k,v in s.injected_reason.value_counts().to_dict().items()},
        },
        'controls': {
            'new_red': int(c.new_red.sum()), 'new_red_rate': new_control_red_rate,
            'new_alert': int(c.new_alert.sum()), 'new_alert_rate': new_control_alert_rate,
            'max_abs_score_change': float(c.score_change.abs().max()),
        },
        'acceptance': acceptance,
        'all_acceptance_passed': bool(all(acceptance.values())),
        'scientific_boundary': 'No model weight, feature, threshold, or policy parameter is fit in this script. Real SAMA outcomes are not used as labels for injected scenarios.',
    }

    s.to_csv(OUT / 'injected_scenarios.csv', index=False)
    c.to_csv(OUT / 'unaffected_controls.csv', index=False)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (OUT / 'summary.md').write_text(
        '# Sales Sentinel v3.3 — Counterfactual SAMA Stress Test\n\n'
        f'- Injected forecastable declines: **{int(s.alerted.sum())}/{len(s)} alerted ({injected_recall:.2%})**\n'
        f'- RED among injected cases: **{int(s.red.sum())}/{len(s)} ({red_rate:.2%})**\n'
        f'- Median risk-score lift: **{median_score_lift:.4f}**\n'
        f'- Unaffected control new RED: **{int(c.new_red.sum())}/{len(c)} ({new_control_red_rate:.2%})**\n'
        f'- Unaffected control new alert: **{int(c.new_alert.sum())}/{len(c)} ({new_control_alert_rate:.2%})**\n'
        f'- All acceptance gates: **{report["all_acceptance_passed"]}**\n',
        encoding='utf-8'
    )
    print(json.dumps(report, indent=2))
    if not report['all_acceptance_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
