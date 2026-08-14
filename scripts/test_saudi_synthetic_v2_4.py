from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from production_city_risk_engine_v2_4 import predict_latest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'saudi_synthetic_v2_4'
FIX = ROOT / 'tests' / 'fixtures' / 'saudi_synthetic_v2_4'
OUT.mkdir(parents=True, exist_ok=True)
FIX.mkdir(parents=True, exist_ok=True)
SEED = 20260814
RNG = np.random.default_rng(SEED)

CITIES = {
    'RIYADH': 4_900_000.0,
    'JEDDAH': 3_550_000.0,
    'MAKKAH': 2_050_000.0,
    'DAMMAM': 1_650_000.0,
    'MADINA': 1_350_000.0,
    'KHOBAR': 1_000_000.0,
    'ABHA': 820_000.0,
    'BURAIDAH': 720_000.0,
    'HAIL': 590_000.0,
    'TABOUK': 560_000.0,
    'OTHER': 2_900_000.0,
}

AVG_TICKET = {
    'RIYADH': 190.0, 'JEDDAH': 185.0, 'MAKKAH': 170.0, 'DAMMAM': 195.0,
    'MADINA': 165.0, 'KHOBAR': 205.0, 'ABHA': 160.0, 'BURAIDAH': 155.0,
    'HAIL': 150.0, 'TABOUK': 158.0, 'OTHER': 165.0,
}

# Gregorian windows used only to make the synthetic market Saudi-like. They are scenario assumptions, not official SAMA observations.
RAMADAN = [
    ('2023-03-23','2023-04-21'), ('2024-03-11','2024-04-09'),
    ('2025-03-01','2025-03-30'), ('2026-02-18','2026-03-19'),
]
HAJJ = [
    ('2023-06-20','2023-07-02'), ('2024-06-08','2024-06-20'),
    ('2025-05-30','2025-06-12'), ('2026-05-20','2026-06-02'),
]

# Calibration-only shocks occur BEFORE the scored period so the production policy has enough
# realized historical declines to operate. They are never counted in the reported test metrics.
CALIBRATION_SHOCKS = {}
_cal_cities = list(CITIES)
for i, ws in enumerate(pd.date_range('2023-07-02','2024-12-29',freq='3W-SUN')):
    CALIBRATION_SHOCKS[(str(ws.date()), _cal_cities[i % len(_cal_cities)])] = 0.60 + 0.02 * (i % 3)

# Strong shocks are fixed BEFORE model scoring. They are intentionally spread across cities and seasons.
STRONG_SHOCKS = {
    ('2025-02-16','RIYADH'): 0.68,
    ('2025-04-20','JEDDAH'): 0.70,
    ('2025-07-20','MAKKAH'): 0.64,
    ('2025-09-14','DAMMAM'): 0.69,
    ('2025-11-30','HAIL'): 0.66,
    ('2026-01-18','ABHA'): 0.68,
    ('2026-03-29','MADINA'): 0.63,
    ('2026-04-12','OTHER'): 0.69,
    ('2026-05-10','BURAIDAH'): 0.68,
    ('2026-06-21','TABOUK'): 0.67,
    ('2026-07-19','KHOBAR'): 0.70,
}

# Mild dips should generally remain below the project's 20% decline definition.
MILD_DIPS = {
    ('2025-01-12','JEDDAH'): 0.88,
    ('2025-05-18','RIYADH'): 0.87,
    ('2025-08-17','MADINA'): 0.89,
    ('2025-10-12','ABHA'): 0.86,
    ('2026-02-01','DAMMAM'): 0.88,
    ('2026-04-26','MAKKAH'): 0.87,
    ('2026-07-05','HAIL'): 0.89,
}


def in_window(ts: pd.Timestamp, windows) -> bool:
    return any(pd.Timestamp(a) <= ts <= pd.Timestamp(b) for a,b in windows)


def week_contains_day(ts: pd.Timestamp, month: int, day: int) -> bool:
    end = ts + pd.Timedelta(days=6)
    for year in {ts.year, end.year}:
        d = pd.Timestamp(year=year, month=month, day=day)
        if ts <= d <= end:
            return True
    return False


def salary_week(ts: pd.Timestamp) -> bool:
    end = ts + pd.Timedelta(days=6)
    for d in pd.date_range(ts, end, freq='D'):
        if d.day in (26,27,28):
            return True
    return False


def seasonal_multiplier(city: str, ts: pd.Timestamp) -> float:
    m = 1.0
    # Smooth annual Saudi retail rhythm.
    w = float(ts.isocalendar().week)
    m *= 1.0 + 0.045 * math.sin(2 * math.pi * (w - 4) / 52.18)
    if salary_week(ts):
        m *= 1.055
    if week_contains_day(ts, 2, 22):  # Founding Day
        m *= 1.035
    if week_contains_day(ts, 9, 23):  # National Day
        m *= 1.060
    if in_window(ts, RAMADAN):
        m *= 1.10
        if city in {'MAKKAH','MADINA'}:
            m *= 1.08
    if in_window(ts, HAJJ):
        if city == 'MAKKAH':
            m *= 1.25
        elif city == 'MADINA':
            m *= 1.12
        else:
            m *= 1.02
    if city == 'ABHA' and ts.month in (6,7,8):
        m *= 1.10
    if city in {'JEDDAH','MAKKAH'} and ts.month in (12,1):
        m *= 1.035
    return m


def generate_panel() -> pd.DataFrame:
    weeks = pd.date_range('2023-01-01','2026-08-02',freq='W-SUN')
    rows = []
    city_state = {c: 1.0 for c in CITIES}
    for wi, ws in enumerate(weeks):
        market_growth = (1.0018 ** wi)
        macro_cycle = 1.0 + 0.018 * math.sin(2 * math.pi * wi / 26.0)
        shared_noise = float(np.exp(RNG.normal(0, 0.018)))
        for city, base in CITIES.items():
            # Slow city-specific state with mean reversion.
            city_state[city] = 0.985 * city_state[city] + 0.015 + RNG.normal(0, 0.006)
            city_state[city] = float(np.clip(city_state[city], 0.92, 1.08))
            city_noise = float(np.exp(RNG.normal(0, 0.028)))
            value = base * market_growth * macro_cycle * shared_noise * city_state[city] * city_noise
            value *= seasonal_multiplier(city, ws)
            scenario = 'normal'
            key = (str(ws.date()), city)
            if key in CALIBRATION_SHOCKS:
                value *= CALIBRATION_SHOCKS[key]
                scenario = 'historical_calibration_decline'
            if key in MILD_DIPS:
                value *= MILD_DIPS[key]
                scenario = 'mild_dip'
            if key in STRONG_SHOCKS:
                value *= STRONG_SHOCKS[key]
                scenario = 'forced_strong_decline'
            # A few non-decline positive shocks to test false RED alerts.
            if (str(ws.date()), city) in {
                ('2025-03-16','MAKKAH'), ('2025-09-21','RIYADH'),
                ('2026-02-22','JEDDAH'), ('2026-05-24','MADINA'),
            }:
                value *= 1.24
                scenario = 'positive_spike'
            ticket = AVG_TICKET[city] * (1 + 0.015 * math.sin(2 * math.pi * wi / 52.18))
            count = max(1.0, value / ticket)
            count *= float(np.exp(RNG.normal(0, 0.012)))
            rows.append({
                'week_start': ws,
                'week_end': ws + pd.Timedelta(days=6),
                'city': city,
                'value_thousand_sar': round(value, 3),
                'transaction_count_thousand': round(count, 3),
                'scenario': scenario,
            })
    return pd.DataFrame(rows).sort_values(['week_start','city']).reset_index(drop=True)


def add_ground_truth(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy().sort_values(['city','week_start']).reset_index(drop=True)
    g = d.groupby('city', sort=False)
    d['baseline4'] = g.value_thousand_sar.transform(lambda s: s.rolling(4,min_periods=4).mean())
    d['next_week_value'] = g.value_thousand_sar.shift(-1)
    d['next_week_ratio'] = d.next_week_value / d.baseline4.replace(0,np.nan)
    d['actual_decline_gt20'] = (d.next_week_ratio < .80).astype('Int64')
    d.loc[d.next_week_ratio.isna(),'actual_decline_gt20'] = pd.NA
    d['actual_decline_pct'] = (1.0 - d.next_week_ratio).clip(lower=0)
    return d


def evaluate(panel: pd.DataFrame, truth: pd.DataFrame):
    weeks = sorted(panel.week_start.unique())
    pred_rows = []
    # Scored period begins only after the historical calibration shocks are complete.
    score_start = pd.Timestamp('2025-02-09')
    for origin in [w for w in weeks[:-1] if pd.Timestamp(w) >= score_start]:
        hist = panel[panel.week_start <= origin][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']].copy()
        result = predict_latest(hist)
        if result.get('status') != 'OK':
            pred_rows.append({'week_start': pd.Timestamp(origin), 'city': '__ENGINE__', 'state':'NO_DECISION', 'reason':result.get('reason','UNKNOWN')})
            continue
        for p in result['predictions']:
            pred_rows.append({
                'week_start': pd.Timestamp(p['week_start']), 'city': p['city'], 'state': p['state'],
                'base_probability': p['base_probability'], 'adjusted_probability': p['adjusted_probability'],
                'global_prior': p['global_prior'], 'city_prior': p['city_prior'], 'effective_prior': p['effective_prior'],
            })
    preds = pd.DataFrame(pred_rows)
    if (preds.city == '__ENGINE__').any():
        bad = preds[preds.city=='__ENGINE__']
        raise RuntimeError('Production engine returned NO_DECISION during synthetic evaluation: ' + bad.head(20).to_json(orient='records'))
    eval_df = preds.merge(
        truth[['week_start','city','scenario','actual_decline_gt20','actual_decline_pct','next_week_ratio']],
        on=['week_start','city'], how='left', validate='one_to_one'
    )
    eval_df['actual_decline_gt20'] = eval_df.actual_decline_gt20.astype(int)
    y = eval_df.actual_decline_gt20.to_numpy(int)
    red = eval_df.state.eq('RED').to_numpy()
    alert = eval_df.state.isin(['RED','AMBER']).to_numpy()
    green = eval_df.state.eq('GREEN').to_numpy()
    def safe(a,b): return float(a/b) if b else None
    red_tp = int((red & (y==1)).sum()); red_fp = int((red & (y==0)).sum())
    red_fn = int((~red & (y==1)).sum()); red_tn = int((~red & (y==0)).sum())
    alert_tp = int((alert & (y==1)).sum()); alert_fp = int((alert & (y==0)).sum())
    green_tn = int((green & (y==0)).sum()); green_fn = int((green & (y==1)).sum())
    report = {
        'version':'SAUDI-SYNTHETIC-STRESS-V2.4-2',
        'synthetic_not_official': True,
        'seed': SEED,
        'scored_period_starts_after_calibration_only_period': True,
        'score_start': str(score_start.date()),
        'model_engine':'SALES-SENTINEL-CITY-RISK-ENGINE-2.4.2',
        'rows_scored': int(len(eval_df)),
        'weeks_scored': int(eval_df.week_start.nunique()),
        'cities': int(eval_df.city.nunique()),
        'actual_declines': int(y.sum()),
        'actual_decline_rate': float(y.mean()),
        'states': {k:int(v) for k,v in eval_df.state.value_counts().to_dict().items()},
        'RED': {
            'TP':red_tp,'FP':red_fp,'FN':red_fn,'TN':red_tn,
            'precision':safe(red_tp,red_tp+red_fp),
            'recall_contribution':safe(red_tp,int(y.sum())),
            'FPR':safe(red_fp,red_fp+red_tn),
        },
        'RED_plus_AMBER': {
            'TP':alert_tp,'FP':alert_fp,
            'recall':safe(alert_tp,int(y.sum())),
            'precision':safe(alert_tp,alert_tp+alert_fp),
        },
        'GREEN': {
            'TN':green_tn,'FN':green_fn,
            'NPV':safe(green_tn,green_tn+green_fn),
            'miss_rate':safe(green_fn,int(y.sum())),
        },
        'scenario_counts_all_data': {k:int(v) for k,v in truth.scenario.value_counts().to_dict().items()},
        'interpretation':'Synthetic Saudi-like robustness/functional stress test only; it is not independent evidence of real-world generalization.',
    }
    return eval_df, report


def main():
    panel = generate_panel()
    truth = add_ground_truth(panel)
    app_input = panel[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']].copy()
    eval_df, report = evaluate(panel, truth)

    app_input.to_csv(FIX/'saudi_synthetic_app_input.csv', index=False)
    truth[['week_start','week_end','city','scenario','baseline4','next_week_value','next_week_ratio','actual_decline_pct','actual_decline_gt20']].to_csv(FIX/'saudi_synthetic_ground_truth.csv', index=False)
    eval_df.to_csv(FIX/'saudi_synthetic_model_predictions.csv', index=False)
    (OUT/'synthetic_test_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'synthetic_test_summary.md').write_text(
        '# Saudi Synthetic v2.4 Stress Test\n\n'
        f"- Scored rows: **{report['rows_scored']:,}**\n"
        f"- Actual declines: **{report['actual_declines']} ({report['actual_decline_rate']:.2%})**\n"
        f"- RED precision: **{report['RED']['precision']:.2%}** ({report['RED']['TP']} TP / {report['RED']['FP']} FP)\n"
        f"- RED FPR: **{report['RED']['FPR']:.2%}**\n"
        f"- RED+AMBER recall: **{report['RED_plus_AMBER']['recall']:.2%}**\n"
        f"- GREEN NPV: **{report['GREEN']['NPV']:.2%}**\n"
        f"- Missed declines in GREEN: **{report['GREEN']['FN']} / {report['actual_declines']}**\n\n"
        'Synthetic Saudi-like test only; not real-world validation.\n',
        encoding='utf-8'
    )
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
