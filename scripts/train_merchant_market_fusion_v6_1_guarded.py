from __future__ import annotations

import numpy as np

import train_merchant_market_fusion_v6_1 as base

_DRIFT_CACHE = {}


def causal_percentile_score(values, lookback=126, min_history=20):
    """Causal score normalization using only earlier model scores.

    No labels are used. For row i, the percentile is computed only against
    scores from rows < i, bounded to a recent lookback window. This makes the
    operational threshold less sensitive to probability-scale drift while
    preserving strict chronological availability.
    """
    x = np.asarray(values, float)
    ranks = np.full(len(x), 0.5, float)
    for i, value in enumerate(x):
        start = max(0, i - int(lookback))
        hist = x[start:i]
        hist = hist[np.isfinite(hist)]
        if len(hist) >= int(min_history):
            ranks[i] = (np.sum(hist < value) + 0.5 * np.sum(hist == value)) / len(hist)
    return ranks


def drift_adjusted_score(Z):
    """Return cached causal drift score for this exact OOF feature frame."""
    key = id(Z)
    cached = _DRIFT_CACHE.get(key)
    if cached is not None and len(cached) == len(Z):
        return cached
    raw = Z["merchant_mean"].to_numpy(float)
    rank = causal_percentile_score(raw)
    adjusted = 0.62 * raw + 0.38 * rank
    _DRIFT_CACHE[key] = adjusted
    return adjusted


def drift_make_policy_pred(Z, low_t, high_t, agree_t, market_t, disagreement_max):
    raw_mean = Z["merchant_mean"].to_numpy(float)
    adjusted = drift_adjusted_score(Z)
    merchant_min = np.minimum(
        Z["merchant_logreg"].to_numpy(float),
        Z["merchant_extra"].to_numpy(float),
    )
    disagreement = Z["merchant_disagreement"].to_numpy(float)
    market_p90 = Z[f"{base.MARKET_PREFIX}risk_p90"].to_numpy(float)
    market_mean = Z[f"{base.MARKET_PREFIX}risk_mean"].to_numpy(float)
    precursor = Z[f"{base.MARKET_PREFIX}precursor_share_2"].to_numpy(float)

    absolute_floor = float(np.quantile(raw_mean, 0.30))
    high = (adjusted >= high_t) & (raw_mean >= absolute_floor)
    marginal = (adjusted >= low_t) & (~high)

    merchant_agree = (merchant_min >= agree_t) & (disagreement <= disagreement_max)
    market_confirm = (
        (market_p90 >= market_t)
        | ((market_mean >= market_t * 0.60) & (precursor >= 0.20))
    )
    confirm = merchant_agree | market_confirm
    pred = high | (marginal & confirm)
    return pred, {
        "high_alerts": int(high.sum()),
        "marginal_candidates": int(marginal.sum()),
        "merchant_confirmed_marginal": int((marginal & merchant_agree).sum()),
        "market_confirmed_marginal": int((marginal & market_confirm).sum()),
        "final_alerts": int(pred.sum()),
        "drift_adjustment": "0.62_raw_plus_0.38_causal_percentile",
        "causal_rank_lookback": 126,
        "causal_rank_min_history": 20,
    }


def candidate_values(x, quantiles):
    x = np.asarray(x, float)
    return sorted(set(float(v) for v in np.quantile(x, quantiles)))


def gate_margin(m):
    terms = [
        m["recall"] / 0.80,
        m["precision"] / 0.32,
        m["f1"] / 0.46,
        m["green_npv"] / 0.95,
        0.43 / max(m["alert_rate"], 1e-9),
        m["worst_fold_recall"] / 0.50,
        0.60 / max(m["max_fold_alert_rate"], 1e-9),
    ]
    return float(min(terms))


def guarded_search_precision_recovery_policy(y, Z, fold_ids, severe=False):
    y = np.asarray(y, int)
    fold_ids = np.asarray(fold_ids, int)
    raw_score = Z["merchant_mean"].to_numpy(float)
    score = drift_adjusted_score(Z)
    market = Z[f"{base.MARKET_PREFIX}risk_p90"].to_numpy(float)

    low_values = candidate_values(score, [0.34, 0.40, 0.46, 0.52, 0.58, 0.64])
    high_values = candidate_values(score, [0.58, 0.64, 0.70, 0.76, 0.82, 0.88])
    agree_values = candidate_values(
        np.minimum(Z["merchant_logreg"], Z["merchant_extra"]),
        [0.32, 0.42, 0.52, 0.62, 0.72],
    )
    market_values = candidate_values(market, [0.35, 0.48, 0.60, 0.72, 0.82])
    disagreement_values = candidate_values(
        Z["merchant_disagreement"], [0.38, 0.52, 0.66, 0.80]
    )

    rows = []
    for low_t in low_values:
        for high_t in high_values:
            if high_t <= low_t:
                continue
            for agree_t in agree_values:
                for market_t in market_values:
                    for disagreement_max in disagreement_values:
                        pred, diagnostics = drift_make_policy_pred(
                            Z, low_t, high_t, agree_t, market_t, disagreement_max
                        )
                        m, per = base.evaluate_policy(y, raw_score, pred, fold_ids)
                        alerts = m["tp"] + m["fp"]
                        margin = gate_margin(m) if not severe else 0.0

                        if severe:
                            supported = (
                                alerts >= 5
                                and m["precision"] >= 0.55
                                and m["recall"] >= 0.20
                                and m["alert_rate"] <= 0.12
                            )
                            objective = (
                                1.60 * m["precision"]
                                + 0.70 * m["f1"]
                                + 0.35 * m["recall"]
                                - 0.60 * m["alert_rate"]
                            )
                        else:
                            supported = (
                                m["recall"] >= 0.80
                                and m["precision"] >= 0.32
                                and m["f1"] >= 0.46
                                and m["green_npv"] >= 0.95
                                and m["alert_rate"] <= 0.43
                                and m["worst_fold_recall"] >= 0.50
                                and m["max_fold_alert_rate"] <= 0.60
                            )
                            objective = (
                                1.20 * m["f1"]
                                + 0.55 * m["precision"]
                                + 0.45 * m["balanced_accuracy"]
                                + 0.60 * m["recall"]
                                + 0.30 * m["green_npv"]
                                + 1.20 * margin
                                - 0.55 * m["alert_rate"]
                                - 0.0015 * m["fp"]
                            )

                        rows.append({
                            "supported": bool(supported),
                            "objective": float(objective),
                            "gate_margin": float(margin),
                            "params": {
                                "low_threshold": float(low_t),
                                "high_threshold": float(high_t),
                                "agreement_threshold": float(agree_t),
                                "market_p90_threshold": float(market_t),
                                "max_model_disagreement": float(disagreement_max),
                            },
                            "metrics": m,
                            "per_fold": per,
                            "diagnostics": diagnostics,
                        })

    if not rows:
        raise RuntimeError("No drift-aware v6.1 policy candidates generated")

    feasible = [r for r in rows if r["supported"]]
    if severe:
        protected = []
    else:
        protected = [
            r for r in rows
            if r["metrics"]["tp"] >= 50
            and r["metrics"]["fp"] < 126
            and r["metrics"]["recall"] >= 0.79
            and r["metrics"]["green_npv"] >= 0.94
            and r["metrics"]["worst_fold_recall"] >= 0.40
            and r["metrics"]["max_fold_alert_rate"] <= 0.68
        ]

    if feasible:
        pool = feasible
        fallback_mode = "fully_feasible"
    elif protected:
        pool = protected
        fallback_mode = "tp_protected_drift_fallback"
    else:
        pool = rows
        fallback_mode = "recall_first_drift_fallback"

    if severe:
        pool.sort(
            key=lambda r: (
                r["supported"],
                r["objective"],
                r["metrics"]["precision"],
                r["metrics"]["recall"],
            ),
            reverse=True,
        )
    elif feasible:
        pool.sort(
            key=lambda r: (
                r["gate_margin"],
                -r["metrics"]["fp"],
                r["metrics"]["f1"],
                r["metrics"]["recall"],
            ),
            reverse=True,
        )
    elif protected:
        pool.sort(
            key=lambda r: (
                r["gate_margin"],
                -r["metrics"]["fp"],
                r["metrics"]["worst_fold_recall"],
                -r["metrics"]["max_fold_alert_rate"],
                r["metrics"]["f1"],
            ),
            reverse=True,
        )
    else:
        pool.sort(
            key=lambda r: (
                r["metrics"]["tp"],
                r["metrics"]["worst_fold_recall"],
                -r["metrics"]["fp"],
                r["gate_margin"],
            ),
            reverse=True,
        )

    best = pool[0]
    best["feasible_candidates"] = int(len(feasible))
    best["protected_fallback_candidates"] = int(len(protected))
    best["total_candidates"] = int(len(rows))
    best["selection_mode"] = fallback_mode
    best["policy_score"] = "drift_adjusted_causal_merchant_score"
    return best


base.search_precision_recovery_policy = guarded_search_precision_recovery_policy
base.make_policy_pred = drift_make_policy_pred

if __name__ == "__main__":
    base.main()
