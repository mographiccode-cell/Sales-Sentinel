from __future__ import annotations

import numpy as np

import train_merchant_market_fusion_v6_1 as base


def guarded_search_precision_recovery_policy(y, Z, fold_ids, severe=False):
    y = np.asarray(y, int)
    fold_ids = np.asarray(fold_ids, int)
    score = Z["merchant_mean"].to_numpy(float)
    market = Z[f"{base.MARKET_PREFIX}risk_p90"].to_numpy(float)

    low_values = base.candidate_values(score, [0.36, 0.40, 0.44, 0.48, 0.52, 0.56])
    high_values = sorted(set(
        [0.3918606905880165]
        + base.candidate_values(score, [0.54, 0.58, 0.62, 0.66, 0.70, 0.76])
    ))
    agree_values = base.candidate_values(
        np.minimum(Z["merchant_logreg"], Z["merchant_extra"]),
        [0.35, 0.45, 0.55, 0.65, 0.75],
    )
    market_values = base.candidate_values(market, [0.35, 0.48, 0.60, 0.72, 0.82])
    disagreement_values = base.candidate_values(
        Z["merchant_disagreement"], [0.40, 0.55, 0.70, 0.82]
    )

    rows = []
    for low_t in low_values:
        for high_t in high_values:
            if high_t <= low_t:
                continue
            for agree_t in agree_values:
                for market_t in market_values:
                    for disagreement_max in disagreement_values:
                        pred, diagnostics = base.make_policy_pred(
                            Z, low_t, high_t, agree_t, market_t, disagreement_max
                        )
                        m, per = base.evaluate_policy(y, score, pred, fold_ids)
                        alerts = m["tp"] + m["fp"]
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
                                1.45 * m["f1"]
                                + 0.65 * m["precision"]
                                + 0.55 * m["balanced_accuracy"]
                                + 0.70 * m["recall"]
                                + 0.25 * m["green_npv"]
                                - 0.60 * m["alert_rate"]
                                - 0.0015 * m["fp"]
                            )
                        rows.append({
                            "supported": bool(supported),
                            "objective": float(objective),
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
        raise RuntimeError("No guarded v6.1 policy candidates generated")

    feasible = [r for r in rows if r["supported"]]
    if severe:
        protected = []
    else:
        # V6 had 55 TP / 126 FP on the same 381 OOF rows. The v6.1 fallback is
        # forbidden from trading away more than five TP merely to improve precision.
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
        fallback_mode = "tp_protected_fallback"
    else:
        # Last-resort safety: prioritize retained TP and recall before objective.
        pool = rows
        fallback_mode = "recall_first_fallback"

    if fallback_mode == "recall_first_fallback" and not severe:
        pool.sort(
            key=lambda r: (
                r["metrics"]["tp"],
                r["metrics"]["recall"],
                -r["metrics"]["fp"],
                r["metrics"]["f1"],
                r["objective"],
            ),
            reverse=True,
        )
    else:
        pool.sort(
            key=lambda r: (
                r["supported"],
                r["metrics"]["f1"],
                r["metrics"]["precision"],
                -r["metrics"]["fp"],
                r["metrics"]["recall"],
                r["objective"],
            ),
            reverse=True,
        )

    best = pool[0]
    best["feasible_candidates"] = int(len(feasible))
    best["protected_fallback_candidates"] = int(len(protected))
    best["total_candidates"] = int(len(rows))
    best["selection_mode"] = fallback_mode
    return best


base.search_precision_recovery_policy = guarded_search_precision_recovery_policy

if __name__ == "__main__":
    base.main()
