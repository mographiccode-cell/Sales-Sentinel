from __future__ import annotations

import numpy as np
import train_merchant_continuous_ratio_v9 as v9
import train_merchant_error_corrector_v7_5 as v75


def corrected_prequential(y, folds, base, strong, risk, pred_ratio):
    final = np.asarray(base, bool).copy()
    details = []
    for f in sorted(np.unique(folds)):
        cur = folds == f
        if f == 0:
            details.append({"fold_id": int(f), "mode": "v8_2_bootstrap"})
            continue
        hist = folds < f
        bm = v75.metrics(y[hist], base[hist], folds[hist])
        best = None
        for rule in v9.rule_grid():
            p = v9.apply_rule(base[hist], strong[hist], risk[hist], pred_ratio[hist], rule)
            m = v75.metrics(y[hist], p, folds[hist])
            feasible = (
                m["recall"] >= bm["recall"] and
                m["green_npv"] >= bm["green_npv"] - .002 and
                m["fp"] <= bm["fp"] and
                m["f1"] >= bm["f1"] and
                m["alert_rate"] <= bm["alert_rate"]
            )
            key = (int(feasible), m["f1"], m["precision"], -m["fp"], m["recall"], -m["alert_rate"])
            if best is None or key > best[0]:
                best = (key, rule, m, feasible)
        _, rule, hm, feasible = best
        if not feasible:
            rule = ("none", 0.0, .95, 1.01)
        final[cur] = v9.apply_rule(base[cur], strong[cur], risk[cur], pred_ratio[cur], rule)
        details.append({"fold_id": int(f), "history_rule": list(rule), "history_feasible": bool(feasible), "history_metrics": hm})
    return final, details


v9.prequential = corrected_prequential
v9.main()
