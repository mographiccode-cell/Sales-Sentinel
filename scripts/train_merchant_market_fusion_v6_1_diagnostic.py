from __future__ import annotations

import pandas as pd

import train_merchant_market_fusion_v6_1 as base
import train_merchant_market_fusion_v6_1_guarded as guarded

_original_search = guarded.guarded_search_precision_recovery_policy


def diagnostic_search(y, Z, fold_ids, severe=False):
    if not severe:
        out = pd.DataFrame({
            "y": y,
            "fold_id": fold_ids,
            "merchant_logreg": Z["merchant_logreg"].to_numpy(float),
            "merchant_extra": Z["merchant_extra"].to_numpy(float),
            "merchant_mean": Z["merchant_mean"].to_numpy(float),
            "merchant_disagreement": Z["merchant_disagreement"].to_numpy(float),
            "drift_adjusted_score": guarded.drift_adjusted_score(Z),
        })
        for c in Z.columns:
            if c.startswith(base.MARKET_PREFIX):
                out[c] = Z[c].to_numpy(float)
        out.to_csv(base.OUT / "oof_policy_diagnostics.csv", index=False)
    return _original_search(y, Z, fold_ids, severe=severe)


base.search_precision_recovery_policy = diagnostic_search
base.make_policy_pred = guarded.drift_make_policy_pred

if __name__ == "__main__":
    base.main()
