from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import train_merchant_error_corrector_v7_5 as v75

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-V13-SAUDI-CALENDAR-REGIME-VERIFIER"
BASE = ROOT / "reports" / "merchant_multihorizon_v11" / "oof_predictions.csv"
V92 = ROOT / "reports" / "merchant_meta_verifier_v9_2" / "oof_predictions.csv"
DIAG = ROOT / "reports" / "merchant_market_fusion_v6_1" / "oof_policy_diagnostics.csv"
OUT = ROOT / "reports" / "merchant_saudi_calendar_v13"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
OOF = OUT / "oof_predictions.csv"
SEED = 42

# Historical Saudi calendar anchors used only as deterministic calendar covariates.
RAMADAN = [
    (pd.Timestamp("2023-03-23"), pd.Timestamp("2023-04-20")),
    (pd.Timestamp("2024-03-11"), pd.Timestamp("2024-04-09")),
]
EID_FITR = [pd.Timestamp("2023-04-21"), pd.Timestamp("2024-04-10")]
EID_ADHA = [pd.Timestamp("2023-06-28"), pd.Timestamp("2024-06-16")]


def window_flag(dates: pd.Series, anchors: list[pd.Timestamp], lo: int, hi: int) -> np.ndarray:
    out = np.zeros(len(dates), dtype=float)
    dt = pd.to_datetime(dates)
    for a in anchors:
        delta = (dt - a).dt.days
        out = np.maximum(out, ((delta >= lo) & (delta <= hi)).astype(float).to_numpy())
    return out


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(dates).reset_index(drop=True)
    x = pd.DataFrame(index=np.arange(len(dt)))
    ram_day = np.zeros(len(dt), dtype=float)
    is_ram = np.zeros(len(dt), dtype=float)
    for start, end in RAMADAN:
        m = (dt >= start) & (dt <= end)
        day = (dt - start).dt.days + 1
        is_ram[m] = 1.0
        ram_day[m] = day[m].to_numpy(float)
    x["cal__is_ramadan"] = is_ram
    x["cal__ramadan_day"] = ram_day / 30.0
    x["cal__ramadan_first10"] = ((ram_day >= 1) & (ram_day <= 10)).astype(float)
    x["cal__ramadan_middle10"] = ((ram_day >= 11) & (ram_day <= 20)).astype(float)
    x["cal__ramadan_last10"] = ((ram_day >= 21) & (ram_day <= 30)).astype(float)
    x["cal__ramadan_sin"] = np.where(is_ram > 0, np.sin(2 * np.pi * ram_day / 30.0), 0.0)
    x["cal__ramadan_cos"] = np.where(is_ram > 0, np.cos(2 * np.pi * ram_day / 30.0), 0.0)

    x["cal__pre_fitr_7"] = window_flag(dt, EID_FITR, -7, -1)
    x["cal__fitr_day0_3"] = window_flag(dt, EID_FITR, 0, 3)
    x["cal__post_fitr_7"] = window_flag(dt, EID_FITR, 4, 10)
    x["cal__pre_adha_7"] = window_flag(dt, EID_ADHA, -7, -1)
    x["cal__adha_day0_3"] = window_flag(dt, EID_ADHA, 0, 3)
    x["cal__post_adha_7"] = window_flag(dt, EID_ADHA, 4, 10)

    # Distance-to-next holiday is clipped and computed from known calendar dates, not outcomes.
    def dist_to_next(anchors: list[pd.Timestamp]) -> np.ndarray:
        vals = []
        for t in dt:
            ds = [(a - t).days for a in anchors if a >= t]
            vals.append(min(ds) if ds else 60)
        return np.clip(np.asarray(vals, float), 0, 60) / 60.0

    x["cal__days_to_fitr"] = dist_to_next(EID_FITR)
    x["cal__days_to_adha"] = dist_to_next(EID_ADHA)
    x["cal__special_regime"] = np.maximum.reduce([
        x["cal__is_ramadan"].to_numpy(), x["cal__pre_fitr_7"].to_numpy(),
        x["cal__fitr_day0_3"].to_numpy(), x["cal__post_fitr_7"].to_numpy(),
        x["cal__pre_adha_7"].to_numpy(), x["cal__adha_day0_3"].to_numpy(),
        x["cal__post_adha_7"].to_numpy(),
    ])
    return x


def build_features():
    b = pd.read_csv(BASE).sort_values(["fold_id", "date"]).reset_index(drop=True)
    v = pd.read_csv(V92).sort_values(["fold_id", "date"]).reset_index(drop=True)
    d = pd.read_csv(DIAG).reset_index(drop=True)
    if not (len(b) == len(v) == len(d) == 381):
        raise RuntimeError("OOF length mismatch")
    y = b.y.to_numpy(int)
    if not np.array_equal(y, v.y.to_numpy(int)) or not np.array_equal(y, d.y.to_numpy(int)):
        raise RuntimeError("OOF target mismatch")

    x = pd.DataFrame(index=b.index)
    x["risk3"] = pd.to_numeric(b.risk3, errors="coerce").fillna(0.0)
    x["risk14"] = pd.to_numeric(b.risk14, errors="coerce").fillna(0.0)
    x["v9_risk"] = pd.to_numeric(v.v9_risk, errors="coerce").fillna(0.0)
    x["v76_score"] = pd.to_numeric(v.v76_score, errors="coerce").fillna(0.0)
    x["risk_mean"] = x[["risk3", "risk14", "v9_risk", "v76_score"]].mean(axis=1)
    x["risk_min"] = x[["risk3", "risk14", "v9_risk", "v76_score"]].min(axis=1)
    x["risk_max"] = x[["risk3", "risk14", "v9_risk", "v76_score"]].max(axis=1)
    x["horizon_gap"] = (x.risk3 - x.risk14).abs()

    for c in [
        "merchant_logreg", "merchant_extra", "merchant_mean", "merchant_disagreement",
        "market_v3__risk_mean", "market_v3__risk_max", "market_v3__risk_p90",
        "market_v3__risk_share_25", "market_v3__precursor_mean",
    ]:
        if c in d.columns:
            x[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    strong, quiet, market, _ = v75.base_components(d)
    x["branch_strong"] = strong.astype(float)
    x["branch_quiet"] = quiet.astype(float)
    x["branch_market"] = market.astype(float)

    cal = calendar_features(pd.to_datetime(b.date))
    for c in cal.columns:
        x[c] = cal[c].to_numpy(float)

    # Compact interactions: whether weak/strong risk occurs during known Saudi seasonal regimes.
    x["calx__special_x_risk_mean"] = x["cal__special_regime"] * x["risk_mean"]
    x["calx__ramadan_x_risk_mean"] = x["cal__is_ramadan"] * x["risk_mean"]
    x["calx__pre_fitr_x_risk_mean"] = x["cal__pre_fitr_7"] * x["risk_mean"]
    x["calx__pre_adha_x_risk_mean"] = x["cal__pre_adha_7"] * x["risk_mean"]
    return b, v, d, x, strong, market


def fit_predict(A: pd.DataFrame, y: np.ndarray, B: pd.DataFrame, C: float):
    A = A.replace([np.inf, -np.inf], np.nan)
    B = B.replace([np.inf, -np.inf], np.nan)
    med = A.median().fillna(0.0)
    A = A.fillna(med); B = B.fillna(med)
    sc = StandardScaler()
    X = sc.fit_transform(A); Z = sc.transform(B)
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=SEED)
    m.fit(X, y)
    return m.predict_proba(X)[:, 1], m.predict_proba(Z)[:, 1]


def causal_candidate(b, x, strong, market, cfg):
    y = b.y.to_numpy(int); folds = b.fold_id.to_numpy(int); base = b.v11_pred.to_numpy(bool)
    core = [c for c in x.columns if not c.startswith(("cal__", "calx__"))]
    cal = [c for c in x.columns if c.startswith(("cal__", "calx__"))]
    cols = core if cfg["features"] == "core" else core + cal
    final = base.copy(); details = []
    for f in sorted(np.unique(folds)):
        cur = folds == f
        if f == 0:
            details.append({"fold_id": int(f), "mode": "v11_bootstrap"}); continue
        hist = folds < f; ha = hist & base
        if ha.sum() < 20 or len(np.unique(y[ha])) < 2:
            details.append({"fold_id": int(f), "mode": "insufficient_history"}); continue
        ptr, pcur = fit_predict(x.loc[ha, cols], y[ha], x.loc[cur, cols], cfg["C"])
        tp_probs = ptr[y[ha] == 1]
        thr = float(max(0.0, np.quantile(tp_probs, cfg["tp_quantile"]) - cfg["margin"]))
        if cfg["scope"] == "market":
            scope = market[cur] & (~strong[cur])
        elif cfg["scope"] == "nonstrong":
            scope = ~strong[cur]
        else:
            scope = np.ones(cur.sum(), bool)
        cur_base = base[cur].copy()
        consensus = x.loc[cur, "risk_mean"].to_numpy(float)
        veto = cur_base & scope & (pcur < thr) & (consensus < cfg["guard"])
        final[cur] = cur_base & (~veto)
        details.append({
            "fold_id": int(f), "history_alerts": int(ha.sum()), "history_tp": int(y[ha].sum()),
            "threshold": thr, "vetoes": int(veto.sum()), "mode": "verified",
        })
    return final, details


def main():
    b, v, d, x, strong, market = build_features()
    y = b.y.to_numpy(int); folds = b.fold_id.to_numpy(int); base = b.v11_pred.to_numpy(bool)
    bm = v75.metrics(y, base, folds)
    configs = []
    for features, C, tpq, margin, scope, guard in product(
        ["core", "calendar"], [.02, .05, .10, .30], [0.0, .10, .20], [.01, .03],
        ["market", "nonstrong", "any"], [.50, .65, 1.01]
    ):
        configs.append({"features": features, "C": C, "tp_quantile": tpq, "margin": margin, "scope": scope, "guard": guard})

    rows, preds = [], []
    for i, cfg in enumerate(configs):
        p, details = causal_candidate(b, x, strong, market, cfg)
        m = v75.metrics(y, p, folds)
        adopt = bool(
            m["recall"] >= bm["recall"] and m["green_npv"] >= bm["green_npv"] and
            m["precision"] > bm["precision"] and m["f1"] > bm["f1"] and
            m["fp"] < bm["fp"] and m["worst_fold_recall"] >= bm["worst_fold_recall"] and
            m["alert_rate"] <= bm["alert_rate"]
        )
        rows.append({"config_id": i, "config": cfg, "metrics": m, "strictly_dominates_v11": adopt, "details": details})
        preds.append(p)

    def key(r):
        m = r["metrics"]
        return (int(r["strictly_dominates_v11"]), m["f1"], m["precision"], -m["fp"], m["recall"], m["green_npv"], -m["alert_rate"])

    sel = max(rows, key=key); pred = preds[sel["config_id"]]; m = sel["metrics"]
    pd.DataFrame({
        "date": b.date, "y": y, "fold_id": folds, "v11_pred": base.astype(int),
        "risk3": b.risk3, "risk14": b.risk14, "calendar_special": x["cal__special_regime"],
        "v13_pred": pred.astype(int),
    }).to_csv(OOF, index=False)

    report = {
        "version": VERSION,
        "status": "DEVELOPMENT_BEST" if sel["strictly_dominates_v11"] else "EXPERIMENTAL_V11_REMAINS_BEST",
        "scientific_boundary": "V13 uses deterministic historical Saudi calendar covariates known independently of merchant outcomes. Each verifier fold is fitted only to earlier V11 alerts. Configuration selection remains development evidence on previously used folds; external real Saudi merchant validation is required.",
        "calendar_feature_count": int(sum(c.startswith(("cal__", "calx__")) for c in x.columns)),
        "candidate_count": len(rows), "v11": bm, "selected": sel,
        "top_candidates": sorted(rows, key=key, reverse=True)[:10], "red_supported": False,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Sales Sentinel V13 — Saudi Calendar Regime Verifier", "",
        f"- Status: **{report['status']}**",
        f"- Calendar features: **{report['calendar_feature_count']}**",
        f"- Candidates: **{len(rows)}**", f"- Selected: **{sel['config']}**", "",
        f"- Precision: V11 **{bm['precision']:.2%}** -> V13 **{m['precision']:.2%}**",
        f"- Recall: V11 **{bm['recall']:.2%}** -> V13 **{m['recall']:.2%}**",
        f"- F1: V11 **{bm['f1']:.2%}** -> V13 **{m['f1']:.2%}**",
        f"- NPV: V11 **{bm['green_npv']:.2%}** -> V13 **{m['green_npv']:.2%}**",
        f"- Alert rate: V11 **{bm['alert_rate']:.2%}** -> V13 **{m['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",
        f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",
        f"- Strictly dominates V11: **{sel['strictly_dominates_v11']}**", "",
        "Scientific boundary: development evidence only; fresh external Saudi merchant validation remains pending.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
