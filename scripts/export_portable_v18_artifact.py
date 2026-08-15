from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone

import train_merchant_category_signals_v7_1 as v71
import evaluate_redsea_portable_v16 as v16
import evaluate_redsea_portable_v16_1 as v161

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports"/"portable_v18_artifact"
OUT.mkdir(parents=True,exist_ok=True)
MODELS=ROOT/"models"; MODELS.mkdir(parents=True,exist_ok=True)
ARTIFACT=MODELS/"sales_sentinel_portable_v18.json.gz"
REPORT=OUT/"artifact_report.json"; SUMMARY=OUT/"summary.md"
V161_REPORT=ROOT/"reports"/"redsea_portable_v16_1"/"diagnostic_report.json"
V162_REPORT=ROOT/"reports"/"redsea_portable_v16_2"/"diagnostic_report.json"
VERSION="SALES-SENTINEL-V18-PORTABLE-EXTRATREES-RUNTIME"


def tree_payload(est):
    t=est.tree_
    vals=t.value[:,0,:]
    denom=vals.sum(axis=1)
    p1=np.divide(vals[:,1],denom,out=np.zeros_like(denom,dtype=float),where=denom!=0)
    return {
        "feature":t.feature.astype(int).tolist(),
        "threshold":[float(x) for x in t.threshold],
        "left":t.children_left.astype(int).tolist(),
        "right":t.children_right.astype(int).tolist(),
        "p1":[float(x) for x in p1],
    }


def pure_predict_one(artifact,row):
    total=0.0
    for tr in artifact["trees"]:
        node=0
        while tr["feature"][node]>=0:
            j=tr["feature"][node]
            node=tr["left"][node] if row[j] <= tr["threshold"][node] else tr["right"][node]
        total+=tr["p1"][node]
    return total/len(artifact["trees"])


def main():
    base161=json.loads(V161_REPORT.read_text(encoding="utf-8"))
    base162=json.loads(V162_REPORT.read_text(encoding="utf-8"))
    d=v16.source_daily()
    meta,X0,_=v16.build_meta_and_features(d)
    X,cols=v161.filter_comparable(X0)
    if len(cols)!=96 or len(meta)!=541:
        raise RuntimeError(f"Unexpected V18 training shape: rows={len(meta)}, features={len(cols)}")
    Xfit,_,prep=v71.fold_prepare(X,X)
    model=clone(v71.factories()["extra_trees"])
    model.fit(Xfit,meta.target.astype(int))

    artifact={
        "version":VERSION,
        "model_type":"pure_python_extra_trees_classifier",
        "scientific_status":"PORTABLE_RUNTIME_ARTIFACT; development-trained; Redsea external evidence is post-open diagnostic, not fresh validation",
        "target_definition":"next 7-day net sales / (7 * trailing 28-day daily mean including prediction date) < 0.85",
        "history_required_days":56,
        "feature_names":cols,
        "preprocessing":prep,
        "classes":[0,1],
        "tree_count":len(model.estimators_),
        "trees":[tree_payload(est) for est in model.estimators_],
        "decision_policy":{
            "static_threshold":float(base161["development"]["threshold"]),
            "causal_percentile_enabled":True,
            "alert_budget":float(base161["development"]["nested_oof_metrics"]["alert_rate"]),
            "percentile_cutoff":float(base162["policy"]["risk_percentile_cutoff"]),
            "percentile_lookback":int(base162["policy"]["selected_lookback_days"]),
            "percentile_warmup":int(base162["policy"]["selected_warmup_rows"]),
        },
        "evidence":{
            "v13_1_development_best":{"tp":52,"fp":65,"fn":11,"tn":253,"precision":0.4444,"recall":0.8254,"f1":0.5778},
            "v16_1_portable_redsea_post_open":base161["redsea"]["metrics"],
            "v16_2_calibrated_redsea_post_open":base162["redsea"]["metrics"],
        },
    }

    # Verify pure JSON tree traversal is numerically identical to sklearn on deterministic sample rows.
    idx=np.unique(np.linspace(0,len(Xfit)-1,41).astype(int))
    skl=model.predict_proba(Xfit.iloc[idx])[:,1]
    pure=np.asarray([pure_predict_one(artifact,Xfit.iloc[i].to_numpy(float)) for i in idx])
    max_err=float(np.max(np.abs(skl-pure)))
    if max_err>1e-12:
        raise RuntimeError(f"Pure runtime parity failed: max abs error {max_err}")

    raw=json.dumps(artifact,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    with ARTIFACT.open("wb") as fh:
        with gzip.GzipFile(fileobj=fh,mode="wb",compresslevel=9,mtime=0) as gz:
            gz.write(raw)
    sha=hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    report={
        "version":VERSION,"artifact":str(ARTIFACT.relative_to(ROOT)),"sha256":sha,
        "compressed_bytes":ARTIFACT.stat().st_size,"uncompressed_json_bytes":len(raw),
        "training_rows":len(meta),"features":len(cols),"trees":len(model.estimators_),
        "pure_python_parity_max_abs_error":max_err,"static_threshold":artifact["decision_policy"]["static_threshold"],
        "alert_budget":artifact["decision_policy"]["alert_budget"],"red_supported":False,
    }
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    lines=["# Sales Sentinel V18 — Portable Runtime Artifact","",f"- Artifact: **{report['artifact']}**",f"- SHA-256: `{sha}`",f"- Training rows / features / trees: **{len(meta)} / {len(cols)} / {len(model.estimators_)}**",f"- Compressed size: **{ARTIFACT.stat().st_size/1024/1024:.2f} MiB**",f"- Pure-Python parity max abs error: **{max_err:.3e}**",f"- Static threshold: **{report['static_threshold']:.3f}**",f"- Adaptive alert budget: **{report['alert_budget']:.2%}**","","This artifact contains the exact trained tree structure and fold-local-style clipping metadata needed for inference without installing scikit-learn at runtime. RED remains disabled."]
    SUMMARY.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(SUMMARY.read_text(encoding="utf-8"))

if __name__=="__main__": main()
