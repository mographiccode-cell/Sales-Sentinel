from __future__ import annotations

import json

import numpy as np
import pandas as pd

import build_saudi_training_safe_v1_2 as pipeline

_original_dumps = pipeline.json.dumps


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _safe_dumps(obj, *args, **kwargs):
    kwargs.setdefault("default", _json_default)
    return _original_dumps(obj, *args, **kwargs)


pipeline.json.dumps = _safe_dumps

if __name__ == "__main__":
    try:
        pipeline.main()
    except Exception:
        if pipeline.AUDIT_JSON.exists():
            audit = json.loads(pipeline.AUDIT_JSON.read_text(encoding="utf-8"))
            print("\n=== SAUDI V1.2 QUALITY GATE DETAILS ===")
            print(json.dumps(audit.get("checks", {}), indent=2))
            print("all_tests_passed =", audit.get("all_tests_passed"))
            print("cleaning =", json.dumps(audit.get("cleaning", {}), indent=2))
            print("localization =", json.dumps(audit.get("localization", {}), indent=2))
            print("calendar_repair =", json.dumps(audit.get("calendar_repair", {}), indent=2))
        raise
