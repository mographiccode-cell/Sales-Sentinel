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
    pipeline.main()
