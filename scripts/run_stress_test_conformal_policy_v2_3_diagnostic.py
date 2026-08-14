from __future__ import annotations

import json
import traceback
from pathlib import Path

import stress_test_conformal_policy_v2_3 as stress

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'sama_city_v2_3'; OUT.mkdir(parents=True,exist_ok=True)
STATUS=OUT/'stress_execution_status.json'

if __name__=='__main__':
    try:
        stress.main()
        result={'status':'SUCCESS','exception':None,'traceback':None}
    except Exception as exc:
        result={'status':'FAILURE','exception':repr(exc),'traceback':traceback.format_exc()}
    STATUS.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
