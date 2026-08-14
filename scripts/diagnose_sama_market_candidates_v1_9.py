from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
import train_sama_market_decline_v1_9 as v

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports'/'sama_market_v1_9'/'candidate_test_diagnosis_v1_9.json'

def main():
    d,X,F=v.prepare()
    trm=(d.origin_week_start>='2023-01-01')&(d.origin_week_start<'2023-07-01')
    vam=(d.origin_week_start>='2023-07-01')&(d.origin_week_start<'2024-01-01')
    tem=d.origin_week_start>='2024-01-01'
    tr,va,te=d[trm],d[vam],d[tem]; Xt,Xv,Xe=X[trm],X[vam],X[tem]
    pos=int(tr.target.sum()); neg=len(tr)-pos
    validation={}
    thresholds={}
    for name,model in v.models(neg/max(pos,1)).items():
        fit=clone(model).fit(Xt,tr.target); p=fit.predict_proba(Xv)[:,1]; t,m,s=v.pick_threshold(va.target,p); validation[name]={'threshold':t,'metrics':m,'score':s}; thresholds[name]=t
    fit_d=pd.concat([tr,va],ignore_index=True); fit_x=pd.concat([Xt,Xv],ignore_index=True); fp=int(fit_d.target.sum()); fn=len(fit_d)-fp
    test={}
    for name,model in v.models(fn/max(fp,1)).items():
        fit=clone(model).fit(fit_x,fit_d.target); p=fit.predict_proba(Xe)[:,1]; test[name]=v.metr(te.target,p,thresholds[name])
    out={'thresholds_selected_on_2023_validation_only':thresholds,'validation':validation,'test_2024_2025_using_frozen_thresholds':test,'diagnostic_only_do_not_select_posthoc_on_test':True}
    REPORT.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
