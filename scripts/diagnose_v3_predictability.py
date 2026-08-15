from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import train_sama_city_risk_v3 as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'sama_city_v3_predictability'; OUT.mkdir(parents=True,exist_ok=True)

def bm(y,p):
    y=np.asarray(y,int); p=np.asarray(p,bool)
    tp=int(((y==1)&p).sum()); fp=int(((y==0)&p).sum()); fn=int(((y==1)&~p).sum()); tn=int(((y==0)&~p).sum())
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'FPR':fp/max(fp+tn,1),'NPV':tn/max(tn+fn,1)}

def main():
    panel=base.source.reconciled_load_panel(base.HISTORY)
    d,X,P,pc=base.featureize(panel,require_target=True); keep=d.week_start<=base.DEV_END
    d=d.loc[keep].reset_index(drop=True); X=X.loc[keep].reset_index(drop=True); pc=pc.loc[keep].reset_index(drop=True)
    oof,folds=base.build_oof(d,X,pc)
    pos=oof[oof.y.eq(1)].copy(); neg=oof[oof.y.eq(0)].copy()
    out={'rows':len(oof),'positives':int(oof.y.sum()),'positive_rate':float(oof.y.mean()),'ranking':{'ROC_AUC':float(roc_auc_score(oof.y,oof.score)),'PR_AUC':float(average_precision_score(oof.y,oof.score))},'positive_precursor_distribution':{},'negative_precursor_distribution':{},'forecastable_subsets':{},'folds':folds}
    for k in range(0,8):
        out['positive_precursor_distribution'][str(k)]=int(pos.precursor_count.eq(k).sum())
        out['negative_precursor_distribution'][str(k)]=int(neg.precursor_count.eq(k).sum())
    for k in (1,2,3,4):
        fpos=oof.y.eq(1)&oof.precursor_count.ge(k)
        stable=oof.y.eq(0)
        subset=oof[fpos|stable].copy(); yy=np.where(subset.y.eq(1),1,0)
        out['forecastable_subsets'][f'precursor_ge_{k}']={'forecastable_positives':int(fpos.sum()),'share_of_all_declines':float(fpos.sum()/max(int(oof.y.sum()),1)),'rows':int(len(subset)),'ROC_AUC':float(roc_auc_score(yy,subset.score)) if len(set(yy))==2 else None,'PR_AUC':float(average_precision_score(yy,subset.score)) if len(set(yy))==2 else None}
    # Threshold tradeoff using score alone, plus precursor-gated alerts.
    trade=[]
    for t in np.unique(np.quantile(oof.score,np.linspace(0.02,.98,60))):
        for k in (0,1,2,3):
            pred=oof.score.ge(t)&oof.precursor_count.ge(k)
            m=bm(oof.y,pred)
            trade.append({'threshold':float(t),'precursor_min':k,'alert_rate':float(pred.mean()),**m})
    out['best_tradeoffs']={}
    for max_rate in (.15,.25,.35,.50):
        candidates=[x for x in trade if x['alert_rate']<=max_rate]
        best=max(candidates,key=lambda x:(x['recall'],x['precision'])) if candidates else None
        out['best_tradeoffs'][f'alert_rate_le_{int(max_rate*100)}pct']=best
    (OUT/'predictability_report.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
