from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

import train_sama_city_risk_v3 as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'sama_city_v3_4'; MOD=ROOT/'models'/'sama_city_v3_4'
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'development_report.json'; MODEL=MOD/'city_risk_v3_4.joblib'
VERSION='SAMA-CITY-RISK-3.4-STRUCTURAL-HYBRID'
CORE=[
    'value_below_short_trend','count_below_short_trend',
    'value_negative_slope','count_negative_slope',
    'value_2w_drop','count_2w_drop',
]
CONTRACT={
    'roc_auc_min':.88,'pr_auc_min':.50,
    'red_precision_min':.70,'red_fpr_max':.015,'min_red_alerts':8,'worst_fold_red_fpr_max':.04,
    'alert_recall_min':.90,'green_npv_min':.99,'alert_precision_min':.18,
    'alert_rate_max':.30,'green_coverage_min':.70,
    'core6_positive_recall_min':.95,
    'min_recall_on_folds_with_5plus_positives':.60,'median_positive_fold_recall_min':.75,
}


def bm(y,p):
    y=np.asarray(y,int); p=np.asarray(p,bool)
    tp=int(((y==1)&p).sum()); fp=int(((y==0)&p).sum()); fn=int(((y==1)&~p).sum()); tn=int(((y==0)&~p).sum())
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'FPR':fp/max(fp+tn,1),'NPV':tn/max(tn+fn,1)}


def build_oof(d,X,P,pc):
    rows=[]; fold_meta=[]
    core_count=P[CORE].sum(axis=1).astype(int)
    for fid,(st,en,tr,va) in enumerate(base.folds(d)):
        ytr=d.loc[tr,'target']
        q=d.loc[va,['week_start','city','target']].rename(columns={'target':'y'}).copy()
        q['fold_id']=fid; q['precursor_count']=pc.loc[va].to_numpy(); q['core_count']=core_count.loc[va].to_numpy()
        for name,factory in base.model_factories().items():
            m=base.fit_one(clone(factory),X.loc[tr],ytr); q[name]=m.predict_proba(X.loc[va])[:,1]
        names=list(base.model_factories()); q['score']=q[names].mean(axis=1); q['agreement']=(q[names]>=.50).sum(axis=1)
        rows.append(q)
        fold_meta.append({'fold_id':fid,'start':str(st.date()),'end':str(en.date()),'train_rows':int(tr.sum()),'validation_rows':int(va.sum()),'train_positive_rate':float(ytr.mean()),'validation_positive_rate':float(q.y.mean()),'validation_positives':int(q.y.sum()),'validation_core6_positives':int((q.y.eq(1)&q.core_count.eq(6)).sum())})
    return pd.concat(rows,ignore_index=True),fold_meta


def eval_watch(q,primary_t,trend_score_floor,core_min=6):
    s=q.score.to_numpy(float); core=q.core_count.to_numpy(int); y=q.y.to_numpy(int)
    structural=(core>=core_min)&(s>=trend_score_floor)
    alert=(s>=primary_t)|structural
    m=bm(y,alert); rate=float(alert.mean()); green=1-rate
    core_pos=(y==1)&(core>=core_min); core_recall=float((alert&core_pos).sum()/max(int(core_pos.sum()),1))
    per=[]
    for fid,z in q.groupby('fold_id'):
        yy=z.y.to_numpy(int); ss=z.score.to_numpy(float); cc=z.core_count.to_numpy(int); aa=(ss>=primary_t)|((cc>=core_min)&(ss>=trend_score_floor)); mm=bm(yy,aa)
        per.append({'fold_id':int(fid),'positives':int(yy.sum()),'core6_positives':int(((yy==1)&(cc>=core_min)).sum()),'recall':mm['recall'],'precision':mm['precision'],'NPV':mm['NPV'],'alert_rate':float(aa.mean())})
    positive=[x for x in per if x['positives']>0]; stable=[x for x in per if x['positives']>=5]
    median=float(np.median([x['recall'] for x in positive])) if positive else 1.; stable_min=min((x['recall'] for x in stable),default=1.)
    ok=(m['recall']>=CONTRACT['alert_recall_min'] and m['NPV']>=CONTRACT['green_npv_min'] and m['precision']>=CONTRACT['alert_precision_min'] and rate<=CONTRACT['alert_rate_max'] and green>=CONTRACT['green_coverage_min'] and core_recall>=CONTRACT['core6_positive_recall_min'] and median>=CONTRACT['median_positive_fold_recall_min'] and stable_min>=CONTRACT['min_recall_on_folds_with_5plus_positives'])
    return ok,m,rate,green,core_recall,median,stable_min,int(structural.sum()),int((structural&y.astype(bool)).sum()),per


def eval_red(q,t):
    red=(q.score.to_numpy(float)>=t)&(q.agreement.to_numpy(int)>=2)&(q.precursor_count.to_numpy(int)>=2); y=q.y.to_numpy(int); m=bm(y,red)
    per=[]
    for fid,z in q.groupby('fold_id'):
        yy=z.y.to_numpy(int); rr=(z.score.to_numpy(float)>=t)&(z.agreement.to_numpy(int)>=2)&(z.precursor_count.to_numpy(int)>=2); mm=bm(yy,rr); per.append({'fold_id':int(fid),'FPR':mm['FPR'],'precision':mm['precision'],'alerts':int(rr.sum())})
    worst=max((x['FPR'] for x in per),default=0.); n=int(red.sum()); ok=n>=CONTRACT['min_red_alerts'] and m['precision']>=CONTRACT['red_precision_min'] and m['FPR']<=CONTRACT['red_fpr_max'] and worst<=CONTRACT['worst_fold_red_fpr_max']
    return ok,m,n,worst,per


def choose(q):
    s=q.score.to_numpy(float)
    primary=np.unique(np.r_[np.quantile(s,np.linspace(.05,.90,90)),np.linspace(.10,.28,70)])
    # Structural floor includes zero: if core-six deterioration alone is historically safe enough, let it stand on its own.
    floor=np.unique(np.r_[0.0,np.quantile(s,np.linspace(.01,.45,70)),np.linspace(.02,.16,60)])
    redc=np.unique(np.r_[np.quantile(s,np.linspace(.90,.999,70)),np.linspace(.55,.95,60)])
    watches=[]; reds=[]
    for pt in primary:
        for ft in floor[floor<=pt]:
            w=eval_watch(q,float(pt),float(ft),6)
            if w[0]: watches.append((float(pt),float(ft))+w[1:])
    for rt in redc:
        r=eval_red(q,float(rt))
        if r[0]: reds.append((float(rt),)+r[1:])
    if not watches: raise RuntimeError('No v3.4 structural WATCH policy meets historical OOF contract')
    if not reds: raise RuntimeError('No v3.4 RED policy meets historical OOF contract')
    best=None
    for w in watches:
        for r in reds:
            if r[0]<w[0]: continue
            # Prefer structural independence (lower floor) only after precision/recall/coverage are satisfied.
            obj=(r[1]['precision'],w[2]['precision'],w[2]['recall'],w[5],w[4],-w[1],-r[1]['FP'],r[0])
            if best is None or obj>best[0]: best=(obj,w,r)
    if best is None: raise RuntimeError('No compatible v3.4 policy pair')
    return best[1],best[2],{'watch_pairs':len(watches),'red_candidates':len(reds),'primary_candidates':len(primary),'trend_floor_candidates':len(floor)}


def main():
    panel=base.source.reconciled_load_panel(base.HISTORY)
    d,X,P,pc=base.featureize(panel,require_target=True); keep=d.week_start<=base.DEV_END
    d=d.loc[keep].reset_index(drop=True); X=X.loc[keep].reset_index(drop=True); P=P.loc[keep].reset_index(drop=True); pc=pc.loc[keep].reset_index(drop=True)
    forbidden=[c for c in X.columns if c.startswith('city_') or 'decline_rate' in c or 'target' in c or 'future' in c]
    if forbidden: raise RuntimeError(f'Forbidden features {forbidden}')
    oof,folds=build_oof(d,X,P,pc); roc=float(roc_auc_score(oof.y,oof.score)); pr=float(average_precision_score(oof.y,oof.score))
    w,r,search=choose(oof)
    pt,ft,wm,rate,gcov,core_rec,median,stable_min,struct_n,struct_tp,wper=w
    rt,rm,rn,rworst,rper=r
    gates={'roc_auc':roc>=CONTRACT['roc_auc_min'],'pr_auc':pr>=CONTRACT['pr_auc_min'],'red_precision':rm['precision']>=CONTRACT['red_precision_min'],'red_fpr':rm['FPR']<=CONTRACT['red_fpr_max'],'alert_recall':wm['recall']>=CONTRACT['alert_recall_min'],'green_npv':wm['NPV']>=CONTRACT['green_npv_min'],'alert_precision':wm['precision']>=CONTRACT['alert_precision_min'],'alert_rate':rate<=CONTRACT['alert_rate_max'],'green_coverage':gcov>=CONTRACT['green_coverage_min'],'core6_positive_recall':core_rec>=CONTRACT['core6_positive_recall_min'],'stable_fold_recall':stable_min>=CONTRACT['min_recall_on_folds_with_5plus_positives'],'median_fold_recall':median>=CONTRACT['median_positive_fold_recall_min']}
    fitted={name:base.fit_one(clone(factory),X,d.target) for name,factory in base.model_factories().items()}
    artifact={'version':VERSION,'models':fitted,'features':list(X.columns),'watch_threshold':pt,'structural_trend_score_floor':ft,'structural_core_columns':CORE,'structural_core_min':6,'red_threshold':rt,'min_precursor_red':2,'ood_profile':base.robust_ood_profile(X),'ood_max_fraction':.15,'expected_cities':sorted(set(panel.city.astype(str))),'development_end':str(base.DEV_END.date()),'target_definition':'next-week city POS value <80% current trailing-4-week mean','scope':'hybrid forecast risk: ML ranking plus independent structural deterioration channel; OOD abstention; no claim for signal-free shocks'}
    joblib.dump(artifact,MODEL)
    report={'version':VERSION,'rows':len(d),'positives':int(d.target.sum()),'positive_rate':float(d.target.mean()),'feature_count':len(X.columns),'ranking':{'ROC_AUC':roc,'PR_AUC':pr},'folds':folds,'search':search,'policy':{'watch_threshold':pt,'structural_trend_score_floor':ft,'structural_core_min':6,'structural_core_columns':CORE,'red_threshold':rt,'RED':rm,'RED_plus_AMBER':wm,'GREEN':{'NPV':wm['NPV'],'coverage':gcov,'missed_declines':wm['FN']},'alert_rate':rate,'alert_precision':wm['precision'],'core6_positive_recall':core_rec,'structural_alert_rows':struct_n,'structural_positive_rows':struct_tp,'median_positive_fold_recall':median,'min_recall_folds_5plus':stable_min,'red_alerts':rn,'worst_fold_red_fpr':rworst},'watch_folds':wper,'red_folds':rper,'contract':CONTRACT,'gates':gates,'all_gates_passed':bool(all(gates.values())),'controls':{'no_city_identity':True,'no_target_prevalence':True,'no_absolute_levels':True,'purged_multi_regime_oof':True,'structural_core_channel_selected_on_historical_oof_only':True,'structural_channel_uses_current_and_past_observations_only':True,'red_consensus_precursor_gate':True,'ood_abstention':True},'scientific_boundary':'No outcome after 2025-06-29 and no semi-synthetic result is read by this trainer.'}
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'development_summary.md').write_text('# Sales Sentinel v3.4 — Structural Hybrid\n\n'+f"- ROC-AUC **{roc:.2%}**, PR-AUC **{pr:.2%}**\n- RED precision **{rm['precision']:.2%}**, FPR **{rm['FPR']:.2%}**\n- RED+AMBER recall **{wm['recall']:.2%}**, precision **{wm['precision']:.2%}**, alert rate **{rate:.2%}**\n- GREEN NPV **{wm['NPV']:.2%}**, coverage **{gcov:.2%}**\n- Core-six positive recall **{core_rec:.2%}**\n- ML watch **{pt:.4f}**, structural score floor **{ft:.4f}**\n- All gates **{report['all_gates_passed']}**\n",encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
