from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import train_sama_sector_decline_v1_9 as v19

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'sales_sentinel_v2_0'
MOD = ROOT / 'models' / 'sales_sentinel_v2_0'
OUT.mkdir(parents=True, exist_ok=True); MOD.mkdir(parents=True, exist_ok=True)
REPORT = OUT / 'development_report.json'
SUMMARY = OUT / 'development_summary.md'
MODEL = MOD / 'sales_sentinel_market_risk_v2_0.joblib'

VERSION = 'SALES-SENTINEL-MARKET-RISK-2.0-FROZEN'
DECLINE = 0.20
DEVELOPMENT_END = pd.Timestamp('2025-06-29')  # target is week ending 2025-07-06; fresh holdout starts after that.
SELECTION_END = pd.Timestamp('2024-12-31')
POLICY_START = pd.Timestamp('2025-01-01')
POLICY_END = DEVELOPMENT_END
SEED = 42

# Production acceptance contract is frozen before the new 2025-2026 holdout is opened.
ACCEPTANCE = {
    'red_precision_min': 0.70,
    'red_false_positive_rate_max': 0.05,
    'alert_recall_min': 0.90,          # RED + AMBER catches at least 90% of material declines
    'green_npv_min': 0.98,             # GREEN should be very safe
    'roc_auc_min': 0.85,
    'pr_auc_min': 0.40,
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(np.clip(-np.asarray(x, float), -35, 35)))


def add_reliability_features(d: pd.DataFrame, X: pd.DataFrame):
    d = d.copy().reset_index(drop=True)
    X = X.copy().reset_index(drop=True)
    # Multiplicative one-step forecast error. shift(1) ensures current outcome is never used.
    ratio = d['actual_next_value'].astype(float) / d['predicted_value_h1'].replace(0, np.nan).astype(float)
    logerr = np.log(ratio.replace([np.inf, -np.inf], np.nan))
    gsector = d.assign(_logerr=logerr).groupby('sector', group_keys=False)
    d['known_logerr_median8'] = gsector['_logerr'].transform(lambda s: s.shift(1).rolling(8, min_periods=4).median())
    d['known_logerr_q10_13'] = gsector['_logerr'].transform(lambda s: s.shift(1).rolling(13, min_periods=6).quantile(.10))
    d['known_logerr_q90_13'] = gsector['_logerr'].transform(lambda s: s.shift(1).rolling(13, min_periods=6).quantile(.90))
    d['corrected_pred_value'] = d['predicted_value_h1'] * np.exp(d['known_logerr_median8'])
    d['corrected_pred_ratio4'] = d['corrected_pred_value'] / d['value_mean4']
    d['upper_pred_ratio90'] = (d['predicted_value_h1'] * np.exp(d['known_logerr_q90_13'])) / d['value_mean4']
    d['lower_pred_ratio10'] = (d['predicted_value_h1'] * np.exp(d['known_logerr_q10_13'])) / d['value_mean4']
    # Count/value disagreement often flags unstable value forecasts.
    d['corrected_market_risk'] = sigmoid(((1.0 - DECLINE) - d['corrected_pred_ratio4']) / 0.055)
    d['upper_bound_market_risk'] = sigmoid(((1.0 - DECLINE) - d['upper_pred_ratio90']) / 0.055)
    extra = [
        'known_logerr_median8','known_logerr_q10_13','known_logerr_q90_13',
        'corrected_pred_ratio4','upper_pred_ratio90','lower_pred_ratio10',
        'corrected_market_risk','upper_bound_market_risk',
    ]
    X = pd.concat([X, d[extra]], axis=1)
    good = X.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    return d.loc[good].reset_index(drop=True), X.loc[good].reset_index(drop=True)


def models_for(pos_weight: float):
    return {
        'LogisticRegression': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight='balanced', C=.45, random_state=SEED),
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=900, max_depth=9, min_samples_leaf=4, max_features=.70,
            class_weight='balanced', random_state=SEED, n_jobs=-1,
        ),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=320, learning_rate=.035, max_leaf_nodes=16, min_samples_leaf=20,
            l2_regularization=4.0, random_state=SEED,
        ),
        'XGBoost': XGBClassifier(
            n_estimators=420, max_depth=3, learning_rate=.025, subsample=.82,
            colsample_bytree=.78, min_child_weight=8, reg_lambda=5.0, reg_alpha=.4,
            scale_pos_weight=pos_weight, eval_metric='logloss', random_state=SEED, n_jobs=-1,
        ),
    }


def time_folds(d: pd.DataFrame):
    starts = pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01','2025-01-01','2025-04-01'])
    folds=[]
    for start in starts:
        end = min(start + pd.DateOffset(months=3) - pd.Timedelta(days=1), DEVELOPMENT_END)
        # one full forecast-week purge: no training target touches validation week.
        train_mask = d.origin_week_start <= start - pd.Timedelta(days=14)
        val_mask = d.origin_week_start.between(start, end)
        if train_mask.sum() < 400 or val_mask.sum() < 100:
            continue
        ytr=d.loc[train_mask,'target']; yv=d.loc[val_mask,'target']
        if ytr.nunique()<2 or yv.nunique()<2:
            continue
        folds.append((start,end,train_mask,val_mask))
    return folds


def oof_scores(d: pd.DataFrame, X: pd.DataFrame):
    names=['LogisticRegression','ExtraTrees','HistGradientBoosting','XGBoost']
    frames=[]; fold_meta=[]
    for start,end,tr,va in time_folds(d):
        ytr=d.loc[tr,'target']; pos=int(ytr.sum()); neg=len(ytr)-pos
        scores={}
        for name, model in models_for(neg/max(pos,1)).items():
            fit=clone(model).fit(X.loc[tr], ytr)
            scores[name]=fit.predict_proba(X.loc[va])[:,1]
        q=d.loc[va].copy()
        # Forecast-rule risks use only forecasts generated from information available at each origin.
        scores['ForecastRaw']=sigmoid(((1.0-DECLINE)-q['pred_value_ratio4'].to_numpy())/.055)
        scores['ForecastCorrected']=q['corrected_market_risk'].to_numpy()
        # Robust fixed ensemble; no learned stacking weights.
        scores['RobustMedian']=np.median(np.column_stack([
            scores['LogisticRegression'], scores['ExtraTrees'], scores['XGBoost'], scores['ForecastCorrected']
        ]),axis=1)
        frame=pd.DataFrame({'idx':q.index,'origin_week_start':q.origin_week_start.to_numpy(),'y':q.target.to_numpy()})
        for name,s in scores.items(): frame[name]=s
        frames.append(frame)
        fold_meta.append({
            'validation_start':str(start.date()),'validation_end':str(end.date()),
            'train_rows':int(tr.sum()),'validation_rows':int(va.sum()),
            'train_positive_rate':float(ytr.mean()),'validation_positive_rate':float(q.target.mean()),
        })
    if not frames: raise RuntimeError('No valid expanding OOF folds')
    return pd.concat(frames,ignore_index=True).sort_values('origin_week_start').reset_index(drop=True), fold_meta


def ranking_metrics(y, score):
    return {
        'ROC_AUC':float(roc_auc_score(y,score)),
        'PR_AUC':float(average_precision_score(y,score)),
    }


def calibration_metrics(y,p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6); y=np.asarray(y,int)
    bins=np.linspace(0,1,11); ece=0.0
    for lo,hi in zip(bins[:-1],bins[1:]):
        mask=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if mask.any(): ece += mask.mean()*abs(p[mask].mean()-y[mask].mean())
    return {'Brier':float(brier_score_loss(y,p)),'ECE_10bin':float(ece)}


def binary_metrics(y,p,t):
    y=np.asarray(y,int); pred=(np.asarray(p)>=t).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {
        'Accuracy':float(accuracy_score(y,pred)),
        'BalancedAccuracy':float(balanced_accuracy_score(y,pred)),
        'Precision':float(precision_score(y,pred,zero_division=0)),
        'Recall':float(recall_score(y,pred,zero_division=0)),
        'F1':float(f1_score(y,pred,zero_division=0)),
        'FalsePositiveRate':float(fp/max(fp+tn,1)),
        'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),
    }


def choose_policy(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float)
    candidates=np.unique(np.r_[np.linspace(.01,.99,197), np.quantile(p,np.linspace(0,1,101))])
    # WATCH: catch declines. Among policies meeting recall/NPV, prefer fewer alerts and better precision.
    watch_options=[]; all_watch=[]
    for t in candidates:
        alert=p>=t; green=~alert
        tp=int(((y==1)&alert).sum()); fn=int(((y==1)&green).sum())
        fp=int(((y==0)&alert).sum()); tn=int(((y==0)&green).sum())
        recall=tp/max(tp+fn,1); precision=tp/max(tp+fp,1); npv=tn/max(tn+fn,1)
        row={'threshold':float(t),'recall':recall,'precision':precision,'green_npv':npv,'alert_rate':float(alert.mean()),'TP':tp,'FP':fp,'FN':fn,'TN':tn}
        all_watch.append(row)
        if recall>=ACCEPTANCE['alert_recall_min'] and npv>=ACCEPTANCE['green_npv_min']:
            watch_options.append(row)
    if watch_options:
        watch=max(watch_options,key=lambda r:(r['precision'],-r['alert_rate'],r['green_npv']))
        watch_fallback=False
    else:
        # Never hide failure: use the best high-recall fallback and mark contract unsatisfied.
        watch=max(all_watch,key=lambda r:(min(r['recall'],.90)+min(r['green_npv'],.98),r['precision'],-r['alert_rate']))
        watch_fallback=True

    # RED: high-confidence alerts only. It must be above WATCH.
    red_options=[]; all_red=[]
    for t in candidates[candidates>=watch['threshold']]:
        m=binary_metrics(y,p,float(t)); positives=m['TP']+m['FP']
        row={'threshold':float(t),**m,'red_alerts':positives,'red_rate':positives/max(len(y),1)}
        all_red.append(row)
        if positives>=8 and m['Precision']>=ACCEPTANCE['red_precision_min'] and m['FalsePositiveRate']<=ACCEPTANCE['red_false_positive_rate_max']:
            red_options.append(row)
    if red_options:
        red=max(red_options,key=lambda r:(r['Recall'],r['Precision'],-r['FalsePositiveRate']))
        red_fallback=False
    else:
        red=max(all_red,key=lambda r:(r['Precision']-2*r['FalsePositiveRate'],r['Recall']))
        red_fallback=True

    return {'watch':watch,'red':red,'watch_fallback':watch_fallback,'red_fallback':red_fallback}


def triage_metrics(y,p,watch_t,red_t):
    y=np.asarray(y,int); p=np.asarray(p,float)
    state=np.where(p>=red_t,'RED',np.where(p>=watch_t,'AMBER','GREEN'))
    red=state=='RED'; alert=state!='GREEN'; green=state=='GREEN'
    def precision(mask): return float(((y==1)&mask).sum()/max(mask.sum(),1))
    red_tp=int(((y==1)&red).sum()); total_pos=int((y==1).sum())
    alert_tp=int(((y==1)&alert).sum()); alert_fp=int(((y==0)&alert).sum())
    red_fp=int(((y==0)&red).sum()); negatives=int((y==0).sum())
    green_tn=int(((y==0)&green).sum()); green_fn=int(((y==1)&green).sum())
    return {
        'rows':int(len(y)),'positive_rate':float(y.mean()),
        'RED':{'coverage':float(red.mean()),'alerts':int(red.sum()),'precision':precision(red),'recall_contribution':red_tp/max(total_pos,1),'false_positive_rate':red_fp/max(negatives,1)},
        'AMBER':{'coverage':float((state=='AMBER').mean()),'rows':int((state=='AMBER').sum()),'positive_rate':precision(state=='AMBER')},
        'GREEN':{'coverage':float(green.mean()),'rows':int(green.sum()),'NPV':green_tn/max(green_tn+green_fn,1),'missed_declines':green_fn,'miss_rate_of_all_declines':green_fn/max(total_pos,1)},
        'RED_plus_AMBER':{'coverage':float(alert.mean()),'precision':alert_tp/max(alert_tp+alert_fp,1),'recall':alert_tp/max(total_pos,1)},
    }


def main():
    d,X,_=v19.prepare()
    d,X=add_reliability_features(d,X)
    d=d[d.origin_week_start<=DEVELOPMENT_END].reset_index(drop=True)
    X=X.loc[:len(d)-1].reset_index(drop=True) if len(X)!=len(d) else X.reset_index(drop=True)
    # Defensive re-alignment after date filtering.
    if len(X)!=len(d): raise RuntimeError('Feature alignment failed')

    oof,folds=oof_scores(d,X)
    selection=oof[oof.origin_week_start<=SELECTION_END].copy()
    policy=oof[oof.origin_week_start.between(POLICY_START,POLICY_END)].copy()
    if len(selection)<300 or len(policy)<200 or selection.y.sum()<20 or policy.y.sum()<15:
        raise RuntimeError(f'Insufficient separated development windows: selection={len(selection)}/{selection.y.sum()}+, policy={len(policy)}/{policy.y.sum()}+')

    candidate_names=[c for c in oof.columns if c not in {'idx','origin_week_start','y'}]
    candidate_metrics={c:ranking_metrics(selection.y,selection[c]) for c in candidate_names}
    # Rare-event model is selected by PR-AUC first, ROC-AUC second. No threshold/test accuracy is used.
    selected=max(candidate_names,key=lambda c:(candidate_metrics[c]['PR_AUC'],candidate_metrics[c]['ROC_AUC']))

    # Platt calibration on the earlier selection window only; late policy window remains separate.
    calibrator=LogisticRegression(C=1.0,max_iter=2000,random_state=SEED).fit(selection[[selected]],selection.y)
    selection_p=calibrator.predict_proba(selection[[selected]])[:,1]
    policy_p=calibrator.predict_proba(policy[[selected]])[:,1]
    policy_contract=choose_policy(policy.y,policy_p)
    watch_t=policy_contract['watch']['threshold']; red_t=policy_contract['red']['threshold']

    dev_triage=triage_metrics(policy.y,policy_p,watch_t,red_t)
    dev_rank=ranking_metrics(policy.y,policy_p); dev_cal=calibration_metrics(policy.y,policy_p)
    dev_gates={
        'red_precision':dev_triage['RED']['precision']>=ACCEPTANCE['red_precision_min'],
        'red_false_positive_rate':dev_triage['RED']['false_positive_rate']<=ACCEPTANCE['red_false_positive_rate_max'],
        'alert_recall':dev_triage['RED_plus_AMBER']['recall']>=ACCEPTANCE['alert_recall_min'],
        'green_npv':dev_triage['GREEN']['NPV']>=ACCEPTANCE['green_npv_min'],
        'roc_auc':dev_rank['ROC_AUC']>=ACCEPTANCE['roc_auc_min'],
        'pr_auc':dev_rank['PR_AUC']>=ACCEPTANCE['pr_auc_min'],
    }

    # Fit chosen base scoring model(s) on all pre-holdout development rows. Fresh holdout is not read here.
    y=d.target
    pos=int(y.sum()); neg=len(y)-pos
    final_models={}
    if selected in models_for(neg/max(pos,1)):
        final_models[selected]=clone(models_for(neg/max(pos,1))[selected]).fit(X,y)
    elif selected=='RobustMedian':
        for name in ('LogisticRegression','ExtraTrees','XGBoost'):
            final_models[name]=clone(models_for(neg/max(pos,1))[name]).fit(X,y)
        # ForecastCorrected is deterministic and contributes at inference.
    # ForecastRaw/ForecastCorrected need no fitted estimator.

    artifact={
        'version':VERSION,'selected_score':selected,'models':final_models,'calibrator':calibrator,
        'features':list(X.columns),'watch_threshold':watch_t,'red_threshold':red_t,
        'decline_threshold':DECLINE,'acceptance_contract':ACCEPTANCE,
        'development_end':str(DEVELOPMENT_END.date()),
        'inference_policy':{'GREEN':f'p < {watch_t:.6f}','AMBER':f'{watch_t:.6f} <= p < {red_t:.6f}','RED':f'p >= {red_t:.6f}'},
    }
    joblib.dump(artifact,MODEL)

    report={
        'version':VERSION,
        'scientific_boundary':'All model choice/calibration/policy thresholds use SAMA data ending 2025-07-06. The newly acquired 2025-2026 PDFs are deliberately not read by this training script.',
        'target':'next official SAMA sector week POS value is >=20% below trailing four completed official weeks mean',
        'development_rows':int(len(d)),'development_positive_rate':float(d.target.mean()),
        'folds':folds,
        'separation':{'candidate_selection_end':str(SELECTION_END.date()),'policy_threshold_window':f'{POLICY_START.date()}..{POLICY_END.date()}','fresh_holdout_not_used':True},
        'candidate_selection_metrics_2024':candidate_metrics,'selected_score':selected,
        'calibration_selection_window':calibration_metrics(selection.y,selection_p),
        'policy_contract':policy_contract,
        'policy_window_ranking':dev_rank,'policy_window_calibration':dev_cal,'policy_window_triage':dev_triage,
        'acceptance_contract':ACCEPTANCE,'development_gates':dev_gates,'development_all_gates_passed':bool(all(dev_gates.values())),
        'leakage_controls':{
            'expanding_time_folds':True,'one_week_purge_before_each_validation_fold':True,
            'future_actual_SAMA_as_feature':False,'forecast_residual_features_shifted_before_use':True,
            'candidate_selection_and_policy_threshold_windows_separated':True,
            'fresh_2025_2026_holdout_used_for_training_or_thresholds':False,
            'shuffle':False,
        },
    }
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    SUMMARY.write_text(f'''# Sales Sentinel Production v2.0 — Frozen Development\n\n- Selected score: **{selected}**\n- Development rows: **{len(d):,}**\n- Development decline rate: **{d.target.mean():.2%}**\n- Candidate selection: **through 2024-12-31**\n- Policy thresholds: **2025-01-01 through 2025-06-29 only**\n- Fresh 2025-2026 holdout used: **No**\n- RED threshold: **{red_t:.4f}**\n- WATCH threshold: **{watch_t:.4f}**\n- RED precision: **{dev_triage['RED']['precision']:.2%}**\n- RED false-positive rate: **{dev_triage['RED']['false_positive_rate']:.2%}**\n- RED+AMBER recall: **{dev_triage['RED_plus_AMBER']['recall']:.2%}**\n- GREEN NPV: **{dev_triage['GREEN']['NPV']:.2%}**\n- PR-AUC: **{dev_rank['PR_AUC']:.2%}**\n- ROC-AUC: **{dev_rank['ROC_AUC']:.2%}**\n- Brier: **{dev_cal['Brier']:.4f}**\n- Development contract passed: **{report['development_all_gates_passed']}**\n''',encoding='utf-8')
    print(json.dumps({
        'selected':selected,'rows':len(d),'positive_rate':float(d.target.mean()),
        'thresholds':{'watch':watch_t,'red':red_t},'triage':dev_triage,'ranking':dev_rank,'calibration':dev_cal,
        'gates':dev_gates,'all_gates':report['development_all_gates_passed'],
    },indent=2))


if __name__=='__main__':
    main()
