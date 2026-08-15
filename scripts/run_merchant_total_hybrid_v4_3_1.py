from __future__ import annotations

import numpy as np
import pandas as pd

import train_merchant_total_hybrid_v4_3 as v

v.VERSION='SALES-SENTINEL-MERCHANT-TOTAL-HYBRID-4.3.1'


def build_fixed():
    d=pd.read_csv(v.DAILY,parse_dates=['date']).sort_values('date').reset_index(drop=True)
    X=v.merchant_features(d)
    cp=v.category_pivot()
    rp=v.rich_pivot()
    q=d[['date']].merge(cp,on='date',how='left',validate='one_to_one')
    q['current_week_start']=v.week_start(q.date)
    q=q.merge(rp,left_on='current_week_start',right_on='available_week_start',how='left',validate='many_to_one')
    q=q.drop(columns=['current_week_start','available_week_start'])
    X=pd.concat([X,q.drop(columns=['date'])],axis=1)
    sales=d.sama_calibrated_net_sales_sar.clip(lower=0).astype(float)
    base=sales.rolling(28,min_periods=28).mean()
    future=sum(sales.shift(-h) for h in range(1,8))
    ratio=future/(7*base.replace(0,np.nan))
    target=(ratio<.8).astype(int)
    X=X.replace([np.inf,-np.inf],np.nan)
    for c in X:
        if 'ratio' in c or 'index' in c:
            X[c]=X[c].fillna(1.)
        else:
            X[c]=X[c].fillna(0.)
    good=(d.date>=d.date.min()+pd.Timedelta(days=56))&ratio.notna()&base.gt(0)
    meta=pd.DataFrame({'date':d.date,'future_ratio':ratio,'target':target}).loc[good].reset_index(drop=True)
    return meta,X.loc[good].reset_index(drop=True)

v.build=build_fixed

if __name__=='__main__':
    v.main()
