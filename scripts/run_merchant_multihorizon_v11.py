from __future__ import annotations

import numpy as np
import pandas as pd
import train_merchant_multihorizon_v11 as v11


def build_labels_tolerant(d):
    s=pd.read_csv(v11.SECTOR,parse_dates=['TrainingSafeDate'])
    daily=s.groupby('TrainingSafeDate')['sales'].sum().sort_index().to_frame('sales')
    daily=daily.reindex(pd.date_range(daily.index.min(),daily.index.max(),freq='D')).fillna(0.0)
    daily['baseline28']=daily.sales.rolling(28,min_periods=28).mean()
    for h in [3,7,14]:
        daily[f'future{h}']=sum(daily.sales.shift(-k) for k in range(1,h+1))
        daily[f'ratio{h}']=daily[f'future{h}']/(h*daily.baseline28.replace(0,np.nan))
    z=d[['date']].merge(daily[['ratio3','ratio7','ratio14']],left_on='date',right_index=True,how='left')
    diff=np.abs(z.ratio7.to_numpy(float)-d.future_ratio.to_numpy(float))
    max_diff=float(np.nanmax(diff))
    if max_diff>1e-6:
        raise RuntimeError(f'Reconstructed 7-day target mismatch exceeds rounding tolerance: {max_diff}')
    d=d.copy(); d['future3_ratio']=z.ratio3; d['future14_ratio']=z.ratio14
    d['target3']=np.where(d.future3_ratio.notna(),(d.future3_ratio<.85).astype(float),np.nan)
    d['target14']=np.where(d.future14_ratio.notna(),(d.future14_ratio<.85).astype(float),np.nan)
    return d

v11.build_labels=build_labels_tolerant
v11.main()
