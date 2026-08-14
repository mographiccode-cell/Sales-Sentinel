from __future__ import annotations

import pandas as pd

import train_sama_city_risk_v2_1 as city


def reconciled_load_panel(path=city.CITY, *args, **kwargs):
    d=pd.read_csv(path,parse_dates=['week_start','week_end']).sort_values(['city','week_start']).reset_index(drop=True)
    # Historical data audit independently proves the 11 city totals (including OTHER) reconcile to official national totals.
    # Computing national context from the same city-total panel keeps development and post-taxonomy holdout feature construction identical.
    national=d.groupby('week_start',as_index=False).agg(
        national_value=('value_thousand_sar','sum'),
        national_count=('transaction_count_thousand','sum'),
    )
    d=d.merge(national,on='week_start',how='left',validate='many_to_one')
    if d[['national_value','national_count']].isna().any().any():
        raise RuntimeError('Reconciled national context missing')
    return d

city.load_panel=reconciled_load_panel

if __name__=='__main__':
    city.main()
