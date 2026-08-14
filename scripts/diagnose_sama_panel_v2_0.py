from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'sama_pos'/'sama_pos_2020_2025_normalized.csv'
OUT=ROOT/'reports'/'sama_panel_v2_0'/'panel_preparation_diagnosis.json'
OUT.parent.mkdir(parents=True,exist_ok=True)

def main():
    raw=pd.read_csv(DATA,parse_dates=['week_start','week_end'])
    ind=raw['indicator'].astype(str).str.lower()
    vm=ind.str.contains('value')&ind.str.contains('transaction')&~ind.str.contains('change')
    nm=ind.str.contains('number')&ind.str.contains('transaction')&~ind.str.contains('change')
    def sub(mask,name):
        z=raw.loc[mask,['week_start','city','sector','value']].copy()
        z['value']=pd.to_numeric(z['value'],errors='coerce')
        return z.rename(columns={'value':name}).drop_duplicates(['week_start','city','sector'],keep='last')
    v=sub(vm,'value'); n=sub(nm,'count')
    merged=v.merge(n,on=['week_start','city','sector'],how='inner')
    concrete=merged[~merged.city.astype(str).str.strip().str.lower().eq('total') & ~merged.sector.astype(str).str.strip().str.lower().eq('total')].copy()
    positive=concrete.dropna(subset=['value','count']); positive=positive[(positive.value>0)&(positive['count']>0)].copy()
    entity_counts=positive.groupby(['city','sector']).size().sort_values(ascending=False)
    keep=entity_counts[entity_counts>=80]
    p=positive.set_index(['city','sector']); retained=p[p.index.isin(set(keep.index))].reset_index().sort_values(['city','sector','week_start']).reset_index(drop=True)
    if len(retained):
        g=retained.groupby(['city','sector'],sort=False)
        retained['gap']=g.week_start.diff().dt.days
        retained['lag52_value']=g.value.shift(52)
        retained['roll52_value']=g.value.transform(lambda x:x.rolling(52).mean())
        retained['next_week']=g.week_start.shift(-1)
        retained['next_gap']=(retained.next_week-retained.week_start).dt.days
        retained['next_value']=g.value.shift(-1)
        retained['baseline4']=g.value.transform(lambda x:x.rolling(4).mean())
        stage52=retained.dropna(subset=['lag52_value','roll52_value','baseline4','next_value']).copy()
        consecutive=stage52[(stage52.next_gap==7)].copy()
        gap_distribution=retained['gap'].value_counts(dropna=False).head(12).to_dict()
        next_gap_distribution=retained['next_gap'].value_counts(dropna=False).head(12).to_dict()
    else:
        stage52=retained.copy(); consecutive=retained.copy(); gap_distribution={}; next_gap_distribution={}
    result={
        'raw_rows':int(len(raw)),
        'indicator_unique':sorted(raw['indicator'].astype(str).unique().tolist()),
        'value_rows':int(len(v)),
        'number_rows':int(len(n)),
        'merged_value_number_rows':int(len(merged)),
        'concrete_city_sector_rows':int(len(concrete)),
        'positive_concrete_rows':int(len(positive)),
        'entities_total':int(len(entity_counts)),
        'entities_ge_80_rows':int(len(keep)),
        'top_entity_counts':[{'city':str(i[0]),'sector':str(i[1]),'rows':int(c)} for i,c in entity_counts.head(25).items()],
        'retained_rows_ge80':int(len(retained)),
        'rows_after_52week_and_target_nonnull':int(len(stage52)),
        'rows_after_next_gap_eq_7':int(len(consecutive)),
        'gap_distribution':{str(k):int(v) for k,v in gap_distribution.items()},
        'next_gap_distribution':{str(k):int(v) for k,v in next_gap_distribution.items()},
        'date_min':str(positive.week_start.min().date()) if len(positive) else None,
        'date_max':str(positive.week_start.max().date()) if len(positive) else None,
        'cities':sorted(positive.city.astype(str).unique().tolist())[:100],
        'sector_count':int(positive.sector.nunique()) if len(positive) else 0,
    }
    OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
