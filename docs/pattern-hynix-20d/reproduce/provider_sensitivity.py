"""Fixed-condition sensitivity check. Never selects a model on 2026."""
import json
import numpy as np
import pandas as pd
import run as experiment

p=experiment.OUT
r=json.loads((p/'results.json').read_text(encoding='utf-8'))
j=json.loads((p/'yahoo_crosscheck.json').read_text(encoding='utf-8'))['chart']['result'][0]
dates=pd.to_datetime(j['timestamp'],unit='s',utc=True).tz_convert('Asia/Seoul').strftime('%Y-%m-%d')
data=pd.DataFrame({'date':pd.to_datetime(dates),'close':j['indicators']['quote'][0]['close']}).dropna().reset_index(drop=True)
experiment.df=data;experiment.c=data.close.to_numpy(float);experiment.dates=data.date.to_numpy()
experiment.target=np.full(len(data),np.nan)
experiment.target[:-20]=experiment.c[20:]/experiment.c[:-20]-1
experiment.target_date=data.date.shift(-20)
s=experiment.stage(2026)
pred,prob,used,choices=experiment.forecast(s,r['selected'])
base=np.full(len(pred),s['mean'])
result={'purpose':'Provider sensitivity only; original Naver-selected condition is fixed, no retuning',
 'provider':'Yahoo chart raw Close, not adjusted close','condition':r['selected'],
 'rows':len(data),'test_n':len(pred),'test_first':str(data.date.iloc[s['queries'][0]].date()),
 'test_last':str(data.date.iloc[s['queries'][-1]].date()),
 'selected_metrics':experiment.metrics(s['y'],pred,prob,used,base),
 'mean_baseline_metrics':experiment.metrics(s['y'],base,np.full(len(pred),s['prob'])),
 'mae_gain_pp':float((np.abs(s['y']-base)-np.abs(s['y']-pred)).mean()*100),
 'calendar_caveat':'Yahoo lacks three pre-2026 Naver dates. Each provider uses its own observed 20-session horizon; historical windows may differ. No filling or replacement.',
 'selection_changed':False}
(p/'provider_sensitivity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
