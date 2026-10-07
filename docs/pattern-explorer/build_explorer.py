"""Build exhaustive cosine search browser from frozen real OHLC snapshots."""
import csv,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
data={}
for key,name in [('samsung','삼성전자'),('hynix','SK하이닉스')]:
    with (ROOT.parent/f'pattern-{key}-20d'/'daily_prices.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    data[key]={'name':name,'dates':[r['date'] for r in rows], 'close':[float(r['close']) for r in rows], 'open':[float(r['open']) for r in rows]}
payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
(ROOT/'prices.json').write_text(payload,encoding='utf-8')
(ROOT/'index.html').write_text((ROOT/'template.html').read_text(encoding='utf-8').replace('__DATA__',payload),encoding='utf-8')
stats={}
with (ROOT/'all-2026-100d-matches.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['stock','query_start','query_end','past_start','past_end','similarity','past_return_20d','actual_return_20d'])
    for key,s in data.items():
        prices=np.array(s['close']);dates=s['dates'];qs=[i for i,d in enumerate(dates) if d.startswith('2026-')];ends=np.array([i for i in range(99,len(dates)-20) if dates[i+20]<'2026-01-01'])
        windows=np.array([prices[i-99:i+1] for i in ends]);vectors=windows-windows.min(axis=1)[:,None];norms=np.linalg.norm(vectors,axis=1);vectors=np.divide(vectors,norms[:,None],out=np.zeros_like(vectors),where=norms[:,None]!=0)
        pairs=found=0
        for q in qs:
            a=prices[q-99:q+1];a=a-a.min();n=np.linalg.norm(a);sims=vectors@(a/n) if n else np.zeros(len(ends));chosen=np.flatnonzero((sims>.98)&(ends+20<q-99))
            found+=bool(len(chosen));pairs+=len(chosen)
            for z in chosen:
                j=int(ends[z]);writer.writerow([s['name'],dates[q-99],dates[q],dates[j-99],dates[j],float(sims[z]),float(prices[j+20]/prices[j]-1),float(prices[q+20]/prices[q]-1) if q+20<len(prices) else ''])
        stats[key]={'query_dates':len(qs),'found_dates':found,'matching_windows':pairs}
(ROOT/'search_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
print(stats)
print('Built',ROOT/'index.html')
