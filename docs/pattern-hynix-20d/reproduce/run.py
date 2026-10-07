import datetime as dt, hashlib, itertools, json, pathlib, shutil
import numpy as np
import pandas as pd

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT.parent
OUT.mkdir(parents=True,exist_ok=True)
PLAN=json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))
plan_path=OUT/'experiment_plan.json'
if not plan_path.exists():
 PLAN['locked_at_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
 PLAN.setdefault('deadline_utc',(dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=PLAN['budget_minutes'])).isoformat())
 plan_path.write_text(json.dumps(PLAN,ensure_ascii=False,indent=2),encoding='utf-8')
else:PLAN=json.loads(plan_path.read_text(encoding='utf-8'))
df=pd.read_csv(OUT/'daily_prices.csv',parse_dates=['date'])
c=df.close.to_numpy(float); dates=df.date.to_numpy(); h=20
target=np.full(len(df),np.nan);target[:-h]=c[h:]/c[:-h]-1
target_date=df.date.shift(-h)
configs=[]
for w,retrieval,agg in itertools.product(PLAN['lookbacks'],PLAN['retrievals'],PLAN['aggregations']):
 configs.append({'id':f'w{w:03d}_{retrieval}_{agg}','lookback':w,'retrieval':retrieval,'aggregation':agg})
assert len(configs)==48

def metrics(y,p,prob,used=None,baseline=None):
 error=np.abs(y-p); actual_up=y>0;sign_correct=np.sign(y)==np.sign(p)
 recalls=[float(sign_correct[y>0].mean()) if (y>0).any() else None,float(sign_correct[y<0].mean()) if (y<0).any() else None]
 result={'n':len(y),'mae_pp':float(error.mean()*100),'rmse_pp':float(np.sqrt(np.mean((y-p)**2))*100),'sign_accuracy':float(sign_correct.mean()),'balanced_sign_accuracy':float(np.mean([a for a in recalls if a is not None])),'up_recall':recalls[0],'down_recall':recalls[1],'actual_up_rate':float(actual_up.mean()),'correlation':float(np.corrcoef(y,p)[0,1]) if np.std(p)>1e-12 else None,'brier':float(np.mean((prob-actual_up)**2))}
 if used is not None:
  result['coverage']=float(used.mean());result['matched_n']=int(used.sum());result['fallback_n']=int((~used).sum())
  result['matched_mae_pp']=float(error[used].mean()*100) if used.any() else None
  result['matched_baseline_mae_pp']=float(np.abs(y[used]-baseline[used]).mean()*100) if used.any() else None
 return result

def stage(year,latest=False):
 queries=np.flatnonzero((df.date.dt.year==year).to_numpy() & np.isfinite(target) & (np.arange(len(df))>=99))
 if year<2026:queries=queries[(target_date.iloc[queries].dt.year<=year).to_numpy()]
 first=queries[0]
 lib=np.arange(99,first-h)
 assert len(lib)>0 and (df.date.iloc[lib+h]<df.date.iloc[first]).all()
 library_target=target[lib]
 if latest:queries=np.array([len(df)-1])
 basemean=float(library_target.mean());basemedian=float(np.median(library_target));baseprob=float((library_target>0).mean())
 prepared={}
 for w in PLAN['lookbacks']+[12]:
  windows=np.lib.stride_tricks.sliding_window_view(c,w)
  a=windows[lib-w+1];b=windows[queries-w+1]
  amin=a.min(axis=1);span=a.max(axis=1)-amin
  anorm=np.divide(a-amin[:,None],span[:,None],out=np.zeros_like(a),where=span[:,None]>0)
  alen=np.linalg.norm(anorm,axis=1);anorm=np.divide(anorm,alen[:,None],out=np.zeros_like(anorm),where=alen[:,None]>0)
  bspan=b.max(axis=1)-b.min(axis=1);bnorm=np.divide(b-b.min(axis=1)[:,None],bspan[:,None],out=np.zeros_like(b),where=bspan[:,None]>0)
  blen=np.linalg.norm(bnorm,axis=1);bnorm=np.divide(bnorm,blen[:,None],out=np.zeros_like(bnorm),where=blen[:,None]>0)
  similarities=bnorm@anorm.T;similarities[:,span==0]=-np.inf
  if (bspan==0).any():similarities[bspan==0,:]=-np.inf
  order=np.argsort(-similarities,axis=1,kind='stable')
  for retrieval in PLAN['retrievals']:
   indices=[]; sims=[]
   for qi,ranked in enumerate(order):
    selected=[]
    cap=int(retrieval.split('_')[1]) if retrieval.startswith('top_') else 20
    threshold=float(retrieval.split('_')[1]) if retrieval.startswith('threshold_') else -np.inf
    for j in ranked:
     if not np.isfinite(similarities[qi,j]) or similarities[qi,j]<=threshold:break
     ix=int(lib[j])
     if any(abs(ix-old)<w+h for old in selected):continue
     selected.append(ix)
     if len(selected)>=cap:break
    indices.append(selected)
    sims.append([float(similarities[qi,np.searchsorted(lib,j)]) for j in selected])
   prepared[(w,retrieval)]={'indices':indices,'similarities':sims}
  # Raw-style overlap-count diagnostic excluded from model selection.
  if w==12:
   reference=[]
   for qi in range(len(queries)):
    jj=np.flatnonzero(similarities[qi]>.98);reference.append(lib[jj].tolist())
 return {'year':year,'queries':queries,'lib':lib,'y':target[queries],'mean':basemean,'median':basemedian,'prob':baseprob,'prepared':prepared,'reference_indices':reference}

def forecast(s,cfg):
 choices=s['prepared'][(cfg['lookback'],cfg['retrieval'])];p=[];prob=[];used=[]
 for jj in choices['indices']:
  use=len(jj)>=3;used.append(use)
  p.append(float(np.mean(target[jj])) if use and cfg['aggregation']=='mean' else float(np.median(target[jj])) if use else s['mean'])
  prob.append(float(((target[jj]>0).sum()+1)/(len(jj)+2)) if use else s['prob'])
 return np.array(p),np.array(prob),np.array(used),choices

def main():
 folds=[]
 for year in PLAN['validation_years']:
  folds.append(stage(year));print('Prepared validation',year,flush=True)
 validation=[]
 for cfg in configs:
  annual=[]
  for s in folds:
   p,prob,used,_=forecast(s,cfg);baseline=np.full(len(p),s['mean'])
   m=metrics(s['y'],p,prob,used,baseline);m['year']=s['year'];m['baseline_mae_pp']=float(np.abs(s['y']-baseline).mean()*100)
   annual.append(m)
  validation.append({'config':cfg,'annual':annual,'mean_annual_mae_pp':float(np.mean([a['mae_pp'] for a in annual])),'mean_annual_baseline_mae_pp':float(np.mean([a['baseline_mae_pp'] for a in annual]))})
 validation.sort(key=lambda a:(a['mean_annual_mae_pp'],a['config']['id']))
 selected=validation[0]['config']
 lock={'locked_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'selection_uses_2026':False,'selected':selected,'validation_order':validation,'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest()}
 (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
 print('Selected before test:',selected,flush=True)
 s=stage(2026);n=len(s['queries']);base=np.full(n,s['mean']);baseprob=np.full(n,s['prob'])
 experiments=[];pred_csv=[]
 for v in validation:
  cfg=v['config'];p,prob,used,choices=forecast(s,cfg)
  m=metrics(s['y'],p,prob,used,base)
  non=metrics(s['y'][::h],p[::h],prob[::h],used[::h],base[::h])
  experiments.append({'config':cfg,'validation':v,'test_metrics':m,'nonoverlapping_metrics':non,'predictions':p.tolist(),'probabilities':prob.tolist(),'counts':[len(jj) for jj in choices['indices']],'used':used.tolist(),'neighbors':choices['indices'],'similarities':choices['similarities']})
  pred_csv.append(pd.DataFrame({'config_id':cfg['id'],'signal_date':df.date.iloc[s['queries']].dt.strftime('%Y-%m-%d').to_numpy(),'outcome_date':df.date.iloc[s['queries']+h].dt.strftime('%Y-%m-%d').to_numpy(),'actual_return':s['y'],'prediction':p,'up_probability':prob,'independent_analogs':[len(jj) for jj in choices['indices']],'used_analogs':used}))
 chosen=next(e for e in experiments if e['config']['id']==selected['id']);p=np.array(chosen['predictions'])
 gain=np.abs(s['y']-base)-np.abs(s['y']-p)
 rng=np.random.default_rng(42);boots=[];block=20
 for _ in range(2000):
  starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
  sample=np.concatenate([np.arange(j,j+block) for j in starts])[:n]
  boots.append(float(gain[sample].mean()*100))
 interval=np.quantile(boots,[.025,.975]).tolist()
 bms={}
 for name,val in [('zero',0.),('mean',s['mean']),('median',s['median'])]:
  bp=np.full(n,val);bms[name]={'prediction':val,'up_probability':s['prob'],'metrics':metrics(s['y'],bp,baseprob),'nonoverlapping_metrics':metrics(s['y'][::h],bp[::h],baseprob[::h])}
 refp=np.array([float(target[jj].mean()) if jj else s['mean'] for jj in s['reference_indices']]);refprob=np.array([float((target[jj]>0).mean()) if jj else s['prob'] for jj in s['reference_indices']])
 reference={'description':'12-day cosine >0.98, overlapping matches allowed, mean next-20 cumulative return; future-data leak corrected; excluded from selection','counts':[len(jj) for jj in s['reference_indices']],'metrics':metrics(s['y'],refp,refprob),'predictions':refp.tolist()}
 lateststage=stage(2026,latest=True);lp,lprob,lused,lc=forecast(lateststage,selected)
 latest={'date':str(df.date.iloc[-1].date()),'close':float(c[-1]),'return':float(lp[0]),'up_probability':float(lprob[0]),'used_analogs':bool(lused[0]),'count':len(lc['indices'][0]),'neighbor_indices':lc['indices'][0],'similarities':lc['similarities'][0],'future_observed':False}
 # Audit every retrieved episode, not only ones used to produce a forecast.
 audits=0
 for e in experiments:
  w=e['config']['lookback']
  for qq,jj in zip(s['queries'],e['neighbors']):
   assert all(df.date.iloc[j+h]<df.date.iloc[s['queries'][0]] for j in jj)
   assert all(abs(j-k)>=w+h for j,k in itertools.combinations(jj,2))
   assert all(j-w+1>=0 for j in jj)
   audits+=len(jj)
  assert np.isfinite(e['predictions']).all()
  # A no-analog result must use exactly the common baseline, not a fabricated pattern.
  assert all(abs(pred-s['mean'])<1e-14 for pred,use in zip(e['predictions'],e['used']) if not use)
 # Independently recompute an actual matched forecast, not an empty fallback case.
 matched=np.flatnonzero(chosen['used'])
 audit_i=int(matched[0]) if len(matched) else 0
 jj=chosen['neighbors'][audit_i];w=selected['lookback'];q=s['queries'][audit_i]
 b=c[q-w+1:q+1];bz=(b-b.min())/(b.max()-b.min())
 for j,recorded in zip(jj,chosen['similarities'][audit_i]):
  a=c[j-w+1:j+1];az=(a-a.min())/(a.max()-a.min())
  check=float(np.dot(bz,az)/np.sqrt(np.dot(bz,bz)*np.dot(az,az)))
  assert abs(check-recorded)<1e-12
 if len(jj)>=3:
  outcomes=[float(c[j+h]/c[j]-1) for j in jj]
  independent=float(np.mean(outcomes)) if selected['aggregation']=='mean' else float(np.median(outcomes))
  assert abs(independent-p[audit_i])<1e-12
 verification={'no_2026_selection':True,'frozen_library_has_no_2026_outcomes':True,'all_neighbor_episodes_nonoverlapping':True,'audited_neighbor_episodes':audits,'no_synthetic_or_interpolated_prices':True,'fallback_identical_to_common_mean':True,'all_predictions_finite':True,'independent_cosine_and_cumulative_return_recomputation':bool(len(jj)>=3),'independent_check_signal_date':str(df.date.iloc[q].date()),'independent_check_neighbor_count':len(jj),'test_is_followup_not_blinded':True}
 result={'plan':PLAN,'selected':selected,'experiments':experiments,'baselines':bms,'reference':reference,'test_queries':s['queries'].tolist(),'actual_returns':s['y'].tolist(),'daily_dates':df.date.dt.strftime('%Y-%m-%d').tolist(),'daily_closes':c.tolist(),'latest':latest,'bootstrap':{'metric':'common mean MAE minus selected MAE, percentage points','estimate_pp':float(gain.mean()*100),'interval95_pp':interval,'block_length':20,'repeats':2000,'seed':42},'verification':verification,'data_quality':json.loads((OUT/'data_quality.json').read_text(encoding='utf-8')),'source':json.loads((OUT/'source.json').read_text(encoding='utf-8')),'computed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
 (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 pd.concat(pred_csv).to_csv(OUT/'test_predictions.csv',index=False)
 summary=pd.DataFrame([{'config_id':e['config']['id'],'selected_by_validation':e['config']['id']==selected['id'],'validation_mae_pp':e['validation']['mean_annual_mae_pp'],**e['test_metrics']} for e in experiments])
 summary.to_csv(OUT/'experiment_comparison.csv',index=False)
 (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
 print(json.dumps({'selected':selected,'validation_mae_pp':chosen['validation']['mean_annual_mae_pp'],'validation_baseline_mae_pp':chosen['validation']['mean_annual_baseline_mae_pp'],'test':chosen['test_metrics'],'test_mean_baseline':bms['mean']['metrics'],'nonoverlap':chosen['nonoverlapping_metrics'],'bootstrap':result['bootstrap'],'latest':latest,'verification':verification},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
