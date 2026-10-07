"""Data-driven SK hynix report text and explicitly post-hoc case descriptions."""
import re
import numpy as np

def populate(r):
 e=r['experiments'][0];c=np.array(r['daily_closes']);y=np.array(r['actual_returns']);used=np.array(e['used']);p=np.array(e['predictions'])
 pairs=[]
 for i,u in enumerate(used):
  if not u:continue
  q=r['test_queries'][i];a=c[q:q+21]/c[q]-1
  for j,sim in zip(e['neighbors'][i],e['similarities'][i]):
   b=c[j:j+21]/c[j]-1
   pairs.append({'qi':i,'q':q,'j':j,'cosine':sim,'path_mae_pp':float(np.abs(a-b).mean()*100),'correlation':float(np.corrcoef(a,b)[0,1]),'actual_return':float(a[-1]),'historical_return':float(b[-1]),'same_direction':bool(np.sign(a[-1])==np.sign(b[-1]))})
 same=[a for a in pairs if a['same_direction']]
 opposite=[a for a in pairs if not a['same_direction']]
 r['case_examples']={'description':'Post-hoc illustration among historical pairs actually retrieved by the validation-selected configuration; not a selection rule known at forecast time.',
  'pair_count':len(pairs),'same_direction_pair_count':len(same),
  'similar':max(same,key=lambda a:a['correlation']) if same else None,
  'opposite':max(opposite,key=lambda a:a['path_mae_pp']) if opposite else None,
  'selection_rule':'Same-direction pair with highest future-path correlation; opposite-direction pair with largest future-path MAE. Both use observed future only to explain results, never for model selection.'}
 r['matched_direction']={'n':int(used.sum()),'model_correct':int((np.sign(p[used])==np.sign(y[used])).sum()),'mean_correct':int((np.sign(r['baselines']['mean']['prediction'])==np.sign(y[used])).sum())}
 matched_y=y[used];matched_ok=np.sign(p[used])==np.sign(matched_y)
 r['matched_direction'].update({'actual_up':int((matched_y>0).sum()),'actual_down':int((matched_y<0).sum()),'always_down_correct':int((matched_y<0).sum()),'balanced_accuracy':float((matched_ok[matched_y>0].mean()+matched_ok[matched_y<0].mean())/2),'always_down_status':'Post-hoc diagnostic of class imbalance, not selected strategy or a claim of future performance.'})
 r['search_summary']={'found_days':sum(n>0 for n in e['counts']),'no_candidate_days':sum(n==0 for n in e['counts']),'one_or_two_candidate_days':sum(0<n<3 for n in e['counts']),'forecast_used_days':int(used.sum())}
 wrong=[]
 for i in np.flatnonzero((p>0)&(y<0)):
  q=r['test_queries'][i]
  wrong.append({'qi':int(i),'signal_date':r['daily_dates'][q],'outcome_date':r['daily_dates'][q+20],'prediction':float(p[i]),'actual_return':float(y[i]),'used_pattern':bool(used[i]),'neighbor_count':e['counts'][i]})
 r['wrong_up_cases']={'all_up_forecasts':int((p>0).sum()),'pattern_up_forecasts':int(((p>0)&used).sum()),'pattern_up_but_down':sum(a['used_pattern'] for a in wrong),'all':wrong,'pattern_cases':[a for a in wrong if a['used_pattern']]}
 r['validation_better_count']=sum(a['validation']['mean_annual_mae_pp']<a['validation']['mean_annual_baseline_mae_pp'] for a in r['experiments'])

def adapt_html(html,r):
 e=r['experiments'][0];m=e['test_metrics'];b=r['baselines']['mean']['metrics'];v=e['validation'];cfg=r['selected'];nm=e['nonoverlapping_metrics'];nb=r['baselines']['mean']['nonoverlapping_metrics'];bs=r['bootstrap'];md=r['matched_direction'];l=r['latest'];dq=r['data_quality'];s=r['provider_sensitivity'];pct=lambda x:f'{x*100:.1f}%';signed=lambda x:f'{x*100:+.2f}%'
 body,script=html.split('<script id="experiment-data"',1)
 def section(heading,content):
  nonlocal body
  pattern=r'<section(?:\s[^>]*)?>.*?</section>'
  found=False
  def sub(match):
   nonlocal found
   if heading in match.group(0):found=True;return content
   return match.group(0)
  body=re.sub(pattern,sub,body,flags=re.S)
  if not found:raise ValueError('Missing section '+heading)
 body=re.sub(r'<header>.*?</header>',f'''<header><div class="eyebrow">000660 · PRICE PATTERN EXPERIMENT · 2026-10-08 보고</div><h1>닮은 과거 차트로<br>SK하이닉스의 한 달 뒤를 살펴보기</h1><p class="lead"><b>닮은 과거 사례가 있었고, 같은 방향으로 이어진 경우도 있었습니다.</b> 다만 다른 방향으로 간 사례도 있고, 수익률 크기를 정확히 맞히는 능력은 제한적이었습니다. 삼성전자와 똑같은 48조건으로 비교했습니다.</p><p class="muted">SK하이닉스 하나 · 다음 20거래일 · 2026년은 조건 선택에서 제외 · 차트 검색과 예측 성능을 구분</p></header>''',body,flags=re.S)
 cards=[('선택 조건의 2026 MAE',f'{m["mae_pp"]:.2f}%p','평균 절대 수익률 오차'),('과거 평균 기준 MAE',f'{b["mae_pp"]:.2f}%p',f'개선 {bs["estimate_pp"]:.2f}%p'),('유사 과거를 찾은 날',f'{r["search_summary"]["found_days"]} / {m["n"]}',f'그중 {m["matched_n"]}일은 사례 3개 이상'),('패턴 예측의 방향 적중',f'{100*md["model_correct"]/md["n"]:.1f}%',f'패턴 사용 {md["n"]}일 중 {md["model_correct"]}일')]
 cardhtml=''.join(f'<div class="card"><span class="caption">{a}</span><span class="value">{b}</span><small>{c}</small></div>' for a,b,c in cards)
 section('먼저 볼 결과',f'''<section><h2>먼저 볼 결과</h2><div class="cards">{cardhtml}</div><p class="notice"><b>2022~2025년 검증으로 선택한 조건은 {cfg['lookback']}일 · 유사도 98% 초과 · 평균입니다.</b> 검증 MAE는 {v['mean_annual_mae_pp']:.3f}%p, 과거 평균은 {v['mean_annual_baseline_mae_pp']:.3f}%p입니다. 48조건 중 평균 기준보다 나은 조건은 {r['validation_better_count']}개였지만 개선 폭은 작았습니다. 삼성전자의 선택 조건을 강제로 적용한 것이 아니라, 동일한 선택 규칙을 적용했습니다.</p><p>2026년 하락일 적중은 <b>{pct(m['down_recall'])}</b>, 상승·하락 균형 적중은 <b>{pct(m['balanced_sign_accuracy'])}</b>입니다. 항상 상승을 말한 결과와 같지는 않습니다. 다만 전체 방향 적중에는 평균으로 대체한 날도 포함됩니다. 실제 패턴을 사용한 {md['n']}일만 보면 모델은 {md['model_correct']}일({100*md['model_correct']/md['n']:.1f}%), 과거 평균은 {md['mean_correct']}일({100*md['mean_correct']/md['n']:.1f}%)에 방향이 맞았습니다.</p><p>20일 간격으로 겹치지 않는 첫날 기준 {nm['n']}건의 MAE는 조건 {nm['mae_pp']:.2f}%p, 평균 {nb['mae_pp']:.2f}%p입니다. 일별 MAE 개선의 20일 블록 부트스트랩 95% 구간은 <b>{bs['interval95_pp'][0]:.2f}~{bs['interval95_pp'][1]:.2f}%p</b>로 0을 포함합니다. 수익률 오차가 안정적으로 개선됐다고 확정할 근거는 부족합니다.</p><p class="muted">삼성전자 조사에서 정한 48조건과 선택 규칙을 하이닉스 가격을 받기 전에 고정했습니다. 2026년 하이닉스 결과는 조건 선택에 사용하지 않았습니다. 같은 시험 시기와 방법을 공유하는 후속 조사이며, 독립적인 실전 검증은 아닙니다. 166개의 일별 목표는 서로 겹칩니다.</p></section>''')
 body=body.replace('다만 전체 방향 적중에는 평균으로 대체한 날도 포함됩니다.',f'전체 166일의 방향 적중은 {pct(m["sign_accuracy"])}이지만 과거 평균으로 대체한 122일도 포함합니다. 이 수치는 위의 패턴 예측 적중률과 시험 대상이 다릅니다.')
 imbalance=f'''<p class="notice"><b>61.4%만으로 좋은 예측이라고 볼 수는 없습니다.</b> 패턴으로 예측한 {md['n']}일의 실제 결과는 상승 {md['actual_up']}일·하락 {md['actual_down']}일이었습니다. 이 날짜들에서 전부 “하락”을 가정하면 {md['always_down_correct']}/{md['n']} = <b>{100*md['always_down_correct']/md['n']:.1f}%</b>가 맞습니다. 상승일과 하락일을 같은 비중으로 평가한 모델 적중률은 <b>{100*md['balanced_accuracy']:.1f}%</b>입니다. 모두 하락 비교는 결과 분포를 설명하는 사후 진단이며, 미래에도 쓸 수 있다고 검증한 전략은 아닙니다.</p>'''
 body=body.replace('<h2>먼저 볼 결과</h2><div class="cards">'+cardhtml+'</div>', '<h2>먼저 볼 결과</h2><div class="cards">'+cardhtml+'</div>'+imbalance)
 body=body.replace('<th>방향 적중</th><th>균형 적중</th>', '<th>전체 166일 방향</th><th>전체 166일 균형</th>')
 examples=[]
 for key,title in [('similar','같은 하락 방향으로 이어진 사례'),('opposite','닮았지만 다른 방향으로 이어진 사례')]:
  a=r['case_examples'][key]
  if a is None:continue
  q=a['q'];j=a['j'];dates=r['daily_dates'];w=cfg['lookback']
  examples.append(f'''<div class="step"><h3>{title}</h3><p><b>{dates[q]}</b>까지 최근 {w}거래일 차트가 <b>{dates[j]}</b>까지의 과거 {w}거래일 차트와 유사도 <b>{a['cosine']*100:.2f}%</b>였습니다.</p><p>과거의 그 뒤 20거래일은 <b>{signed(a['historical_return'])}</b>, 이번 실제 결과는 <b>{signed(a['actual_return'])}</b>였습니다. 이번 결과일까지는 {dates[q+20]}입니다.</p><p>중간 수익률 경로의 상관은 {a['correlation']:.2f}, 평균 간격은 {a['path_mae_pp']:.2f}%p였습니다. {'같은 방향으로 끝났지만 중간 경로까지 동일하지는 않았습니다.' if key=='similar' else '높은 입력 유사도만으로 다음 움직임이 같다고 볼 수 없습니다.'}</p><button id="example-{key}">이 사례를 그래프로 보기</button></div>''')
 casehtml='<section><h2>“과거에 이런 패턴이 있었어?”</h2><p><b>있었습니다.</b> 아래는 선택 조건에서 실제로 검색된 과거 차트입니다. 닮은 모양이 같은 방향으로 이어진 사례와 반대로 이어진 사례를 나란히 볼 수 있습니다.</p><div class="grid2">'+''.join(examples)+'</div><p class="caption">이 두 예시는 미래 결과를 확인한 뒤 설명용으로 골랐습니다. 당시에는 어느 과거 사례를 따라갈지 알 수 없었습니다. 같은 방향 예시는 이후 경로 상관이 가장 높은 쌍, 반대 방향 예시는 이후 경로 오차가 가장 큰 쌍입니다. 원래 모델은 검색된 여러 사례의 결과를 합쳐 예측했습니다.</p></section>'
 casehtml=casehtml.replace('<p><b>있었습니다.</b>',f'<p>선택한 60일·98% 초과 기준으로는 166일 중 <b>{r["search_summary"]["found_days"]}일</b>에 과거 후보가 하나 이상 있었고, <b>{r["search_summary"]["no_candidate_days"]}일</b>에는 없었습니다. {r["search_summary"]["one_or_two_candidate_days"]}일은 후보 1~2개라서 검색에는 성공했지만, 예측용 최소 3개 규칙에는 못 미쳤습니다.</p><p><b>있었습니다.</b>')
 wu=r['wrong_up_cases']
 rows=''.join(f'''<tr><td><button data-wrong-qi="{a['qi']}">{a['signal_date']} 보기</button></td><td>{a['outcome_date']}</td><td>{signed(a['prediction'])}</td><td>{signed(a['actual_return'])}</td><td>{a['neighbor_count']}</td></tr>''' for a in wu['pattern_cases'])
 denominators=f'''<div class="tablewrap"><table><thead><tr><th>어떤 날짜 전체를 기준으로?</th><th>상승 예측→실제 하락 / 전체</th><th>비율</th></tr></thead><tbody><tr><td>전체 시험 (평균 대체 포함)</td><td>{len(wu['all'])} / {m['n']}</td><td>{len(wu['all'])/m['n']*100:.1f}%</td></tr><tr><td>실제 패턴으로 예측한 날만</td><td>{wu['pattern_up_but_down']} / {md['n']}</td><td>{wu['pattern_up_but_down']/md['n']*100:.1f}%</td></tr><tr><td>패턴으로 상승을 예측한 날만</td><td>{wu['pattern_up_but_down']} / {wu['pattern_up_forecasts']}</td><td>{wu['pattern_up_but_down']/wu['pattern_up_forecasts']*100:.1f}%</td></tr></tbody></table></div>'''
 wronghtml=f'''<section><h2>상승을 예측했는데 실제로 하락한 경우만</h2><p><b>전체 {m['n']}일 중 {len(wu['all'])}일, {len(wu['all'])/m['n']*100:.1f}%입니다.</b> 이 중 실제 패턴 예측은 {wu['pattern_up_but_down']}건, 평균 대체 예측은 {len(wu['all'])-wu['pattern_up_but_down']}건입니다. 분모를 무엇으로 잡느냐에 따라 아래처럼 비율이 달라집니다. 이 비율은 요청한 오류 유형만 세며, 하락을 예측했는데 상승한 경우는 포함하지 않습니다.</p>{denominators}<p><b>패턴으로 상승을 예측한 {wu['pattern_up_forecasts']}번 중 {wu['pattern_up_but_down']}번은 실제로 하락했습니다.</b> 상승 예측이 맞은 것은 {wu['pattern_up_forecasts']-wu['pattern_up_but_down']}번, 즉 {(wu['pattern_up_forecasts']-wu['pattern_up_but_down'])/wu['pattern_up_forecasts']*100:.1f}%였습니다. 아래는 실제 패턴 예측이 반대로 하락한 경우만 모은 표입니다.</p><div class="tablewrap"><table><thead><tr><th>예측 기준일 / 보기</th><th>20거래일 결과일</th><th>상승 예측</th><th>실제 하락</th><th>과거 사례 수</th></tr></thead><tbody>{rows}</tbody></table></div><p class="caption">7일의 기준일이 가까워 같은 하락 구간을 겹쳐 관측합니다. 서로 독립적인 7번의 사건이 아닙니다. <a href="up_predicted_but_down.csv" download>平均 대체 포함 전체 해당 날짜 CSV</a></p></section>'''
 body=body.replace('<section id="interactive">',casehtml+wronghtml+'<section id="interactive">',1)
 lateststatus=f'독립 후보는 {l["count"]}개입니다. '+(f'출력 {signed(l["return"])}는 사례들을 합친 값입니다.' if l['used_analogs'] else f'출력 {signed(l["return"])}는 과거 평균으로 대체한 값이며 패턴 예측이 아닙니다.')
 section('최근 자료를 넣으면?',f'''<section><h2>최근 자료를 넣으면?</h2><p><b>{l['date']} 종가 기준 선택 조건:</b> {lateststatus}</p><p class="muted">네이버 스냅샷 종가 {int(l['close']):,}원. 다음 20거래일 결과는 아직 관측되지 않았습니다. 현재 차트에서 후보가 없다면 “이 기준으로는 없다”고 답하면 됩니다. 평균 대체값은 매일 예측을 비교하기 위한 규칙이며, 과거 사례 검색의 답이 아닙니다.</p></section>''')
 # The copied template's data section includes figures that must use this stock's snapshot.
 body=body.replace('그중 2026년은 14일입니다.',f'그중 2026년은 {dq["crosscheck"]["different_closes_2026"]}일입니다.')
 body=body.replace('55,534개 검색 사례',f'{r["verification"]["audited_neighbor_episodes"]:,}개 검색 사례')
 body=body.replace('<a href="../">기존 실험실</a>','<a href="../pattern-samsung-20d/">삼성전자 같은 실험</a> · <a href="../">기존 실험실</a>')
 body=body.replace('<a href="source.json">가격 출처</a>','<a href="source.json">가격 출처</a><a href="naver_raw.txt" download>네이버 원응답</a>')
 body=body.replace('<div class="grid2"><div><h3>입력 차트의 모양</h3>', '<p class="caption">위의 사례 보기 버튼을 누르면 설명에 나온 과거 구간을 초록색으로 강조합니다. 파란색은 이번 실제 자료입니다.</p><div class="grid2"><div><h3>입력 차트의 모양</h3>')
 html=body+'<script id="experiment-data"'+script
 html=html.replace('let ci=0,qi=0; const colors=', 'let ci=0,qi=0,highlighted=null; const colors=')
 html=html.replace("</b><br>${used?", "</b><br><b>${nn.length?'이 기준으로 과거 패턴: 있음':'이 기준으로 과거 패턴: 없음'}</b><br>${used?")
 html=html.replace("nn=e.neighbors[qi];$('config')", "nn=e.neighbors[qi];if(!nn.includes(highlighted))highlighted=null;$('config')")
 html=html.replace('color:colors[i%colors.length],width:1', "color:j===highlighted?'#077968':colors[i%colors.length],width:j===highlighted?2.8:1")
 html=html.replace("D.daily_dates[j-w+1]+'~'+D.daily_dates[j]", "(j===highlighted?'[예시 과거] ':'')+D.daily_dates[j-w+1]+'~'+D.daily_dates[j]")
 handlers='''
for(const key of ['similar','opposite']){const btn=document.getElementById('example-'+key),ex=D.case_examples[key];if(btn&&ex)btn.addEventListener('click',()=>{ci=0;qi=ex.qi;highlighted=ex.j;render();document.getElementById('interactive').scrollIntoView({behavior:'smooth',block:'start'})});}
for(const btn of document.querySelectorAll('button[data-wrong-qi]'))btn.addEventListener('click',()=>{ci=0;qi=Number(btn.dataset.wrongQi);highlighted=null;render();document.getElementById('interactive').scrollIntoView({behavior:'smooth',block:'start'})});
'''
 pos=html.rfind('</script>');html=html[:pos]+handlers+html[pos:]
 return html

def make_report(r):
 e=r['experiments'][0];m=e['test_metrics'];b=r['baselines']['mean']['metrics'];v=e['validation'];cfg=r['selected'];bs=r['bootstrap'];md=r['matched_direction'];l=r['latest'];a=r['case_examples']['similar'];z=r['case_examples']['opposite'];dates=r['daily_dates']
 return f'''# SK하이닉스 유사 패턴 48조건 실험

삼성전자와 동일한 48조건, 다음 20거래일 누적 가격 수익률. 조건은 하이닉스 자료 수집 전에 고정하고 2022~2025년으로 선택했다. 2026년 결과는 선택에 쓰지 않았다.

- 선택: {cfg['lookback']}일, 코사인 유사도 .98 초과, 평균. 검증 MAE {v['mean_annual_mae_pp']:.6f}%p vs 과거 평균 {v['mean_annual_baseline_mae_pp']:.6f}%p. 평균보다 나은 조건 {r['validation_better_count']}/48.
- 2026년 {m['n']}일: MAE {m['mae_pp']:.6f}%p vs 평균 {b['mae_pp']:.6f}%p. 패턴 사용 {m['matched_n']}일, 평균 대체 {m['fallback_n']}일.
- 과거 후보 하나 이상 찾음 {r['search_summary']['found_days']}일, 후보 없음 {r['search_summary']['no_candidate_days']}일. 후보 1~2개 {r['search_summary']['one_or_two_candidate_days']}일은 검색 성공이지만 예측용 3개 조건 미달이다.
- 방향 적중 {m['sign_accuracy']*100:.2f}% vs 항상 상승 {b['sign_accuracy']*100:.2f}%. 균형 적중 {m['balanced_sign_accuracy']*100:.2f}%, 하락 적중 {m['down_recall']*100:.2f}%.
- 패턴 사용 날짜만: 방향 적중 {md['model_correct']}/{md['n']}, 동일 날짜 과거 평균 {md['mean_correct']}/{md['n']}.
- 그 {md['n']}일의 실제 결과는 상승 {md['actual_up']}일·하락 {md['actual_down']}일. 모두 하락 가정 시 {100*md['always_down_correct']/md['n']:.2f}%가 맞는다. 모델의 패턴 사용 날짜 균형 적중은 {md['balanced_accuracy']*100:.2f}%. 모두 하락 비교는 사후 분포 설명이며 사전 선택한 전략이나 미래 성과 주장이 아니다.
- 패턴 상승 예측 {r['wrong_up_cases']['pattern_up_forecasts']}번 중 실제 하락 {r['wrong_up_cases']['pattern_up_but_down']}번. 해당 날짜는 2026-07-02,03,06,07,08,09,10이고 목표 기간이 겹친다. 평균 대체 포함 전체 상승 예측→실제 하락 {len(r['wrong_up_cases']['all'])}건도 CSV로 제공한다.
- 상승 예측→실제 하락 비율: 전체 시험 {len(r['wrong_up_cases']['all'])}/{m['n']} = {len(r['wrong_up_cases']['all'])/m['n']*100:.2f}%, 실제 패턴 예측 전체 {r['wrong_up_cases']['pattern_up_but_down']}/{md['n']} = {r['wrong_up_cases']['pattern_up_but_down']/md['n']*100:.2f}%, 패턴 상승 예측만 {r['wrong_up_cases']['pattern_up_but_down']}/{r['wrong_up_cases']['pattern_up_forecasts']} = {r['wrong_up_cases']['pattern_up_but_down']/r['wrong_up_cases']['pattern_up_forecasts']*100:.2f}%. 다른 오류 유형은 포함하지 않는다.
- MAE 개선의 20일 블록 부트스트랩 95% 구간 [{bs['interval95_pp'][0]:.6f}, {bs['interval95_pp'][1]:.6f}]%p. 0 포함, 선택 불확실성 전체를 포함하지 않은 탐색적 점검.

같은 방향 예시: {dates[a['q']]}의 최근 {cfg['lookback']}거래일 vs {dates[a['j']]}까지의 과거 구간, 유사도 {a['cosine']*100:.2f}%. 이후 20일 과거 {a['historical_return']*100:+.2f}%, 실제 {a['actual_return']*100:+.2f}%. 경로 상관 {a['correlation']:.3f}, 경로 MAE {a['path_mae_pp']:.3f}%p. 중간 경로가 동일하다는 뜻은 아니다.

반대 방향 예시: {dates[z['q']]} vs {dates[z['j']]} 구간, 유사도 {z['cosine']*100:.2f}%. 이후 과거 {z['historical_return']*100:+.2f}%, 실제 {z['actual_return']*100:+.2f}%.

두 사례는 결과 확인 후 설명용으로 선택했으며 미래 예측 규칙으로 쓰지 않았다. 비슷한 과거를 찾는 기능과 향후 정확한 수익률 예측은 구분한다.

최신 {l['date']}: 후보 {l['count']}개, 패턴 사용 {l['used_analogs']}, 출력 {l['return']*100:+.3f}%. 미래 결과 미관측.

자료는 Naver 2019-01-02~2026-10-07 실제 일별 1905행, 합성/보간 없음. Yahoo 원종가로 고정 조건 민감도 점검도 제공. 제공처 차이, 배당 제외, 가격 조정 미확인, 한 부분 연도 시험과 목표 기간 중복 한계가 있다. 거래비용 후 실전 수익을 검증한 것이 아니다.

index.html에서 무작위 날짜, 같은 방향/반대 방향 예시, 모든 조건을 비교할 수 있다. 원자료 CSV, 날짜별 예측, 선택 기록과 재현 코드를 제공한다.
'''
