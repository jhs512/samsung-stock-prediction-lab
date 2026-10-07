# 재현 방법

Python 3.14, NumPy 2.5.3, pandas 3.0.6으로 계산했다.

보고서 폴더 전체를 내려받은 뒤 이 reproduce 폴더에서:

```
python -m pip install -r requirements.txt
python run.py
python provider_sensitivity.py
python build_report.py
```

부모 폴더의 고정 daily_prices.csv, source.json, data_quality.json, yahoo_crosscheck.json, experiment_plan.json을 사용한다. 가격을 새로 다운로드하지 않으며 현재 웹 자료로 대체하지 않는다. run.py는 선택 기록과 결과를 다시 저장하므로 원본 다운로드 파일을 보관한 사본에서 실행한다. 선택 조건은 2022~2025년으로만 정하며 2026년 결과로 선택하지 않는다. provider_sensitivity.py는 이미 선택한 조건을 Yahoo 원종가에 적용할 뿐 재선택하지 않는다.

원본 코드: https://github.com/teddylee777/stock-pattern
미래 검색, 중복 사례, 목표 기간을 수정한 연구용 비교 실험이며 원본의 완전한 복제는 아니다.
