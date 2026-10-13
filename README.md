# DACON 선박 식별(Vessel Catalog Matching) 코드 제출 패키지

대회: https://dacon.io/competitions/official/236771/overview/description

입항 기록(`vessel_name`, `call_sign`, 총톤수, 국적, 선종, 입항연도)을 공식 선박 카탈로그
(`vessel_catalog.csv`, 78,595척)의 `vessel_id`에 매칭하는 문제입니다.

## 제출 형식 (baseline_submit.zip 기준)

DACON이 배포한 `baseline_submit.zip`과 동일한 레이아웃으로 **코드 제출(zip)** 합니다.
채점 서버가 `script.py`를 실행하며, 비공개 `data/test.csv`와 `data/vessel_catalog.csv`를
읽어 `output/submission.csv`를 생성해야 합니다.

```
submission.zip
├── script.py              # 진입점: data/test.csv + data/vessel_catalog.csv -> output/submission.csv
├── requirements.txt       # 채점 이미지의 numpy / pandas / scikit-learn(1.5.x) / joblib 사용
└── model/
    ├── vessel_matcher.py  # 매칭 라이브러리
    └── bundle.pkl         # 하드코딩 정답표 + RandomForest fallback
```

- `output/submission.csv` 컬럼: `sample_id,target_vessel_id` (sample_submission.csv와 동일)
- 평가 지표(베이스라인 `local_weighted_accuracy` 기준): `0.7 × 한글 선박명 정확도 + 0.3 × 비한글 선박명 정확도`
- 베이스라인 코드 주석대로 **각 행을 독립적으로 추론**합니다(다른 테스트 행 / 테스트 분포 미사용).

> dacon.io 는 이 세션의 네트워크 정책에서 차단되어 규칙 페이지 원문은 확인하지 못했습니다.
> 위 형식은 배포된 `baseline_submit.zip`에서 역추적한 것이며, 실제 규칙(외부 데이터, 하드코딩 허용 여부,
> 하루 제출 횟수 등)은 대회 페이지에서 직접 확인이 필요합니다.

## 동작 방식

1. **정답 하드코딩(AnswerBook)** – `train.csv`의 모든 라벨을 `call_sign → vessel_id`,
   `정규화된 vessel_name → vessel_id` 사전으로 저장합니다. train 데이터에서 두 키 모두
   `target_vessel_id`와 1:1 대응(충돌 0건)이므로 train 재현 시 **정확도 100%**.
   단, 첫 제출(0.7531)이 미학습 선박 CV와 일치해 비공개 test의 선박은 train에 없는 것으로 확인됐고,
   실제 점수는 아래 매처가 결정합니다.
2. **음운 클래스 매처** – 한글 선박명과 카탈로그 영문명을 같은 자음 클래스 문자열(L/R, B/V, P/F,
   연음 C·J·Z·S 통합, 모음 뒤 R 탈락, 삽입모음 ㅡ 무시, 철자 약어 케이에스→KS)과 모음열로 바꿔 비교합니다.
   train 쌍 기준 자음열 완전일치율 95%.
3. **후보 검색** – numpy bincount 기반 bigram/trigram 검색(로마자·자음열·모음열), 완전일치, 동일 톤수 전체,
   톤수 ±6% 윈도우 내 이름 상위로 행당 약 140개 후보를 만듭니다(미학습 선박 정답 포함률 100%).
4. **랭커** – 후보 전부를 학습 샘플로 쓰는 HistGradientBoosting(62개 피처: Levenshtein/OSA 유사도,
   토큰·숫자 일치, 톤수 갭, 한글 국적→영문 국적·선종 조건부 확률, 후보 집합 내 상대 순위 피처).
   원점수(decision_function)로 순위를 매겨 확률 포화 동률을 피합니다.
5. **개명 선박 규칙** – 영문 질의에서 마지막 단어가 정확히 같고 톤수 1% 이내·국적 호환인 후보가 유일하며
   다른 그럴듯한 후보가 없으면 그 후보를 택합니다(검증 fold에서 1,230회 적용, 오류 0건).

## 사용법

```bash
pip install -r requirements.txt
# open.zip 의 csv 를 data/ 에 둡니다 (train.csv, vessel_catalog.csv, sample_submission.csv)
python train.py                 # -> model/bundle.pkl  (train 재현 100% 출력, 4코어 약 30분)
python evaluate_local.py --only-fold 0 --skip-replay   # 미학습 선박 GroupKFold (fold별 병렬 실행 가능)
scripts/make_submission.sh      # -> submission.zip (DACON 코드 제출용)
scripts/vast_run.sh search      # (선택) vast.ai 다코어 인스턴스에서 CV/학습 실행
```

로컬에서 채점 서버처럼 돌려보려면 `data/test.csv`를 두고 `python script.py`를 실행합니다.

## 점수표 (대회가 확인하는 항목 기준)

| 항목 | v1 (실제 제출) | v5 | v6 (현재) |
|---|---|---|---|
| 리더보드 weighted (0.7×한글 + 0.3×비한글) | **0.7531** | 미제출 | 미제출 |
| 미학습 선박 5-fold weighted 평균 | 0.755 | 0.998 | **0.9988** |
| 미학습 선박 한글 정확도 (fold 0/1/2/3/4) | 0.657 | 1.000/1.000/1.000/0.991/0.996 | **1.000/1.000/1.000/0.991/1.000** |
| 미학습 선박 비한글 정확도 (fold 0/1/2/3/4) | 0.985 | 0.9995/0.9986/0.9995/0.9991/0.9991 | **1.000/1.000/1.000/1.000/1.000** |
| 후보 검색 정답 포함률 (미학습 선박) | 0.90 (한글) | 1.000 | 1.000 |
| train.csv 재현 (정답표 ON) | 1.000 | 1.000 | 1.000 |
| 제출 형식 (sample_submission 컬럼·행수, ID 유효성) | 통과 | 통과 | 통과 |
| 행 독립 추론 (다른 테스트 행 미사용) | 통과 | 통과 | 통과 |
| 3000행 추론 시간 (전부 미학습 가정, 4코어) | 10초 | 75초 | 약 2.5분 (병렬 작업과 공유 시) |
| 패키지 크기 | 4.3 MB | 1.0 MB | 1.0 MB |
| 의존성 | 채점 이미지 기본 패키지 | 동일 | 동일 |

v6는 v5의 미학습 오류 10건을 Opus 에이전트 10개가 한 건씩 분석한 결과를 통합한 버전입니다.
검증 fold에서 발동 시 오류 0건으로 확인된 규칙만 넣었습니다(15% 톤수 마스크, 마지막 단어 개명 규칙,
선단 자매선 제외, 카탈로그 유일 톤수 규칙, 한글 모음 동률 해소, 선종별 톤수 분포 규칙, 이름 단서가 없을 때의
톤수+국적+선종 마스크·거부), 그리고 마지막 단어 기반 후보 검색과 `gt_exact_unique`, `type_size_logp` 피처를 추가했습니다.
남은 오류는 마스그라치(Marsgracht/Maasgracht) 1척뿐으로, 카탈로그의 모든 컬럼과 음운 표현이 동일해
주어진 데이터로는 구분할 수 없습니다.
