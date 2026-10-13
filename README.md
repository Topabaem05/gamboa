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
2. **Fallback 랭커** – 정답표에 없는 행은 카탈로그 후보 검색(이름 n-gram, 한글 로마자 변환/자음 골격,
   총톤수 근접) 후 RandomForest 쌍 분류기로 1척을 고릅니다. train에서 학습한
   한글 국적→영문 국적, 한글 선종→카탈로그 선종, 한글 토큰→영문 토큰 사전을 피처로 씁니다.

## 사용법

```bash
pip install -r requirements.txt
# open.zip 의 csv 를 data/ 에 둡니다 (train.csv, vessel_catalog.csv, sample_submission.csv)
python train.py                 # -> model/bundle.pkl  (train 재현 100% 출력)
python evaluate_local.py        # train 재현(정답표 ON) + 미학습 선박 GroupKFold(정답표 OFF)
scripts/make_submission.sh      # -> submission.zip (DACON 코드 제출용)
```

로컬에서 채점 서버처럼 돌려보려면 `data/test.csv`를 두고 `python script.py`를 실행합니다.

## 로컬 결과

| 구분 | accuracy | 한글 | 비한글 | weighted |
|---|---|---|---|---|
| train.csv 재현 (정답표 ON) | 1.000 | 1.000 | 1.000 | **1.000** |
| DACON 베이스라인(rf.pkl) train 재현 | – | 0.326 | 0.984 | 0.523 |
| 미학습 선박 GroupKFold 5-fold (정답표 OFF, fallback만) | 0.953 | 0.657 | 0.985 | 0.755 |

주의: 비공개 test의 선박이 train에 없으면 정답표가 적중하지 않으므로 실제 점수는 fallback 성능에 좌우됩니다.
