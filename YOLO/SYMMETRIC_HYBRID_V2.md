# Symmetric Hybrid v2

기존 Hybrid 엔진과 Experimental Adaptive Grid를 변경하지 않는 별도 엔진이다.

## 예측 구성

1. 입력 포메이션을 좌우·상하 반전과 O1/O2 교환으로 8개 변환한다.
2. 75-tree, depth-3 CatBoost가 8개를 예측하고 평균한다.
3. Neighbor는 과거 원본 타구별 8개 대칭 거리 중 최솟값 하나만 사용한다.
4. Adaptive Grid는 8개 stateKey 중 canonical key 하나만 사용한다.
5. Neighbor 유효 표본과 Grid 원본 표본에 따라 Model 중심으로 결합한다.
6. 최종 조합의 경기 단위 OOF 예측으로 학습한 Platt calibration을 적용한다.
7. 좌표 오차 범위에서 32회 Monte Carlo를 수행해 평균과 표준편차를 반환한다.

증강된 8개 레이아웃은 CatBoost 학습 행에만 사용한다. Neighbor와 Grid의 표본 수,
분할 조건, 신뢰도에는 원본 타구 한 건을 한 번만 집계한다.

## 현재 모델

- 원본 기록: 503건
- CatBoost: iterations 75, depth 3, learning rate 0.03
- Regularization: l2_leaf_reg 30, random_strength 1
- Class weight: 사용하지 않음
- Grid: 4×2 → 8×4 → 16×8
- 분할 기준: 부모 6건, 자식 2건, 성공률 차이 20%p
- 실제 선택: Level 0 421건, Level 1 76건, Level 2 6건
- 활성 부모: Level 1 11개, Level 2 1개
- 경기 단위 OOF Log Loss: 0.669205
- 경기 단위 OOF Brier Score: 0.238283
- 0.5 기준 OOF 분류 정확도: 57.26%

## 학습

```powershell
.\.venv\Scripts\python.exe train_symmetric_probability_model.py
```

학습 결과는 `outputs/symmetric_hybrid_v2`에 저장된다.

## 웹 실행

```powershell
.\run_symmetric_adaptive.ps1
```

브라우저 주소는 `http://127.0.0.1:8767`이다. 실행 중인 PowerShell에서 `Ctrl+C`를
누르면 종료된다.
