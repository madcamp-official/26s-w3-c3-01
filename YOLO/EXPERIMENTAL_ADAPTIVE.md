# Experimental Adaptive Grid

기존 `HybridConfig`와 기본 UI를 변경하지 않는 별도 실험판이다.

## 설정

- Root Grid: 4 × 2 (셀 크기 710 × 710mm)
- Level 1: 8 × 4
- Level 2: 16 × 8
- 분할 최소 부모 표본: 3
- 권장 부모 표본: 6
- 분할 최소 자식 표본: 1
- 최소 성공률 차이: 25%p
- Grid prior: 16 / 10 / 7
- 좌표 불확실성: 공별 25mm, 쿠션 인접 시 최소 35mm, 32회 Monte Carlo

확률 모델은 경기 단위 홀드아웃에서 CatBoost보다 안정적이었던 Logistic Hybrid를
사용한다. 최종 확률에는 동일 홀드아웃 예측으로 적합한 Platt calibration을 적용한다.
현재 503건에서는 Level 1 분할 부모 46개와 Level 2 분할 부모 1개가 생성되며,
174건이 Level 1, 3건이 Level 2를 사용한다. 자식 1건부터 분할을 허용하는 대신 강한
prior와 최종 확률 보정으로 단일 관측의 영향을 제한한다.

## 실행

PowerShell에서 다음 파일을 실행한다.

```powershell
.\run_experimental_adaptive.ps1
```

또는 직접 실행한다.

```powershell
.\.venv\Scripts\python.exe experimental_adaptive_server.py
```

브라우저 주소는 `http://127.0.0.1:8766`이다. 서버를 종료하려면 실행한 PowerShell에서
`Ctrl+C`를 누른다.
