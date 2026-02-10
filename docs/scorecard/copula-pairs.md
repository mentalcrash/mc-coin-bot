# 전략 스코어카드: Copula Pairs

> 자동 생성 | 평가 기준: [evaluation-standard.md](../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Copula Pairs (`copula-pairs`) |
| **유형** | 통계적 차익 (Cointegration MR) |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | 미평가 |
| **경제적 논거** | 공적분 관계의 페어 스프레드가 평균회귀하는 구조적 특성 활용 |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 20/30점
G1 백테스트  [PENDING] pair_close 데이터 구성 필요
G2 IS/OOS    [    ]
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 1 차단 사유

- `pair_close` 컬럼이 Silver 데이터에 없음
- 주 자산 + 페어 자산의 OHLCV를 병합하는 데이터 구성 로직 필요
- **다음 단계**: MarketDataService에 pair data loader 추가, 페어 선택 전략 정의

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0 | PASS | 20/30점 |
| 2026-02-10 | G1 | PENDING | pair_close 데이터 부재로 백테스트 불가 |
