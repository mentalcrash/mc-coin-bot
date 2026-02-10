# 전략 스코어카드: Funding Carry

> 자동 생성 | 평가 기준: [strategy-evaluation-standard.md](../strategy-evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Funding Carry (`funding-carry`) |
| **유형** | 캐리 (Funding rate premium) |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | 미평가 |
| **경제적 논거** | 무기한 선물 펀딩비의 구조적 프리미엄을 캐리 수익으로 수취 |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 25/30점
G1 백테스트  [PENDING] funding_rate 데이터 수집 필요
G2 IS/OOS    [    ]
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 1 차단 사유

- `funding_rate` 컬럼이 Silver 데이터에 없음
- Binance Perpetuals API에서 펀딩비 히스토리 수집 후 Silver layer에 병합 필요
- **다음 단계**: Bronze/Silver 파이프라인에 funding rate fetcher 추가

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0 | PASS | 25/30점 |
| 2026-02-10 | G1 | PENDING | funding_rate 데이터 부재로 백테스트 불가 |
