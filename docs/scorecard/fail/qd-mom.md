# 전략 스코어카드: QD-Mom

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | QD-Mom (`qd-mom`) |
| **유형** | Quarter-Day TSMOM |
| **타임프레임** | 6H |
| **상태** | `폐기 (Gate 1 FAIL)` |
| **Best Asset** | DOGE/USDT (Sharpe -0.51) |
| **경제적 논거** | 이전 6H session return이 다음 session return을 양(+)으로 예측. Late-informed trader의 정보 흡수 지연 |

---

## 성과 요약 (6년, 2020-2025, 6H TF)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| 1 | DOGE/USDT | -0.51 | -21.6% | -82.8% | 2870 | 0.92 | -6089% | 0.05 |
| 2 | ETH/USDT | -1.18 | -29.1% | -88.8% | 2798 | 0.87 | -2265% | 0.02 |
| 3 | BNB/USDT | -1.53 | -35.8% | -94.7% | 2782 | 0.87 | -6240% | 0.03 |
| 4 | SOL/USDT | -1.56 | -34.3% | -92.8% | 2458 | 0.80 | -4369% | 0.02 |
| 5 | BTC/USDT | -2.10 | -43.2% | -96.9% | 2698 | 0.77 | -1220% | 0.05 |

---

## Gate 진행 현황

```
G0A 아이디어  [PASS] 25/30점
G0B 코드검증  [PASS] C1-C7 전항목 PASS
G1 백테스트  [FAIL] 전 에셋 Sharpe 음수 (-0.51 ~ -2.10), MDD 82~97%
```

### Gate 1 상세

- **즉시 폐기 조건 해당**: 전 에셋 Sharpe 음수 + 전 에셋 MDD > 50%
- **과다 거래**: 연 410~480건 (6H bar의 ~70%에서 거래). 비용 drag 연 45~53%
- **Quarter-day autocorrelation 부재**: 6H return → 다음 6H return 예측이 크립토에서 작동하지 않음. FX/Equity 시장의 intraday momentum anomaly가 24/7 크립토에서 구조적으로 부재
- **Vol filter (volume median) 무효**: volume 필터가 거래 빈도를 충분히 감소시키지 못함 (여전히 연 400건+)

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-11 | G0A | PASS | 25/30점 |
| 2026-02-11 | G0B | PASS | C1-C7 전항목 PASS |
| 2026-02-11 | G1 | FAIL | 전 에셋 Sharpe 음수 (-0.51~-2.10), MDD 82~97%. 6H autocorrelation edge 부재 |
