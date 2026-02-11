# 전략 스코어카드: Accel-Skew

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Accel-Skew (`accel-skew`) |
| **유형** | Acceleration + Rolling Skewness |
| **타임프레임** | 12H |
| **상태** | `폐기 (Gate 1 FAIL)` |
| **Best Asset** | DOGE/USDT (Sharpe 0.47) |
| **경제적 논거** | 가격 가속도가 양(+)이고 rolling skewness도 양(+)이면 우상향 테일이 reward로 전환 |

---

## 성과 요약 (6년, 2020-2025, 12H TF)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| 1 | DOGE/USDT | 0.47 | +10.0% | -27.4% | 839 | 1.11 | -5927% | 0.05 |
| 2 | ETH/USDT | 0.12 | +0.3% | -38.5% | 816 | 1.02 | -2159% | -0.01 |
| 3 | BTC/USDT | -0.03 | -2.9% | -36.4% | 785 | 0.99 | -1132% | 0.01 |
| 4 | SOL/USDT | -0.21 | -6.0% | -46.5% | 702 | 0.93 | -4304% | 0.02 |
| 5 | BNB/USDT | -0.41 | -10.2% | -55.1% | 794 | 0.90 | -6226% | -0.00 |

---

## Gate 진행 현황

```
G0A 아이디어  [PASS] 24/30점
G0B 코드검증  [PASS] C1-C7 전항목 PASS
G1 백테스트  [FAIL] Best Sharpe 0.47 < 1.0, Best CAGR +10.0% < 20%
```

### Gate 1 상세

- **Sharpe < 1.0**: Best Asset (DOGE) 0.47. 전 에셋 Sharpe < 1.0
- **CAGR < 20%**: Best +10.0%. 3/5 에셋 음수 CAGR
- **부분 양호 (DOGE)**: MDD -27.4%는 양호하나, Sharpe/CAGR 기준 미달
- **Skewness 필터의 한계**: Rolling skewness가 12H에서 노이즈 과다. Acceleration과 결합해도 alpha 생성 실패
- **에셋간 편차 큼**: DOGE(+10%) vs BNB(-10.2%) = 20%p 격차. DOGE의 밈코인 특성(극단적 skewness)에만 미약한 적합

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-11 | G0A | PASS | 24/30점 |
| 2026-02-11 | G0B | PASS | C1-C7 전항목 PASS |
| 2026-02-11 | G1 | FAIL | Best Sharpe 0.47 < 1.0, CAGR +10.0% < 20%. Skewness 필터 alpha 부재 |
