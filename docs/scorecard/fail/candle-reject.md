# 전략 스코어카드: Candle-Reject

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Candle-Reject (`candle-reject`) |
| **유형** | Mean Reversion (Candlestick rejection wick reversal) |
| **타임프레임** | 4H (annualization_factor=2190, 1D 데이터 백테스트) |
| **상태** | `폐기 (Gate 1 FAIL)` |
| **Best Asset** | BTC/USDT (Sharpe 0.65) |
| **2nd Asset** | ETH/USDT (Sharpe 0.60) |
| **경제적 논거** | 긴 꼬리(rejection wick)는 가격 거부를 의미 → 반대 방향이 시장의 진정한 방향 |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Win Rate |
|------|------|--------|------|-----|--------|------|---------|
| **1** | **BTC/USDT** | **0.65** | +206.1% | -4.5% | 51 | 1.95 | 49.0% |
| 2 | ETH/USDT | 0.60 | +160.8% | -2.8% | 41 | 2.14 | 53.7% |
| 3 | DOGE/USDT | 0.06 | +16.9% | -8.6% | 38 | 1.11 | 55.3% |
| 4 | BNB/USDT | -0.40 | -190.3% | -14.5% | 43 | 0.54 | 44.2% |
| 5 | SOL/USDT | -0.71 | -1070.8% | -56.2% | 36 | 0.23 | 50.0% |

### Best Asset 핵심 지표 (BTC/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.65 | > 1.0 | **FAIL** |
| CAGR | +206.1% | > 20% | PASS |
| MDD | -4.5% | < 40% | PASS |
| Trades | 51 | > 50 | PASS |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 24/30점
G0B 코드검증 [PASS]
G1 백테스트  [FAIL] Best Sharpe 0.65 < 1.0. SOL/BNB 음수 Sharpe
```

### Gate 1 상세

- **Sharpe 미달**: Best Asset BTC 0.65, 2nd ETH 0.60 — 모두 1.0 미달
- **SOL/BNB 역전 패턴**: 추세추종 전략과 반대로 SOL이 최악(-0.71). 반전 전략이 고변동 추세 에셋에서 역효과
- **MDD 양호**: BTC -4.5%, ETH -2.8%로 매우 낮음 — rejection 시그널의 보수적 sizing
- **거래 수 양호**: 36~51건으로 통계적 최소 기준 충족 (SOL 제외)
- **BTC/ETH 제한적 양호**: Sharpe 0.6 수준으로 PASS 기준에 근접하나, SOL/BNB에서 크게 손실
- **CAGR 왜곡**: annualization_factor=2190 (4H) 적용으로 1D 데이터의 수익률이 6배 연환산 → 비현실적 CAGR 수치

**에셋 패턴 분석**:

- BTC/ETH에서 양호: 안정적 시장에서 rejection wick 반전이 효과적
- SOL/BNB에서 FAIL: 고변동/강추세 에셋에서 rejection이 false signal로 전환
- DOGE 중립: 노이즈 환경에서 약한 양수 (edge 부재)

**CTREND 비교**: CTREND Best SOL Sharpe 2.05 대비 candle-reject BTC 0.65 — 3.2배 열등

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 24/30점 |
| 2026-02-10 | G0B | PASS | Critical 7항목 결함 0개 |
| 2026-02-11 | G1 | **FAIL** | Best Sharpe 0.65 < 1.0. 2/5 에셋 음수 Sharpe (SOL -0.71, BNB -0.40). 고변동 에셋에서 rejection reversal 무효 |
