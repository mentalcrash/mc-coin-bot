# 전략 스코어카드: Candle-Reject

> 자동 생성 | 평가 기준: `pipeline gates-list`

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Candle-Reject (`candle-reject`) |
| **유형** | Mean Reversion (Candlestick rejection wick reversal) |
| **타임프레임** | 4H (annualization_factor=2190) |
| **상태** | `폐기 (Gate 1 FAIL — 4H 재검증 확정)` |
| **Best Asset** | N/A (전 에셋 Sharpe 음수, 4H 재검증) |
| **경제적 논거** | 긴 꼬리(rejection wick)는 가격 거부를 의미 → 반대 방향이 시장의 진정한 방향 |

---

## 성과 요약 (6년, 2020-2025, 4H 재검증)

### 에셋별 비교 (4H TF, 2026-02-12 재검증)

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Win Rate |
|------|------|--------|------|-----|--------|------|---------|
| 1 | SOL/USDT | -0.71 | -4.7% | -29.6% | 245 | 0.72 | 43.3% |
| 2 | DOGE/USDT | -0.78 | -6.1% | -35.2% | 310 | 0.70 | 45.2% |
| 3 | BNB/USDT | -0.94 | -7.0% | -41.0% | 270 | 0.65 | 41.9% |
| 4 | ETH/USDT | -1.09 | -7.2% | -36.5% | 279 | 0.61 | 42.7% |
| 5 | BTC/USDT | -1.62 | -9.3% | -45.6% | 270 | 0.48 | 39.3% |

**판정**: 즉시 폐기 — 전 에셋 Sharpe 음수 (조건 #2)

### 이전 결과 (1D 왜곡, 참고용)

| 에셋 | Sharpe (1D) | Sharpe (4H) | 변화 |
|------|------------|------------|------|
| BTC/USDT | 0.65 | -1.62 | 급격 악화 |
| ETH/USDT | 0.60 | -1.09 | 급격 악화 |
| SOL/USDT | -0.71 | -0.71 | 동일 |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 24/30점
G0B 코드검증 [PASS] C1-C7 전항목 PASS (2026-02-12 재검증). Warning 1건 (W3 Regime)
G1 백테스트  [FAIL] 4H 재검증 확정. 전 에셋 Sharpe 음수 (-0.71 ~ -1.62)
```

### Gate 1 상세 (4H 재검증, 2026-02-12)

- **즉시 폐기**: 전 에셋 Sharpe 음수 (조건 #2). SOL -0.71 ~ BTC -1.62
- **1D→4H 전환 시 BTC/ETH 급격 악화**: 1D에서 BTC 0.65, ETH 0.60이었으나 4H에서 -1.62, -1.09로 반전. 1D에서의 양의 수익은 annualization_factor 왜곡(6배 연환산)의 인공물
- **거래 수 정상화**: 1D(36~51건) → 4H(245~310건). 충분한 거래가 발생하지만 기대값 음수
- **Rejection wick 반전 전략의 구조적 실패**: 4H candle에서 rejection wick이 반전이 아닌 추세 지속 확인(continuation)으로 작용. Win Rate 39~45%로 random 이하
- **PF 전 에셋 < 1.0**: 0.48~0.72 범위. rejection 시그널의 예측력이 4H에서도 부재

**근본 원인**: Rejection wick은 전통 주식/FX 일봉에서 유효한 반전 시그널이지만, 크립토 4H bar에서는 가격 경로의 일시적 변동일 뿐 진정한 방향 거부를 의미하지 않음. 24/7 크립토 시장에서 wick-based reversal은 noise-dominated

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 24/30점 |
| 2026-02-10 | G0B | PASS | Critical 7항목 결함 0개 |
| 2026-02-11 | G1 | **FAIL** | 1D 왜곡: Best BTC Sharpe 0.65 < 1.0 |
| 2026-02-12 | G1 (4H 재검증) | **FAIL 확정** | 전 에셋 Sharpe 음수 (-0.71 ~ -1.62). 즉시 폐기 조건 #2. Rejection wick reversal 4H에서도 무효 |
