# Multi-Asset Backtest Comparison (20 Assets)

ACTIVE 전략들의 20개 주요 에셋 VBT 백테스트 비교.
전략별 에셋 적합성을 평가하여 Orchestrator Pod 배정 최적화에 활용.

## Test Configuration

| Item | Value |
|------|-------|
| Period | 2020-01-01 ~ 2026-03-01 (6.2년) |
| Timeframe | 12H |
| Capital | $100,000 per asset |
| Short Mode | DISABLED (Long-only) |
| Trailing Stop | 3.0x ATR |
| System Stop-Loss | 10% |
| Rebalance Threshold | 10% |
| Max Leverage | 2.0x |
| use_intrabar_trailing_stop | false |
| Cost Model | maker 0.02% / taker 0.04% / slippage 0.05% / funding 0.01% / impact 0.02% |

### Test Assets (20)

BTC, ETH, BNB, SOL, DOGE, XRP, ADA, AVAX, LINK, DOT,
POL, ATOM, UNI, NEAR, APT, ARB, OP, FTM, FIL, LTC

> APT: 2022~, ARB: 2023~, OP: 2022~ (짧은 히스토리 주의)

---

## 1. Tri-Channel Multi-Scale Trend

**Test Date**: 2026-03-06
**Script**: `scripts/tri_channel_20assets.py`

### Parameters

```yaml
scale_short: 20
scale_mid: 60
scale_long: 150
bb_std_dev: 2.0
keltner_multiplier: 1.5
entry_threshold: 0.22
vol_window: 30
vol_target: 0.35
annualization_factor: 730.0
```

### Results (Sharpe 순 정렬)

| # | Symbol | Return % | CAGR % | Sharpe | Sortino | Max DD % | Calmar | Win % | PF | Trades |
|---|--------|----------|--------|--------|---------|----------|--------|-------|-----|--------|
| 1 | ETH/USDT | +196.0 | +19.3 | **1.37** | 1.50 | -11.7 | 1.65 | 58.3 | 2.84 | 338 |
| 2 | ADA/USDT | +227.6 | +21.1 | **1.37** | 1.91 | -18.7 | 1.13 | 61.5 | 2.12 | 175 |
| 3 | BTC/USDT | +177.2 | +18.0 | **1.33** | 1.47 | -12.8 | 1.41 | 53.7 | 2.07 | 419 |
| 4 | SOL/USDT | +175.1 | +18.4 | **1.25** | 1.46 | -10.6 | 1.74 | 61.8 | 2.30 | 191 |
| 5 | XRP/USDT | +154.7 | +17.0 | **1.08** | 1.27 | -24.7 | 0.69 | 42.4 | 2.30 | 210 |
| 6 | BNB/USDT | +158.4 | +16.8 | **1.08** | 1.07 | -17.4 | 0.97 | 52.4 | 2.02 | 357 |
| 7 | DOGE/USDT | +511.4 | +34.3 | **1.08** | 1.95 | -36.8 | 0.93 | 45.5 | 2.87 | 188 |
| 8 | AVAX/USDT | +100.3 | +12.3 | 0.88 | 1.07 | -20.1 | 0.61 | 46.7 | 2.03 | 152 |
| 9 | FIL/USDT | +101.6 | +12.5 | 0.77 | 0.91 | -16.2 | 0.77 | 47.7 | 2.01 | 130 |
| 10 | FTM/USDT | +180.2 | +19.1 | 0.71 | 0.74 | -52.5 | 0.36 | 62.5 | 3.63 | 112 |
| 11 | UNI/USDT | +59.3 | +8.1 | 0.66 | 0.89 | -14.1 | 0.57 | 49.0 | 2.17 | 148 |
| 12 | POL/USDT | +53.8 | +7.5 | 0.59 | 0.67 | -17.7 | 0.42 | 45.6 | 1.82 | 180 |
| 13 | APT/USDT | +19.5 | +4.6 | 0.45 | 0.47 | -18.3 | 0.25 | 34.9 | 1.50 | 83 |
| 14 | LINK/USDT | +30.1 | +4.4 | 0.39 | 0.45 | -40.9 | 0.11 | 45.0 | 1.35 | 240 |
| 15 | DOT/USDT | +23.0 | +3.5 | 0.35 | 0.39 | -28.5 | 0.12 | 37.8 | 1.33 | 186 |
| 16 | NEAR/USDT | +16.8 | +2.6 | 0.27 | 0.32 | -22.0 | 0.12 | 47.1 | 1.42 | 136 |
| 17 | ATOM/USDT | +11.0 | +1.8 | 0.21 | 0.20 | -21.4 | 0.08 | 44.0 | 1.16 | 176 |
| 18 | LTC/USDT | -6.8 | -1.2 | -0.02 | -0.02 | -32.3 | 0.04 | 44.3 | 0.96 | 235 |
| 19 | OP/USDT | -9.4 | -2.5 | -0.20 | -0.19 | -29.0 | 0.09 | 37.1 | 0.72 | 97 |
| 20 | ARB/USDT | -14.4 | -5.1 | -0.27 | -0.28 | -27.5 | 0.19 | 37.6 | 0.65 | 85 |

### Summary

- **Avg Sharpe**: 0.67 | **Avg Return**: +108.3%
- **Positive**: 17/20 (85%)
- **Sharpe >= 1.0**: 7개 (ETH, ADA, BTC, SOL, XRP, BNB, DOGE)
- **Best**: ETH (Sharpe 1.37, MDD -11.7%)
- **Worst**: ARB (Sharpe -0.27, -14.4%)

### Observations

- ETH/SOL이 Sharpe와 MDD 양쪽에서 최상위 — 현재 Orchestrator 배정 유지 적합
- ADA가 장기(6년) 기준 Sharpe 1.37로 서프라이즈 — 21년 불장 기여도 높음
- DOGE는 +511% 최고 수익이나 MDD -36.8%로 고위험
- FTM은 수익 +180%이나 MDD -52.5%로 실전 부적합
- L2 토큰(ARB, OP) 일관적 부진 — 짧은 히스토리 + 트렌드 미성숙
- 단기(2024~) vs 장기(2020~) 비교: SOL 0.32→1.25, ADA 0.61→1.37 등 장기에서 크게 개선

---

## 2. Donch-Multi (TODO)

> 테스트 예정

---

## 3. Anchor-Mom (TODO)

> 테스트 예정

---

## 4. MAD-Channel (TODO)

> 테스트 예정

---

## Cross-Strategy Comparison (TODO)

> 전략별 테스트 완료 후 비교 테이블 작성 예정

| Strategy | Avg Sharpe | Sharpe>=1.0 Assets | Best Asset | Worst Asset |
|----------|------------|-------------------|------------|-------------|
| Tri-Channel | 0.67 | 7/20 | ETH (1.37) | ARB (-0.27) |
| Donch-Multi | - | - | - | - |
| Anchor-Mom | - | - | - | - |
| MAD-Channel | - | - | - | - |
