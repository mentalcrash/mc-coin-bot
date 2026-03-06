---
paths:
  - "src/backtest/**"
  - "src/cli/backtest.py"
---

# Backtest Rules

## Walk-Forward Analysis (WFA)

파라미터 최적화 시 필수:

```
┌─────────────┬──────────────┐
│ Train (IS)  │ Test (OOS)   │  Window 1
├─────────────┼──────────────┤
│             │ Train (IS)   │  Test (OOS) │  Window 2
└─────────────┴──────────────┴─────────────┘
```

- 단일 분할 금지 — 롤링 윈도우 필수

## Survivorship Bias Prevention

```python
# Bad
symbols = get_current_listed_symbols()

# Good (point-in-time)
symbols = get_symbols_at_date(backtest_date)
```

## Cost Model

```python
# Binance Futures (Maker): 0.02%, (Taker): 0.04%
futures_maker_fee: Decimal = Decimal("0.0002")
futures_taker_fee: Decimal = Decimal("0.0004")

# Binance Spot (Maker/Taker): 0.1%
spot_taker_fee: Decimal = Decimal("0.001")
```

## Tiered Validation (`src/backtest/validation/`)

| Level | Method | Pass Criteria |
|-------|--------|---------------|
| **Quick** | IS/OOS Split (70/30) | OOS Sharpe > 0.5, Decay < 50% |
| **Milestone** | Walk-Forward (5-fold) | Consistency > 60%, OOS Sharpe > 0.3 |
| **Final** | CPCV + DSR + PBO | PBO < 0.4, DSR > 1.0 |

```python
from src.backtest.validation import TieredValidator, ValidationLevel
validator = TieredValidator()
result = validator.validate(level=ValidationLevel.QUICK, ...)
```

## Diagnostics & Advisor (`src/backtest/advisor/`)

- **Signal Analyzer**: 시그널 품질, 승률, 보유 기간 분포
- **Regime Analyzer**: 시장 국면별 성과
- **Loss Analyzer**: 손실 구간, 연속 손실 패턴
- **Overfit Analyzer**: 과적합 징후 탐지
