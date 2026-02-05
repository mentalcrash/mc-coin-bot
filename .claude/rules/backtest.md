---
paths:
  - "src/backtest/**"
  - "src/cli/backtest.py"
---

# Backtest Rules

## Engine

- **VectorBT 기반:** 벡터화된 백테스트 엔진
- **비용 모델:** Binance 수수료 적용 (현물 0.1%, 선물 0.02%)

## Walk-Forward Analysis (WFA)

파라미터 최적화 구현 시 필수:

```
┌─────────────┬──────────────┐
│ Train (IS)  │ Test (OOS)   │  Window 1
├─────────────┼──────────────┤
│             │ Train (IS)   │  Test (OOS) │  Window 2
│             ├──────────────┼─────────────┤
│             │              │ Train (IS)  │  Test (OOS) │  Window 3
└─────────────┴──────────────┴─────────────┴─────────────┘
```

- **In-Sample (IS):** 파라미터 최적화
- **Out-of-Sample (OOS):** 성과 검증
- **단일 분할 금지:** 롤링 윈도우 필수

## Survivorship Bias Prevention

여러 심볼 백테스트 시:

```python
# ❌ Bad (survivorship bias)
symbols = get_current_listed_symbols()  # 현재 상장 목록

# ✅ Good (point-in-time)
symbols = get_symbols_at_date(backtest_date)  # 해당 시점 상장 목록
```

## Lookahead Bias Prevention

시그널 생성 시 **반드시** `.shift(1)`:

```python
# ❌ Bad (uses current bar's close to decide on current bar)
signal = (df['close'] > df['sma_20']).astype(int)

# ✅ Good (uses previous bar's data)
signal = (df['close'].shift(1) > df['sma_20'].shift(1)).astype(int)
```

## Cost Model

```python
# src/portfolio/cost_model.py
class CostModel:
    # Binance Spot: 0.1%
    spot_fee: Decimal = Decimal("0.001")

    # Binance Futures (Maker): 0.02%
    futures_maker_fee: Decimal = Decimal("0.0002")

    # Binance Futures (Taker): 0.04%
    futures_taker_fee: Decimal = Decimal("0.0004")
```

## Report Generation

```bash
# Generate QuantStats HTML report
python -m src.cli.backtest run tsmom BTC/USDT \
    --start 2024-01-01 \
    --end 2025-12-31 \
    --report
```

## Performance Metrics

핵심 지표:
- Sharpe Ratio (연율화)
- Max Drawdown
- Win Rate
- Profit Factor
- Calmar Ratio
