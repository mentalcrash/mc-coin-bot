---
paths:
  - "src/eda/**"
  - "src/core/event_bus.py"
  - "src/core/events.py"
---

# EDA (Event-Driven Architecture) Rules

## Event Chain

```
DataFeed → BAR(1m) → CandleAggregator → BAR(TF)
  → StrategyEngine → SIGNAL
  → PortfolioManager → ORDER_REQUEST(validated=False)
  → RiskManager → ORDER_REQUEST(validated=True) or ORDER_REJECTED
  → OMS → ORDER_ACK → Executor → FILL
  → PM(fill) → POSITION_UPDATE + BALANCE_UPDATE
  → RM(equity) → CIRCUIT_BREAKER (if drawdown threshold)
```

## EventBus Rules (Critical)

- **`flush()` 필수**: bar-by-bar 동기 처리 보장. 누락 시 이벤트 순서 깨짐
- **Priority ordering**: 낮은 숫자가 먼저 실행 (`bisect.insort` 기반)
- **Handler isolation**: 한 handler 에러가 다른 handler 차단하지 않음

### Event Drop Policy

| Policy | Events | Behavior |
|--------|--------|----------|
| **DROPPABLE** | BAR, HEARTBEAT, RISK_ALERT | Queue full → drop (non-blocking `put_nowait`) |
| **NEVER_DROP** | SIGNAL, FILL, ORDER_REQUEST, ORDER_ACK, ORDER_REJECTED, POSITION_UPDATE, BALANCE_UPDATE, CIRCUIT_BREAKER | Block until queue space (awaitable `put`) |

- Consecutive drop 10+ → circuit breaker alert

## Deferred Execution (VBT Parity)

- **일반 주문** (price=None): `_pending_orders`에 저장 → 다음 bar open에 체결
- **SL/TS 주문** (price 지정): 즉시 체결 (intrabar)
- **Slip**: BUY `fill_price * (1 + slip)`, SELL `fill_price * (1 - slip)`
- `apply_fills()` → 반드시 `_on_bar()` subscribers보다 **먼저** 호출

## Validated Flag State Machine

```
PM → ORDER_REQUEST(validated=False) → RM checks
  PASS → ORDER_REQUEST(validated=True) → OMS executes
  FAIL → ORDER_REJECTED → logged
```

- **RM**: `if order.validated: return` (재처리 방지)
- **OMS**: `if not order.validated: return` (미검증 실행 방지)

## PortfolioManager Modes

### Batch Mode (multi-asset)

- Signal 수집: `_pending_signals[symbol]` per timestamp
- Timestamp 변경 시 `_flush_signal_batch()` 일괄 처리
- SL/TS: `_deferred_close_targets[symbol] = 0.0` → 다음 batch flush에 포함

### Single Mode

- Signal 즉시 `evaluate_rebalance()` 호출

### Rebalance Threshold

- `abs(target - last_executed_target) >= threshold` 비교
- **last_executed_target** 사용 (current_weight 아님 = VBT parity)
- First entry (0.0 → nonzero): 항상 trigger

### Guards

- `_pending_close`: close 주문 대기 중 → rebalance/SL/TS skip
- `_stopped_this_bar`: SL/TS 발생 bar → same-bar re-entry 방지 (TF bar 변경 시 clear)

## Risk Manager: 4-Layer Pre-Trade Check

1. **Circuit breaker active** → reject all
1. **Aggregate leverage** >= `max_leverage_cap` AND not reducing → reject
1. **Max open positions** AND new position → reject
1. **Single order size** > `max_order_size_usd` AND not reducing → reject

Position reduction (반대 방향): 항상 통과

### System Stop-Loss

- `BalanceUpdateEvent` → drawdown = `1 - equity / peak_equity`
- Peak 업데이트: drawdown check **이후** (`peak = max(peak, equity)`)
- Threshold 초과 시 `CircuitBreakerEvent` → OMS close-all (RM bypass)

## OMS Idempotency

- **Key**: `client_order_id` (format: `{source}-{symbol}-{counter}`)
- **FIFO dedup dict**: 100,000 entries max
- Duplicate → `OrderRejectedEvent(reason="Duplicate order")`

## ATR Caching (Intrabar SL/TS)

- TF bar에서 true range 계산 → `pos.atr_values` 누적 (14-period SMA)
- `_last_tf_atr[symbol]`: intrabar 1m bar에서 재사용
- 조건: `len(pos.atr_values) >= 14` (warmup 미달 시 SL/TS skip)
- Position flip 시 `atr_values = []` 초기화

## Port Protocol

- **DataFeedPort**: `start()`, `stop()` — Backtest: Parquet iteration, Live: WebSocket
- **ExecutorPort**: `execute(order)` → `FillEvent | None`
  - Backtest: deferred fill, Live: Binance API
- Strategy enrichment: features
