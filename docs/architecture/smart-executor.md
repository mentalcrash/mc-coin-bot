# SmartExecutor

Limit Order 우선 실행기. Decorator 패턴으로 `LiveExecutor`를 래핑하여
일반 주문은 지정가(Limit)를 우선 시도하고, 긴급 주문은 즉시 시장가(Market)로 실행합니다.

---

## 1. 아키텍처

```text
OrderRequestEvent
       │
       ▼
  SmartExecutor
       │
  _classify_urgency()
       │
  ┌────┴────┐
  │         │
URGENT    NORMAL
  │         │
  ▼         ▼
Market   Limit Order
(inner)   ├─ place → poll → fill ✅
          ├─ price deviation → cancel → market fallback
          └─ timeout → cancel → market fallback
```

**핵심 설계**: Decorator 패턴 — `SmartExecutor`는 `LiveExecutor(inner)`를 래핑합니다.
`ExecutorPort` Protocol을 만족하므로 기존 OMS/Runner 코드 변경 없이 교체 가능합니다.

---

## 2. Urgency 분류

`_classify_urgency()`는 모든 주문을 URGENT/NORMAL로 분류합니다.

| 조건 | Urgency | 이유 |
|------|---------|------|
| `order.price is not None` (SL/TS exit) | URGENT | 가격 지정된 방어 주문 |
| `order.target_weight == 0` (포지션 청산) | URGENT | 청산은 즉시 실행 |
| 방향 전환 (롱→숏, 숏→롱) | URGENT | 중간 상태 최소화 |
| API 비정상 (`is_api_healthy == False`) | URGENT | 안전 모드 |
| 동시 limit 수 초과 | URGENT | 과부하 방지 |
| SmartExecutor 비활성화 | URGENT | 기존 동작 유지 |
| 그 외 (신규 진입, 리밸런싱) | NORMAL | Limit 시도 |

---

## 3. Limit Order 라이프사이클

```text
1. fetch_ticker() → bid/ask 조회
2. _compute_limit_price() → spread 안쪽 가격 계산
   BUY:  ask × (1 - offset_bps/10000)
   SELL: bid × (1 + offset_bps/10000)
3. create_order() → 거래소 Limit 주문 배치
4. _monitor_limit_order() → poll 루프
   ├─ 완전 체결 → FillEvent 반환 ✅
   ├─ 가격 이탈 (> max_price_deviation_pct) → 조기 취소
   └─ 타임아웃 (> limit_timeout_seconds) → 취소
5. _cancel_and_handle_remainder()
   ├─ cancel race 대응: 이미 체결이면 그대로 반환
   ├─ 부분 체결: limit_fill 보존
   └─ 잔량: market fallback → _merge_fills() VWAP 합산
```

---

## 4. 설정 (SmartExecutorConfig)

`src/eda/smart_executor_config.py` — Pydantic `frozen=True` 모델.

| 필드 | 기본값 | 범위 | 설명 |
|------|--------|------|------|
| `enabled` | `false` | bool | opt-in 활성화 |
| `limit_timeout_seconds` | `30.0` | 5~300 | Limit 대기 시간 |
| `price_offset_bps` | `1.0` | 0~10 | bid/ask 대비 offset (bps) |
| `max_price_deviation_pct` | `0.3` | 0.05~2.0 | 조기 취소 임계값 (%) |
| `poll_interval_seconds` | `2.0` | 0.5~30 | 상태 확인 주기 |
| `max_concurrent_limit_orders` | `4` | 1~20 | 동시 limit 제한 |
| `fallback_to_market` | `true` | bool | timeout 시 market 전환 |

---

## 5. 비용 절감 효과

```text
Binance Futures 수수료:
  Taker fee:  0.04%  (시장가)
  Maker fee:  0.02%  (지정가)

12H 전략 (연 146회 왕복, 70% limit 체결 가정):
  기존:  146 × 2 × 0.04% = 11.68%
  개선:  146 × 2 × (0.02%×0.7 + 0.04%×0.3) = 6.47%
  절감:  연 5.21%p
```

---

## 6. 백테스트 시뮬레이션

`BacktestExecutor(smart_execution=True)` 설정 시:

- **Maker fee** 적용 (0.02%)
- **Slippage 0** (limit order는 원하는 가격에 체결)
- 긴급 주문은 기존대로 taker fee + slippage 적용

---

## 7. Metrics

`SmartExecutorMetrics` (`src/monitoring/metrics.py`):

- `on_limit_placed(symbol)` — limit 주문 배치
- `on_limit_filled(symbol)` — limit 완전 체결
- `on_limit_timeout(symbol)` — 타임아웃 발생
- `on_market_fallback(symbol)` — market fallback
- `on_partial_fill_merged(symbol)` — 부분 체결 VWAP 합산

---

## 8. Code Map

```text
src/eda/
├── smart_executor.py         # SmartExecutor, Urgency, LimitOrderState
├── smart_executor_config.py  # SmartExecutorConfig (Pydantic)
└── executors.py              # LiveExecutor (inner), BacktestExecutor
```

---

## 9. 안전 장치

| 장치 | 설명 |
|------|------|
| Market fallback | Limit 실패 시 항상 market으로 전환 (주문 누락 방지) |
| Cancel race 대응 | 취소 후 최종 상태 재조회 — 이미 체결이면 그대로 반환 |
| API 비정상 감지 | `is_api_healthy == False` → 즉시 market 모드 |
| 동시 limit 제한 | `max_concurrent_limit_orders` 초과 시 market 전환 |
| Stale order 정리 | `cleanup_stale_orders()` — 시작 시 미체결 limit 일괄 취소 |
| 가격 이탈 감지 | mid price 기준 deviation 초과 시 조기 취소 → market fallback |
