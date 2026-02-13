# MC Coin Bot — Monitoring Reference

Prometheus + Grafana 기반 모니터링 시스템 레퍼런스.

## Architecture

```
mc-bot (:8000/metrics)
    → Prometheus (scrape 10s)
        → Grafana (dashboard)
        → Alertmanager (alerts.yml)

node-exporter (:9100/metrics)
    → Prometheus (scrape 15s)
```

## Metric Layers

### Layer 1: Order Execution

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_orders_total` | Counter | symbol, side, order_type, status | 주문 건수 (status: ack/filled/rejected) |
| `mcbot_order_latency_seconds` | Histogram | symbol | 주문요청→체결 지연시간 |
| `mcbot_slippage_bps` | Histogram | symbol, side | 슬리피지 (basis points) |
| `mcbot_order_rejected_total` | Counter | symbol, reason | 거부 주문 |
| `mcbot_fees_usdt_total` | Counter | symbol | 누적 수수료 (USDT) |
| `mcbot_fills` | Counter | symbol, side | 체결 건수 |
| `mcbot_signals` | Counter | symbol | 시그널 건수 |
| `mcbot_bars` | Counter | timeframe | 처리된 bar 수 |

### Layer 2: Position & PnL

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_equity_usdt` | Gauge | — | 포트폴리오 총 가치 |
| `mcbot_drawdown_pct` | Gauge | — | 고점 대비 낙폭 |
| `mcbot_cash_usdt` | Gauge | — | 가용 현금 |
| `mcbot_open_positions` | Gauge | — | 열린 포지션 수 |
| `mcbot_position_size` | Gauge | symbol | 포지션 수량 |
| `mcbot_position_notional_usdt` | Gauge | symbol | 포지션 명목 금액 |
| `mcbot_unrealized_pnl_usdt` | Gauge | symbol | 미실현 손익 |
| `mcbot_realized_pnl_usdt_total` | Counter | symbol | 누적 실현 손익 |
| `mcbot_aggregate_leverage` | Gauge | — | 포트폴리오 총 레버리지 |
| `mcbot_margin_used_usdt` | Gauge | — | 사용 중인 마진 |

### Layer 3: Exchange API

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_exchange_api_calls_total` | Counter | endpoint, status | API 호출 수 |
| `mcbot_exchange_api_latency_seconds` | Histogram | endpoint | API 응답 시간 |
| `mcbot_exchange_ws_connected` | Gauge | symbol | WS 연결 상태 (1/0) |
| `mcbot_exchange_consecutive_failures` | Gauge | — | 연속 API 실패 횟수 |

### Layer 4: Bot Health

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_uptime_seconds` | Gauge | — | 봇 가동 시간 |
| `mcbot_heartbeat_timestamp` | Gauge | — | 마지막 heartbeat |
| `mcbot_errors_total` | Counter | component, error_type | 에러 수 |
| `mcbot_eventbus_queue_depth` | Gauge | — | 큐 대기 이벤트 수 |
| `mcbot_eventbus_events_dropped_total` | Counter | — | 이벤트 드롭 수 |
| `mcbot_eventbus_handler_errors_total` | Counter | — | 핸들러 에러 수 |
| `mcbot_circuit_breaker` | Counter | — | 서킷 브레이커 발동 |
| `mcbot_risk_alerts` | Counter | level | 리스크 알림 |

### Meta

| Metric | Type | Description |
|--------|------|-------------|
| `mcbot_info` | Info | 봇 메타데이터 (version, mode, exchange, strategy) |
| `mcbot_trading_mode` | Enum | 현재 모드 (backtest/paper/shadow/live) |

## Alert Rules

| Alert | Condition | For | Severity |
|-------|-----------|-----|----------|
| HighDrawdown | `mcbot_drawdown_pct > 10` | 1m | WARNING |
| APIUnhealthy | `mcbot_exchange_consecutive_failures >= 5` | 30s | CRITICAL |
| EventsDropped | `rate(events_dropped[5m]) > 0` | 2m | WARNING |
| WSDisconnected | `mcbot_exchange_ws_connected == 0` | 2m | CRITICAL |
| HighSlippage | `histogram_quantile(0.95, slippage_bps) > 20` | 5m | WARNING |
| QueueCongestion | `mcbot_eventbus_queue_depth > 5000` | 1m | WARNING |

## Alert Runbook

### HighDrawdown
**원인**: 포트폴리오 손실이 고점 대비 10% 초과.
**대응**:
1. Grafana에서 포지션별 PnL 확인
2. 시장 전체 하락인지 단일 포지션 문제인지 판단
3. 필요 시 `/kill` 명령으로 전 포지션 청산

### APIUnhealthy
**원인**: Binance API 5회 연속 실패.
**대응**:
1. Binance 상태 페이지 확인
2. 네트워크 연결 상태 점검
3. Rate limit 초과 여부 확인 (API 호출률 패널)
4. 자동 복구 대기 (성공 시 카운터 리셋)

### EventsDropped
**원인**: EventBus 큐 가득 → 드롭 가능 이벤트(BAR) 제거.
**대응**:
1. Queue Depth 패널에서 추이 확인
2. Handler Errors 증가 시 특정 핸들러 병목 점검
3. queue_size 설정 증가 검토

### WSDisconnected
**원인**: WebSocket 연결 끊김 (네트워크 문제 또는 거래소 유지보수).
**대응**:
1. 자동 재연결 대기 (exponential backoff)
2. 2분 이상 지속 시 봇 로그 확인
3. 전 심볼 동시 끊김이면 네트워크 문제

### HighSlippage
**원인**: 주문 체결가가 기대가 대비 20bp 이상 차이.
**대응**:
1. 유동성 부족 코인인지 확인 (volume 패널)
2. 주문 크기가 시장 대비 과대한지 점검
3. 시장 급변 시 일시적 현상

### QueueCongestion
**원인**: EventBus 처리 속도 < 이벤트 발생 속도.
**대응**:
1. Bars Processed Rate 이상 없는지 확인
2. 핸들러 에러로 인한 지연 확인
3. 큐 크기 설정 검토

## Glossary

| Term | Meaning |
|------|---------|
| **Histogram** | 값의 분포를 구간(bucket)별로 세는 메트릭. P95 등 분위수 계산에 사용 |
| **Slippage** | 기대가 vs 체결가 차이. Basis Points(bp)로 측정. 1bp = 0.01% |
| **Basis Points (bp)** | 1bp = 0.01%. 금융에서 작은 비율 변화를 표현하는 단위 |
| **Drawdown** | 고점 대비 현재 가치 하락 비율. MDD = 역사적 최대 낙폭 |
| **Circuit Breaker** | 시스템 손실 임계값 초과 시 전 포지션 강제 청산 안전장치 |
| **Cardinality** | 라벨 조합의 고유 수. 높으면 Prometheus 메모리 폭증 |
| **Backpressure** | 이벤트 큐 가득 시 덜 중요한 이벤트를 버리는 메커니즘 |
| **Rate Limit** | 거래소 API 호출 제한. Binance: 1200 req/min |
| **Gauge** | 현재 값 메트릭 (올/내림 가능). 온도계 비유 |
| **Counter** | 증가만 하는 메트릭. `rate()`로 초당 증가율 계산 |
| **Info / Enum** | 봇의 정적 메타데이터를 노출하는 특수 메트릭 |

## Local Verification

```bash
# 1. Docker 환경 시작
docker-compose up -d

# 2. Grafana 접속
open http://localhost:3000  # admin / admin (기본)

# 3. 메트릭 확인
curl localhost:8000/metrics | grep mcbot_

# 4. 알림 규칙 확인
curl localhost:9090/api/v1/rules | jq .
```
