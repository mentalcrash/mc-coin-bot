# MC Coin Bot — Monitoring Reference

Prometheus + Grafana 기반 모니터링 시스템 레퍼런스.

## Architecture

```
mc-bot (:8000/metrics)
    -> Prometheus (scrape 10s)
        -> Grafana (9 primary + 3 quick-overview dashboards)
        -> Alertmanager (alerts.yml, 45 rules)

node-exporter (:9100/metrics)
    -> Prometheus (scrape 15s)

Anomaly Detectors (in-process)
    -> GBM Drawdown Monitor
    -> Distribution Drift (KS test)
    -> Execution Anomaly Detector
    -> Conformal-RANSAC Decay Detector
```

### Key Files

| File | Description |
|------|-------------|
| `src/monitoring/metrics.py` | MetricsExporter — Prometheus 메트릭 정의 + EventBus handler |
| `src/monitoring/process_monitor.py` | Event loop lag, CPU, memory, FD 모니터링 |
| `src/monitoring/chart_generator.py` | Matplotlib 차트 생성 (equity, drawdown, heatmap) |
| `src/monitoring/anomaly/gbm_drawdown.py` | GBM 기반 drawdown 정상 범위 검증 |
| `src/monitoring/anomaly/distribution.py` | KS test 기반 수익률 분포 변화 감지 |
| `src/monitoring/anomaly/execution_quality.py` | 실행 품질 이상 패턴 감지 |
| `src/monitoring/anomaly/conformal_ransac.py` | RANSAC + Conformal 구조적 쇠퇴 감지 |
| `src/orchestrator/metrics.py` | OrchestratorMetrics — Pod/Portfolio Prometheus 메트릭 |

---

## Metric Layers

### Layer 1: Order Execution

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_orders_total` | Counter | symbol, side, order_type, status | 주문 건수 (status: ack/filled/rejected). 단일 주문이 ack+filled 또는 ack+rejected로 **2회** 카운트됨. CircuitBreaker 직접 체결은 pending 없어 미반영 (mcbot_fills만 증가) |
| `mcbot_order_latency_seconds` | Histogram | symbol | 주문요청→체결 지연시간. **주의**: backtest에서는 코드 실행 시간(마이크로초), live/paper에서만 실제 주문 지연 |
| `mcbot_slippage_bps` | Histogram | symbol, side | 슬리피지 절대값 (basis points). signal-to-execution 가격 변화 측정 (순수 마이크로스트럭처 슬리피지 아님). MARKET 주문 기준가 = last bar close (1D TF에서 최대 24h 전 가격) |
| `mcbot_slippage_signed_bps` | Histogram | symbol, side | 방향성 슬리피지 (양수=불리, 음수=유리). BUY: fill-expected, SELL: expected-fill |
| `mcbot_order_rejected_total` | Counter | symbol, reason | 거부 주문. reason: leverage_exceeded, max_positions, order_size_exceeded, circuit_breaker, duplicate, other |
| `mcbot_fees_usdt_total` | Counter | symbol | 누적 수수료 (USDT) |
| `mcbot_fills` | Counter | symbol, side | 체결 건수 |
| `mcbot_signals` | Counter | symbol | 시그널 건수 |
| `mcbot_bars` | Counter | timeframe | 처리된 bar 수 |

### Layer 1b: Live Execution (Live/Paper 모드 전용)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_live_min_notional_skip_total` | Counter | symbol | MIN_NOTIONAL 미달로 스킵된 주문 |
| `mcbot_live_api_blocked_total` | Counter | symbol | API unhealthy로 차단된 주문 |
| `mcbot_live_partial_fill_total` | Counter | symbol | Partial fill 발생 건수 |
| `mcbot_live_fill_parse_failure_total` | Counter | symbol | Fill 응답 파싱 실패 건수 |

### Layer 2: Position & PnL

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_equity_usdt` | Gauge | -- | 포트폴리오 총 가치 |
| `mcbot_drawdown_pct` | Gauge | -- | HWM 기반 고점 대비 낙폭 (0-100%) |
| `mcbot_cash_usdt` | Gauge | -- | 가용 현금 |
| `mcbot_open_positions` | Gauge | -- | 열린 포지션 수 |
| `mcbot_position_size` | Gauge | symbol | 포지션 수량 |
| `mcbot_position_notional_usdt` | Gauge | symbol | 포지션 명목 금액 (mark-to-market, last_price 기준) |
| `mcbot_unrealized_pnl_usdt` | Gauge | symbol | 미실현 손익 |
| `mcbot_realized_profit_usdt_total` | Counter | symbol | 누적 실현 수익 |
| `mcbot_realized_loss_usdt_total` | Counter | symbol | 누적 실현 손실 |
| `mcbot_aggregate_leverage` | Gauge | -- | 포트폴리오 총 레버리지 (total_abs_notional / equity) |
| `mcbot_margin_used_usdt` | Gauge | -- | 사용 중인 마진 |

### Layer 3: Exchange API

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_exchange_api_calls_total` | Counter | endpoint, status | API 호출 수. endpoint: create_order, create_stop_market, fetch_positions, fetch_balance, fetch_ticker, fetch_open_orders, fetch_order, cancel_order, cancel_all_orders (9종). status: "success" \| "failure". 미계측: setup_account(), 데이터 수집 6종 (fetch_funding_rate_history 등) |
| `mcbot_exchange_api_latency_seconds` | Histogram | endpoint | API 응답 시간 (monotonic 시계). Buckets: 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0s. `_retry_with_backoff` 포함 총 소요 시간 (retry backoff 대기 시간 포함) |
| `mcbot_exchange_ws_connected` | Gauge | symbol | WS 연결 상태 (1=connected, 0=disconnected). `WsStatusCallback.on_ws_status()` 콜백으로 실시간 갱신. 상세: Layer 6 참조 |
| `mcbot_exchange_consecutive_failures` | Gauge | -- | 주문 실행(create_order, create_stop_market) 연속 실패 횟수. **write-ops만 추적** — 읽기 API 실패는 미반영. InvalidOrder는 로직 에러로 CB 카운트 제외. 5 도달 시 주문 차단 (is_healthy=False). 성공 시 0 리셋 |

**운영 주의사항:**

- **consecutive_failures 범위 제한**: `_record_failure()`는 create_order/create_stop_market에서만
  호출됨. InvalidOrder는 CB 카운트 제외 (로직 에러). 읽기 API(fetch_positions, fetch_balance 등)
  실패는 consecutive_failures에 반영되지 않으므로 `APIUnhealthy` alert는 주문 실행 실패에만 반응함
- **Latency 측정 특성**: `time.monotonic()` 기반. `_retry_with_backoff` 호출 전후 측정이므로
  retry 시도 + exponential backoff 대기 시간이 모두 포함됨
- **미계측 영역**: `setup_account()` (일회성 설정, 계측 불필요), 데이터 수집 6종
  (fetch_funding_rate_history, fetch_open_interest_history, fetch_long_short_ratio,
  fetch_taker_buy_sell_ratio, fetch_top_long_short_account_ratio,
  fetch_top_long_short_position_ratio)은 인제스트 전용으로 라이브 트레이딩과 무관

### Layer 4: Bot Health

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_uptime_seconds` | Gauge | -- | `time.monotonic()` 기반 봇 가동 시간. MetricsExporter 생성 시점부터 측정, 30초 주기 갱신. 프로세스 재시작 시 0 리셋 |
| `mcbot_heartbeat_timestamp` | Gauge | -- | HeartbeatEvent 수신 시 `event.timestamp.timestamp()` 기록. 30초 주기로 LiveRunner `_periodic_metrics_update()`에서 발행. Event loop hung 감지에 사용 |
| `mcbot_errors_total` | Counter | component, error_type | 컴포넌트별 에러 수. component: EventBus \| LiveRunner \| Exchange. error_type: exception class name. EventBus handler error, LiveRunner crash, Exchange API failure 시 증가 |
| `mcbot_eventbus_queue_depth` | Gauge | -- | 30초 주기 `bus.queue_size` 조회. 기본 큐 크기 10,000. 5,000 초과 시 QueueCongestion alert, 8,000 초과 시 QueueCongestionCritical alert |
| `mcbot_eventbus_events_dropped_total` | Counter | -- | Delta 방식 — 내부 카운터 vs 이전 스냅샷 차이분만 반영. 드롭 대상(DROPPABLE): BAR, HEARTBEAT, RISK_ALERT, REGIME_CHANGE. 비드롭(SIGNAL, FILL, ORDER_*, CB 등)은 큐 여유까지 blocking |
| `mcbot_eventbus_handler_errors_total` | Counter | -- | Delta 방식. handler 내 exception 시 증가. 에러 격리: 한 handler 실패해도 나머지 계속 실행 |
| `mcbot_last_bar_age_seconds` | Gauge | symbol | 마지막 bar 수신 후 경과 시간. BarEvent(1m+target TF) handler에서 0 리셋, `update_bar_ages()`에서 30초 주기 갱신. `time.monotonic()` 기반. Layer 6 교차 기재 — WS 데이터 파이프라인 종단 건강 지표 |
| `mcbot_circuit_breaker` | Counter | -- | CircuitBreakerEvent 수신 시 1 증가. 트리거: RiskManager (drawdown >= system_stop_loss). OMS가 전량 청산 실행. reason 필드는 metric 미노출 (JSONL audit log 확인) |
| `mcbot_risk_alerts` | Counter | level | level: WARNING \| CRITICAL. 발행 소스: RiskManager, ProcessMonitor (loop lag/RSS/FD), MetricsExporter (slippage/latency), Anomaly Detectors. source label 미노출 → audit log에서 확인 |

**운영 주의사항:**

- **Delta 방식 counter 갱신 주기**: EventBus 내부 카운터(events_dropped, handler_errors)는
  30초 주기로 Prometheus counter에 delta 반영. 30초 미만의 burst는 다음 갱신 주기에 일괄 반영됨
- **Backpressure 정책**: `DROPPABLE_EVENTS`(BAR, HEARTBEAT, RISK_ALERT, REGIME_CHANGE)는
  큐 가득 시 즉시 드롭. `NEVER_DROP`(SIGNAL, FILL, ORDER_REQUEST, ORDER_ACK, ORDER_REJECTED,
  CIRCUIT_BREAKER 등)은 큐 여유 생길 때까지 blocking. DROPPABLE 이벤트 드롭은 데이터 지연을
  의미하지만 거래 체인은 보호됨
- **Circuit Breaker 재발동 방지**: RiskManager 내부 `_circuit_breaker_triggered=True` 플래그로
  동일 세션에서 CB 중복 발동 방지. 재시작 시 리셋
- **risk_alerts source 추적**: `mcbot_risk_alerts` counter는 level label만 노출. 발행 소스
  (RiskManager, ProcessMonitor, MetricsExporter, Anomaly Detectors)는 JSONL audit log의
  `source` 필드에서 확인 가능
- **미노출 EventBus 내부 메트릭**: `events_published`, `events_dispatched`, `max_queue_depth`는
  EventBusMetrics에 내부 추적되지만 Prometheus로 미노출. `bus.metrics.snapshot()` 또는
  shutdown 로그에서 확인 가능

<!-- 향후 추가 검토 메트릭:
- `mcbot_eventbus_events_published_total`: 처리량(throughput) 측정. published vs dispatched 비교로 backpressure 정량화
- `mcbot_eventbus_max_queue_depth`: 갱신 주기(30초) 사이 peak depth. EventBusMetrics에 이미 내부 추적됨
- `mcbot_pending_orders_count`: MetricsExporter._pending_orders dict 크기. fill 누락/executor hung 감지
-->

### Layer 5: Per-Strategy

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_strategy_pnl_usdt` | Gauge | strategy | 전략별 누적 realized PnL (USDT). Fill 체결 시 `realized_pnl_delta` 귀속. 봇 재시작 시 0 리셋 |
| `mcbot_strategy_signals_total` | Counter | strategy, side | 전략별 시그널 수. side: LONG/SHORT/NEUTRAL |
| `mcbot_strategy_fills_total` | Counter | strategy, side | 전략별 체결 수. `_PendingOrder` 매칭된 fill만 카운트 |
| `mcbot_strategy_fees_usdt_total` | Counter | strategy | 전략별 누적 수수료 (USDT). fee > 0인 fill만 |
| `mcbot_strategy_slippage_bps` | Histogram | strategy, symbol, side | 전략별 슬리피지 (basis points). Buckets: 0,1,2,5,10,20,50,100. 카디널리티 ~320 (10전략 × 16심볼 × 2방향) |
| `mcbot_strategy_realized_profit_usdt_total` | Counter | strategy | 전략별 누적 실현 수익 (USDT) |
| `mcbot_strategy_realized_loss_usdt_total` | Counter | strategy | 전략별 누적 실현 손실 (USDT) |
| `mcbot_strategy_trade_count_total` | Counter | strategy | 전략별 완료 거래 수 (realized PnL 발생 fill만) |

**운영 주의사항:**

- **Strategy Name 추출 메커니즘**: `client_order_id = "{strategy}-{symbol_slug}-{counter}"` →
  `rsplit("-", 2)` 파싱. 하이픈 포함 전략명(예: `anchor-mom`) 정상 처리
- **`_PendingOrder` 매칭 한계**: `_pending_orders`에서 `client_order_id`로 매칭 실패 시
  per-strategy 카운터 미반영 (symbol-level 카운터만 증가). SL/TS 직접 체결은 PM이
  별도 client_order_id 발급하므로 정상 추적
- **PnL 귀속 방식**: `_last_fill_strategy[symbol]`로 최근 fill의 전략을 추적,
  PositionUpdateEvent의 `realized_pnl_delta`를 해당 전략에 귀속.
  단일전략 모드에서는 정확, 멀티전략 동일심볼 시 마지막 fill 전략에 귀속됨
- **카디널리티 경고**: `strategy_slippage_bps`는 3-label
  (strategy × symbol × side) → 10전략 × 16심볼 × 2방향 = 320 시계열.
  전략 수 20 초과 시 Prometheus 메모리 주의
- **Gauge 리셋**: `strategy_pnl_usdt`는 봇 재시작 시 0부터 시작 (persistent state 아님)

### Layer 6: WebSocket

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_exchange_ws_connected` | Gauge | symbol | WS 연결 상태 (1=connected, 0=disconnected). `on_ws_status()` 콜백으로 실시간 갱신. WSDisconnected alert 기준 메트릭. Layer 3 교차 기재 |
| `mcbot_ws_reconnects_total` | Counter | symbol | 끊김 후 재연결 **성공** 시에만 증가 (끊김 자체는 미카운트). `_stream_symbol()`에서 `was_disconnected=True` 상태의 재연결 시 `on_ws_reconnect()` 콜백 경유 |
| `mcbot_ws_last_message_age_seconds` | Gauge | symbol | 마지막 WS 메시지 후 경과 시간. `time.monotonic()` 기반 (NTP 변경 무관). `PrometheusWsDetailCallback.update_message_ages()`에서 **30초 주기** 갱신 — 실시간이 아닌 snapshot. WSNoMessages(60초) alert 기준 메트릭 |
| `mcbot_ws_messages_received_total` | Counter | symbol | `watch_ohlcv()` 반환마다 1 증가. `on_ws_message()` 콜백 경유. `rate()` 쿼리로 심볼별 throughput 추적 가능. 급감 시 시그널 생성 지연의 선행 지표 |
| `mcbot_last_bar_age_seconds` | Gauge | symbol | BarEvent(1m+target TF) handler에서 0 리셋. `update_bar_ages()`에서 30초 주기 갱신. `time.monotonic()` 기반. Layer 4(Bot Health) 정의, WS 파이프라인 종단 지표로 교차 기재 |

**운영 주의사항:**

- **gauge 갱신 주기와 alert 정확성**: `ws_last_message_age_seconds`, `last_bar_age_seconds`
  모두 `_periodic_metrics_update()` 30초 주기 갱신. WSNoMessages(60초 threshold + 1분 for)
  alert 실제 감지까지 최대 ~90초 지연 가능
- **monotonic 시계**: 두 age gauge 모두 `time.monotonic()` 기반. NTP 시계 변경에 무관하게
  정확한 경과 시간 측정
- **이중 staleness 감지**: LiveDataFeed 내부 `_staleness_monitor()`(120초 timeout) →
  RiskAlertEvent → Discord 알림 vs Prometheus `ws_last_message_age_seconds`(60초) +
  WSNoMessages alert → Alertmanager. 두 경로가 독립 동작하여 이중 방어
- **이중 heartbeat 추적**: `_last_received`(LiveDataFeed 내부 staleness용) +
  `_last_message_time`(PrometheusWsDetailCallback Prometheus용) 별도 추적.
  `_stream_symbol()`에서 `_record_heartbeat()` + `on_ws_message()` 동시 호출로 갱신
- **Reconnection 파라미터**: 초기 `_INITIAL_RECONNECT_DELAY=1.0`초 → 최대
  `_MAX_RECONNECT_DELAY=60.0`초 exponential backoff. `_MAX_CONSECUTIVE_FAILURES=10`
  도달 시 CRITICAL RiskAlertEvent 발행. 성공 시 delay 즉시 리셋
- **Stagger delay**: 심볼당 0.5초 offset (`stagger_delay=i * 0.5`). 16심볼 기준 최종
  연결까지 7.5초. Binance WS 동시 연결 에러(1008) 방지 목적
- **데이터 파이프라인 건강 지표**: `ws_messages_received_total`(입구) →
  `last_bar_age_seconds`(출구). `rate(ws_messages_received_total[5m])` 급감 시
  bar 생성 지연 및 시그널 생성 지연의 선행 지표

### Layer 7: Process & Event Loop

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_event_loop_lag_seconds` | Gauge | -- | Event loop 스케줄링 지연. `sleep(interval)` 전후 monotonic 차이 - interval |
| `mcbot_active_tasks` | Gauge | -- | 활성 asyncio Task 수. `asyncio.all_tasks()` 기반 |
| `mcbot_process_memory_rss_bytes` | Gauge | -- | **현재** RSS 메모리. Linux: `/proc/self/statm` field[1] × page_size. macOS: `ru_maxrss` (peak fallback) |
| `mcbot_process_memory_rss_peak_bytes` | Gauge | -- | Peak (high-water mark) RSS. `ru_maxrss` 기반. 진단용 — GC 후 감소하지 않음 |
| `mcbot_process_cpu_percent` | Gauge | -- | CPU 사용률 (%). `os.times()` user+system delta / wall delta × 100 |
| `mcbot_process_open_fds` | Gauge | -- | 열린 file descriptor 수. Linux: `/proc/self/fd`, macOS: `/dev/fd`. 실패 시 0 + warning 로그 |
| `mcbot_process_gc_collections_total` | Counter | generation | GC collection 횟수 (generation: 0, 1, 2). `gc.get_stats()` delta 방식 |
| `mcbot_process_thread_count` | Gauge | -- | 활성 스레드 수. `threading.active_count()` 기반 |

**운영 주의사항:**

- **RSS current vs peak**: `mcbot_process_memory_rss_bytes`는 Linux에서 `/proc/self/statm`을
  통해 실시간 현재값을 반환하므로 GC 후 감소가 반영됩니다. `mcbot_process_memory_rss_peak_bytes`는
  `ru_maxrss` 기반 high-water mark로 프로세스 수명 동안 증가만 합니다.
  macOS에서는 `/proc` 미지원으로 두 값이 동일할 수 있습니다 (둘 다 peak)
- **갱신 주기**: 기본 10초. `ProcessMonitorConfig.interval`로 설정 가능
- **GC 영향**: gen2 수집은 full GC — 대량 객체 순환 시 event loop lag spike의 근인.
  `rate(mcbot_process_gc_collections_total{generation="2"}[5m]) > 0.1` 시 주의
- **Thread count 의미**: asyncio 이벤트 루프 외의 스레드 (ThreadPoolExecutor, DB driver 등).
  급증 시 blocking I/O 과다 또는 thread pool 미회수
- **Alert 쿨다운**: 모든 alert에 key별 60초 쿨다운 적용 (기본값). 동일 alert의 반복 발행 방지.
  `ProcessMonitorConfig.alert_cooldown`으로 조정 가능
- **임계값 설정**: `ProcessMonitorConfig` frozen dataclass로 환경별 튜닝 가능.
  기본값: loop_lag 1.0s, RSS 2GB, FD 1000, CPU 80%, active_tasks 200
- **FD count 실패**: 컨테이너 환경에서 `/proc/self/fd` 접근 불가 시 0 반환 + warning 로그.
  FD 고갈을 0으로 착각하지 않도록 로그 확인 필요

### Layer 8: Anomaly Detection

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_distribution_ks_statistic` | Gauge | strategy | KS test 통계량 |
| `mcbot_distribution_p_value` | Gauge | strategy | KS test p-value |
| `mcbot_ransac_slope` | Gauge | strategy | RANSAC 추정 기울기 |
| `mcbot_ransac_conformal_lower` | Gauge | strategy | RANSAC conformal 하한 |
| `mcbot_ransac_decay_detected` | Gauge | strategy | RANSAC 구조적 쇠퇴 감지 (0/1) |
| `mcbot_ransac_current_cumulative` | Gauge | strategy | RANSAC 현재 누적 수익 |
| `mcbot_gbm_drawdown_depth` | Gauge | strategy | 현재 drawdown 깊이 (0.0\~1.0) |
| `mcbot_gbm_drawdown_duration_days` | Gauge | strategy | 현재 drawdown 지속 일수 |
| `mcbot_gbm_severity` | Gauge | strategy | GBM 심각도 (0=NORMAL, 1=WARNING, 2=CRITICAL) |
| `mcbot_execution_consecutive_rejections` | Gauge | -- | 연속 거부 횟수 |
| `mcbot_execution_fill_rate` | Gauge | -- | 1h 윈도우 fill rate (0.0\~1.0) |

**운영 주의사항:**

- **메트릭 업데이트 경로**: OrchestratorMetrics.update()에서 30초 주기로 Prometheus gauge에 반영.
  실제 detector 결과는 LifecycleManager.\_check\_degradation()에서 일별 수익률 입력 시 갱신되므로,
  실질 변화는 리밸런싱 주기(1D bar)에 종속
- **strategy 라벨**: pod\_id를 strategy 라벨로 사용. 단일 전략 Pod에서는 전략명과 동일
- **Detector 역할 — 관측 전용(observability-only)**: GBM Drawdown, Distribution Drift,
  Conformal-RANSAC 3종은 **Pod 상태 전이를 유발하지 않습니다**. 이 detector들은 결과를
  저장(last\_gbm\_result, last\_dist\_result, last\_ransac\_result)하고 Prometheus gauge로
  export할 뿐입니다. Pod lifecycle 상태 전이(WARNING → PROBATION → RETIRED)는 오직
  **PageHinkley detector**만이 트리거합니다 (`lifecycle.py:_check_degradation()` — PH만 return).
  GBM/Distribution/RANSAC는 운영자의 **수동 판단 보조** 메트릭으로 활용하세요
- **GBM Drawdown Monitor**: 전용 gauge 3종(`gbm_drawdown_depth`, `gbm_drawdown_duration_days`,
  `gbm_severity`). CRITICAL 시 RiskAlertEvent도 경유하여 `mcbot_risk_alerts` counter에 반영
- **Execution Anomaly Detector**: 전용 gauge 2종(`execution_consecutive_rejections`,
  `execution_fill_rate`). `MetricsExporter._on_fill()`, `_on_order_rejected()`에서 실시간 갱신.
  fill rate는 `_periodic_metrics_update()` 30초 주기로 gauge 갱신 + 이상 시 RiskAlertEvent 발행
- **Execution Anomaly 전역 단일 인스턴스**: `MetricsExporter.__init__()`에서 단일
  `ExecutionAnomalyDetector()` 생성. 모든 심볼의 fill/rejection이 하나의 detector로 집계됨.
  심볼별 독립 패턴 감지 미지원 (향후 개선 고려)
- **최소 샘플 요구 (Cold Start)**:

| Detector | 최소 샘플 | 미달 시 동작 |
|----------|----------|-------------|
| GBM Drawdown | 30일 (`_MIN_OBSERVATION_DAYS`) | NORMAL 반환, gauge 미갱신 |
| Distribution Drift | 30개 (`_MIN_RECENT_SAMPLES`) | NORMAL 반환, gauge 이전 값 유지 |
| Conformal-RANSAC | 60개 (`_DEFAULT_MIN_SAMPLES`) | NORMAL 반환, gauge 이전 값 유지 |
| Execution Latency | 5 fills (`_LATENCY_COLD_START`) | spike 감지 비활성 |

- **auto\_init\_detectors() 호출 경로**: LiveRunner warmup 완료 후
  (`live_runner.py:1267-1272`), 각 Pod의 `daily_returns`로
  `lifecycle.auto_init_detectors(pod_id, returns)`를 호출합니다.
  Pod에 daily\_returns가 없으면 (신규 Pod) 초기화 스킵.
  mu/sigma가 0에 가까우면(`sigma < 1e-12`) 초기화 스킵
- **RANSAC 파라미터 상세**: `ransac_min_samples_ratio = 0.5` (50% inlier 요구),
  `conformal alpha = 0.05` (95% prediction interval),
  `auto_init_detectors()`에서 `ransac_alpha=0.05, ransac_window_size=180` 전달
- **GBM Duration heuristic**: drift-ratio 기반 heuristic 근사이며,
  Magdon-Ismail closed-form의 정확한 구현이 아님.
  정밀 Confidence Interval이 필요하면 Monte Carlo 시뮬레이션 검증 권장
- **State Persistence**: GBM/Distribution/RANSAC는
  `LifecycleManager.to_dict()/restore_from_dict()`로 직렬화.
  Execution Anomaly는 타임스탬프 `time.monotonic()` 기반 → 재시작 시 1h 윈도우 리셋.
  재시작 후 `auto_init_detectors()` warmup 단계에서 자동 재호출

### Layer 9: On-chain Data

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_onchain_fetch_total` | Counter | source, status | On-chain fetch 시도 (success/failure/empty) |
| `mcbot_onchain_fetch_latency_seconds` | Histogram | source | Fetch 소요 시간 |
| `mcbot_onchain_fetch_rows` | Gauge | source, name | 마지막 fetch 반환 행 수 |
| `mcbot_onchain_last_success_timestamp` | Gauge | source | 마지막 성공 Unix timestamp |
| `mcbot_onchain_cache_size` | Gauge | symbol | 심볼별 캐시된 on-chain 컬럼 수 |
| `mcbot_onchain_cache_refresh_total` | Counter | status | 캐시 refresh 횟수 (success/failure) |

**운영 주의사항:**

- **Histogram Bucket (fetch\_latency\_seconds)**: (0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
- **갱신 메커니즘**: LiveOnchainFeed가 소스별 독립 asyncio task로 API polling.
  DeFiLlama: 6h, Sentiment: 6h, CoinMetrics: 12h, mempool(BTC): 6h, Etherscan(ETH): 12h
- **Callback 주입**: PrometheusOnchainCallback이 OnchainFetcher에 주입되어
  fetch\_total/latency/rows/last\_success 4개 메트릭 자동 기록
- **cache\_refresh\_total**: 각 polling 사이클 완료 시 success/failure 카운트.
  5개 fetcher(defillama, sentiment, coinmetrics, btc\_mining, eth\_supply) 각각 독립 반영
- **cache\_size gauge**: LiveOnchainFeed.update\_cache\_metrics()에서 30초 주기 갱신
  (LiveRunner.\_periodic\_metrics\_update 호출 경유)
- **Graceful Degradation**: API 실패 시 이전 캐시 값 유지. 전체 실패 시에도
  Silver 초기 로드 데이터로 cold start 보장
- **ETH supply polling 조건**: ETHERSCAN\_API\_KEY 환경변수 + ETH 심볼 포함 시에만 활성

### Layer 10: Orchestrator

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_pod_equity_usdt` | Gauge | pod_id | Pod 현재 equity (USDT) |
| `mcbot_pod_allocation_fraction` | Gauge | pod_id | Pod 자본 배분 비율 |
| `mcbot_pod_rolling_sharpe` | Gauge | pod_id | Pod Rolling Sharpe |
| `mcbot_pod_drawdown_pct` | Gauge | pod_id | Pod 현재 drawdown (%) |
| `mcbot_pod_risk_contribution` | Gauge | pod_id | Pod 리스크 기여도 PRC (%) |
| `mcbot_pod_lifecycle_state` | Enum | pod_id | Pod 생명주기 (incubation/production/warning/probation/retired) |
| `mcbot_portfolio_effective_n` | Gauge | -- | 포트폴리오 유효 분산도 (1/HHI) |
| `mcbot_portfolio_avg_correlation` | Gauge | -- | 평균 pair-wise 상관계수 |
| `mcbot_active_pods` | Gauge | -- | 활성 Pod 수 |
| `mcbot_netting_gross_exposure` | Gauge | -- | Pod간 총 gross exposure (sum of abs weights) |
| `mcbot_netting_net_exposure` | Gauge | -- | 넷팅 후 net exposure (sum of abs netted weights) |
| `mcbot_netting_offset_ratio` | Gauge | -- | 포지션 상쇄 비율 (0=상쇄 없음, 1=완전 상쇄) |

**운영 주의사항:**

- **갱신 주기**: OrchestratorMetrics.update()에서 30초 주기 갱신
  (LiveRunner.\_periodic\_metrics\_update 호출 경유)
- **카디널리티**: Pod-level 메트릭 6종 x pod\_id = 최대 N\_pods x 6 시계열.
  10 Pod 기준 60 시계열. Portfolio-level 3종 + Netting 3종 = 고정 6 시계열
- **lifecycle\_state Enum**: 5개 상태 중 정확히 1개만 1.0, 나머지 0.0.
  쿼리 형식: `mcbot_pod_lifecycle_state{mcbot_pod_lifecycle_state="production"} == 1`
- **Netting 메트릭 해석**: offset\_ratio 0.5 = 50% 상쇄. Pod간 반대 포지션이 많을수록 1에 가까움.
  last\_pod\_targets가 비어있으면 3개 gauge 모두 0.0
- **PRC 계산 조건**: active\_pods < 2일 때 균등 배분 (1/N).
  Pod 수익률 길이 불일치 시 0으로 앞쪽 패딩하여 정렬
- **리셋 동작**: 봇 재시작 시 모든 gauge 0에서 시작.
  Orchestrator 첫 rebalance 실행 전까지 기본값 유지
- **에러 처리**: update() 내부 exception은 catch → logger.exception()으로 기록.
  하나의 sub-update 실패가 다른 메트릭 갱신을 차단하지 않음

### Meta

| Metric | Type | Description |
|--------|------|-------------|
| `mcbot_info` | Info | 봇 메타데이터 (version, mode, exchange, strategy) |
| `mcbot_trading_mode` | Enum | 현재 모드 (backtest/paper/shadow/live) |

---

## Anomaly Detection

### GBM Drawdown Monitor

GBM(Geometric Brownian Motion)으로 전략 PnL을 모델링하여 drawdown의 정상 범위를 검증.

| 조건 | Severity |
|------|----------|
| Depth OR Duration > 95% CI | WARNING |
| Depth AND Duration > 95% CI | CRITICAL |

```python
# 초기화: 백테스트 수익률에서 파라미터 추정
mu, sigma = GBMDrawdownMonitor.estimate_params(daily_returns)
monitor = GBMDrawdownMonitor(mu=mu, sigma=sigma, confidence=0.95)

# 매일 업데이트
result: DrawdownCheckResult = monitor.update(daily_return)
# -> severity, current_depth, expected_max_depth, depth_exceeded, ...
```

### Distribution Drift Detector

백테스트 수익률 분포 vs 최근 N일 수익률 분포를 KS 2-sample test로 비교.

| 조건 | Severity |
|------|----------|
| p-value < 0.05 | WARNING |
| p-value < 0.01 | CRITICAL |

- Default window: 60일, 최소 30개 샘플 필요

### Conformal-RANSAC Decay Detector

RANSAC regression + conformal prediction으로 구조적 성과 쇠퇴를 감지.

| 조건 | Severity |
|------|----------|
| Slope <= 0 OR Level breach | WARNING |
| Slope <= 0 AND Level breach | CRITICAL |

- Default window: 180일, 최소 60개 샘플
- RANSAC: 단일 outlier에 의한 왜곡 방지 (high breakdown point)

### Execution Anomaly Detector

실행 품질의 이상 패턴을 실시간 감지.

| 항목 | 정상 기준 | 이상 판정 |
|------|----------|----------|
| Fill Latency | EWMA baseline | > 3x EWMA (spike) |
| Consecutive Rejections | < 3건 | >= 3 WARNING, >= 5 CRITICAL |
| Fill Rate (1h) | > 80% | < 80% (min 5 orders) |
| Slippage Trend | Stable | 3건 연속 증가 |

---

## Alert Rules

| Alert | Condition | For | Severity |
|-------|-----------|-----|----------|
| HighDrawdown | `mcbot_drawdown_pct > 10` | 1m | WARNING |
| APIUnhealthy | `mcbot_exchange_consecutive_failures >= 5` | 30s | CRITICAL |
| EventsDropped | `rate(mcbot_eventbus_events_dropped_total[5m]) > 0` | 2m | WARNING |
| WSDisconnected | `mcbot_exchange_ws_connected == 0` | 2m | CRITICAL |
| HighSlippage | `histogram_quantile(0.95, rate(mcbot_slippage_bps_bucket[15m])) > 20` | 5m | WARNING |
| QueueCongestion | `mcbot_eventbus_queue_depth > 5000` | 1m | WARNING |
| HighEventLoopLag | `mcbot_event_loop_lag_seconds > 1.0` | 1m | WARNING |
| HighMemoryUsage | `mcbot_process_memory_rss_bytes > 2147483648` | 5m | WARNING |
| HighFDCount | `mcbot_process_open_fds > 1000` | 2m | WARNING |
| WSFrequentReconnects | `increase(mcbot_ws_reconnects_total[5m]) >= 3` | -- | WARNING |
| WSNoMessages | `mcbot_ws_last_message_age_seconds > 60` | 1m | CRITICAL |
| DistributionDrift | `mcbot_distribution_p_value < 0.05` | 1d | WARNING |
| DistributionDriftCritical | `mcbot_distribution_p_value < 0.01 and mcbot_distribution_p_value > 0` | 1d | CRITICAL |
| StructuralDecay | `mcbot_ransac_decay_detected == 1` | 1d | WARNING |
| StructuralDecayCritical | `mcbot_ransac_slope <= 0 AND cumulative < conformal_lower` | 1d | CRITICAL |
| GBMDrawdownWarning | `mcbot_gbm_severity == 1` | 0s | WARNING |
| GBMDrawdownCritical | `mcbot_gbm_severity == 2` | 0s | CRITICAL |
| ExecutionLowFillRate | `mcbot_execution_fill_rate < 0.8 and mcbot_execution_fill_rate > 0` | 5m | WARNING |
| ExecutionConsecutiveRejections | `mcbot_execution_consecutive_rejections >= 5` | 0s | CRITICAL |
| OnchainFetchHighFailureRate | `rate(mcbot_onchain_fetch_total{status="failure"}[24h]) / rate(mcbot_onchain_fetch_total[24h]) > 0.5` | 1h | WARNING |
| OnchainDataStale | `time() - mcbot_onchain_last_success_timestamp > 172800` | 1h | WARNING |
| OnchainCacheEmpty | `mcbot_onchain_cache_size == 0` | 30m | WARNING |
| OnchainFetchSlow | `histogram_quantile(0.95, rate(mcbot_onchain_fetch_latency_seconds_bucket[1h])) > 60` | 5m | INFO |
| APIReadFailures | `rate(mcbot_exchange_api_calls_total{endpoint=~"fetch_.*", status="failure"}[5m]) > 0.1` | 2m | WARNING |
| APILatencyHigh | `histogram_quantile(0.95, rate(mcbot_exchange_api_latency_seconds_bucket[5m])) > 5` | 2m | WARNING |
| APIRateLimitApproaching | `sum(rate(mcbot_exchange_api_calls_total[1m])) * 60 > 900` | 1m | WARNING |
| BotRestarted | `resets(mcbot_uptime_seconds[10m]) > 0` | 0s | WARNING |
| HeartbeatStale | `time() - mcbot_heartbeat_timestamp > 120` | 1m | CRITICAL |
| HandlerErrorsHigh | `rate(mcbot_eventbus_handler_errors_total[5m]) > 0.01` | 2m | WARNING |
| CircuitBreakerTriggered | `increase(mcbot_circuit_breaker_total[5m]) > 0` | 0s | CRITICAL |
| QueueCongestionCritical | `mcbot_eventbus_queue_depth > 8000` | 30s | CRITICAL |
| StrategyPnlNegative | `mcbot_strategy_pnl_usdt < -500` | 1h | WARNING |
| StrategySignalSilent | `rate(mcbot_strategy_signals_total[6h]) == 0` | 1d | WARNING |
| HighCPUUsage | `mcbot_process_cpu_percent > 80` | 2m | WARNING |
| HighTaskCount | `mcbot_active_tasks > 200` | 5m | WARNING |
| HighGCPressure | `rate(mcbot_process_gc_collections_total{generation="2"}[5m]) > 0.1` | 5m | WARNING |
| NettingLowOffset | `mcbot_netting_offset_ratio < 0.1 and mcbot_netting_gross_exposure > 0` | 1h | WARNING |
| PodInProbation | `mcbot_pod_lifecycle_state{mcbot_pod_lifecycle_state="probation"} == 1` | 0s | WARNING |
| HighGrossExposure | `mcbot_netting_gross_exposure > 3.0` | 5m | WARNING |
| HighAggregateLeverage | `mcbot_aggregate_leverage > 5` | 1m | CRITICAL |
| LiveAPIBlocked | `increase(mcbot_live_api_blocked_total[5m]) > 0` | 0s | CRITICAL |
| HighMarginUtilization | `mcbot_margin_used_usdt / mcbot_equity_usdt > 0.8 and mcbot_equity_usdt > 0` | 2m | WARNING |
| HighThreadCount | `mcbot_process_thread_count > 50` | 5m | WARNING |
| LiveFillParseFailure | `increase(mcbot_live_fill_parse_failure_total[5m]) > 0` | 0s | WARNING |
| LiveFrequentMinNotionalSkip | `increase(mcbot_live_min_notional_skip_total[1h]) >= 3` | 0s | INFO |

---

## Dual-Layer Alerting

MC Coin Bot은 두 가지 독립 경로로 alert를 발행합니다. 한쪽이 실패해도 다른 경로가 동작하여 이중 방어를 구성합니다.

### Architecture

```
Layer A: Code-level (즉시)
  MetricsExporter / ProcessMonitor → RiskAlertEvent → Discord 알림

Layer B: Prometheus (지속)
  alerts.yml → Prometheus evaluate → Alertmanager → notification
```

### 이중 경로 매핑

| 항목 | Layer A (Code → Discord) | Layer B (Prometheus → Alertmanager) |
|------|--------------------------|-------------------------------------|
| Slippage | MetricsExporter: 15bp WARNING, 30bp CRITICAL | `HighSlippage`: P95 > 20bp (5m) |
| API Latency | MetricsExporter: 5s WARNING, 10s CRITICAL | `APILatencyHigh`: P95 > 5s (2m) |
| Process Health | ProcessMonitor: loop_lag 1s, RSS 2GB, FD 1000, CPU 80%, tasks 200 | 동일 threshold alert rules |
| WS Staleness | LiveDataFeed `_staleness_monitor`: 120s WARNING, 180s (1.5x) CRITICAL | `WSNoMessages`: 60s + for 1m |
| Execution | ExecutionAnomalyDetector: fill_rate < 80% (min 5 orders), rejections >= 3 WARNING / >= 5 CRITICAL, latency > 3x EWMA, slippage 3건 연속 증가 | `ExecutionLowFillRate` < 0.8 (5m), `ExecutionConsecutiveRejections` >= 5 |
| Drawdown | RiskManager: `system_stop_loss` (config 기반) → CircuitBreaker | `HighDrawdown`: 10% (1m) |

### 운영 주의사항

- **Layer A (Discord)**: 즉시 전달. 프로세스 내부에서 직접 발행하므로 `for` 지속 조건 없음
- **Layer B (Prometheus)**: `for` 조건으로 일시적 spike 필터링. scrape 주기(10s) + evaluate 주기에 종속
- **양쪽 모두 미수신**: 프로세스 자체가 다운됨을 의미. 호스트 모니터링(systemd, Docker healthcheck) 필요
- **Threshold 설계 의도**: Layer A(Code)는 즉시 반응이므로 WARNING + CRITICAL 2단계 에스컬레이션.
  Layer B(Prometheus)는 `for` 지속 조건으로 일시적 spike를 필터링하므로 단일 threshold로 충분.
  동일 항목의 Layer A/B 임계값이 다른 것은 의도된 설계 (예: WS 120s vs 60s, Slippage 15/30 vs 20)
- **`_pending_orders` 타임아웃 부재**: MetricsExporter의 `_pending_orders` dict는
  OrderRequest 후 Fill/Reject 미수신 시 무한 누적. 장기 운영 시 메모리 증가 및
  stale latency 계산의 원인. 향후 개선 대상

---

## Reconciliation Monitoring

LiveRunner의 position/balance reconciliation은 거래소 실제 상태와 내부 상태의 drift를 감지합니다.

### 실행 구조

- **독립 asyncio task**: `_periodic_reconciliation()` — `_RECONCILER_INTERVAL = 60.0`초 주기
- **startup 검증**: `reconciler.initial_check()` — 봇 시작 시 즉시 1회 실행
- `_periodic_metrics_update()`(30초)와 **별도 task**로 독립 동작

### 임계값 및 행동

| 항목 | 임계값 | 행동 | Severity |
|------|--------|------|----------|
| Position drift (수량) | > 2% (`_DRIFT_THRESHOLD`) | Discord 알림 | WARNING |
| Position drift (방향) | PM LONG ↔ Exchange SHORT | Discord 알림 | WARNING |
| Position orphan | 한쪽만 포지션 보유 | Discord 알림 | WARNING |
| Balance drift | >= 2.0% (`_balance_notify_threshold`) | Discord 알림, RM peak sync 스킵 | WARNING |
| Balance drift | >= 5.0% (`_BALANCE_DRIFT_THRESHOLD`) | 로그 CRITICAL (Discord는 2%에서 이미 발행) | CRITICAL (로그) |
| Balance drift | < 2.0% | 정상, RM peak equity sync 실행 | -- |

### Auto-Correction 모드

- **기본값**: `auto_correct=False` (safety-first — 경고만 발행, PM 상태 미수정)
- **활성화 시**: drift > 10% (`_AUTO_CORRECT_THRESHOLD`) 인 포지션의 PM size를 거래소 기준으로 보정
- **현재 LiveRunner**: `PositionReconciler()` — auto_correct 비활성

### 운영 주의사항

- **RM Peak Sync 경로**: balance drift < 2% 시 `rm.sync_exchange_equity(exchange_equity)` 호출
  → 내부에서 peak equity 갱신 + `system_stop_loss` 대비 drawdown 재평가 → CircuitBreaker 트리거 가능
- **실패 시 동작**: `periodic_check()` 내부 exception → catch → `return []` (이전 상태 유지, 무음 계속)
  → API 일시 장애 시 안전하지만, 지속 실패 시 drift 누적 위험
- **Race Condition 인지**: OMS fill 처리와 Reconciler가 동시 실행될 수 있음.
  Fill 전파 중 거래소 조회 시 stale 상태와 비교 → 거짓 drift 감지 가능 (60초 주기로 자연 해소)
- **Balance drift 2% 초과 시 RM sync 스킵 이유**: 입금/출금으로 equity 급변 시
  거짓 peak 설정 → 거짓 CircuitBreaker 방지. 단, API 장애로 인한 부정확한 잔고에도 동일 동작

---

## Grafana Dashboards

`monitoring/grafana/dashboards/` 에 JSON으로 버전 관리 (9개 primary).

| Dashboard | File | 주요 패널 |
|-----------|------|----------|
| **Trading Overview** | `trading.json` | Equity curve, Drawdown, PnL (profit-loss), Leverage, Positions, Fills, Slippage, Fees |
| **Strategy Performance** | `strategy.json` | Per-strategy PnL, Signal Rate, Fill Rate, Fee, Slippage P95, Profit Factor |
| **Execution Quality** | `execution.json` | Fill latency, Slippage distribution, Fee accumulation, Live execution (min notional, API blocked, partial fills, fill rate) |
| **Exchange Health** | `exchange.json` | API latency, WS connection, Rate limit, Consecutive failures |
| **System Health** | `system.json` | Event loop lag, Queue depth, Memory, Active tasks, FD count, GC collections, Thread count, Peak RSS |
| **Market Regime** | `regime.json` | Bar age, Stale symbols, Heartbeat, Risk alerts, Circuit breaker, Errors |
| **Orchestrator** | `orchestrator.json` | Pod equity/drawdown/Sharpe, Lifecycle state, Allocation, PRC, Effective N, Correlation, Netting |
| **Anomaly Detection** | `anomaly.json` | KS statistic, P-value, RANSAC slope/decay, GBM drawdown depth/duration/severity, Fill rate |
| **On-chain Data** | `onchain.json` | Fetch rate/latency, Rows per source, Last success age, Cache size/refresh |

### Quick-Overview Dashboards (infra/)

`infra/grafana/dashboards/` 에 경량 quick-overview 대시보드 3개 (간결한 단일 페이지).

| Dashboard | File | 주요 패널 |
|-----------|------|----------|
| **Overview** | `overview.json` | Equity, Drawdown, Realized PnL, Open positions, Leverage, Signals vs Fills, Latency |
| **Strategy** | `strategy.json` | Per-strategy PnL, Rolling Sharpe, Profit Factor, Trade count, Pod Drawdown, Signal/Fill rates |
| **System** | `system.json` | Event loop lag, Active tasks, RSS memory, CPU%, FDs, WebSocket status, EventBus queue |

---

## Alert Runbook

### HighDrawdown

**원인**: 포트폴리오 손실이 고점 대비 10% 초과.
**대응**:

1. Grafana에서 포지션별 PnL 확인
1. 시장 전체 하락인지 단일 포지션 문제인지 판단
1. 필요 시 `/kill` 명령으로 전 포지션 청산

### APIUnhealthy

**원인**: Binance API 주문 실행(create_order/create_stop_market) 5회 연속 실패.
**대응**:

1. Binance 상태 페이지 확인
1. 네트워크 연결 상태 점검
1. Rate limit 초과 여부 확인 (`rate(mcbot_exchange_api_calls_total[1m])` 쿼리)
1. 자동 복구 대기 (성공 시 카운터 리셋)

**주의**: 이 alert는 주문 실행(write-ops) 연속 실패에만 반응합니다. 읽기 API
(fetch_positions, fetch_balance 등) 실패는 `APIReadFailures` alert로 별도 모니터링됩니다.

### EventsDropped

**원인**: EventBus 큐 가득 -> 드롭 가능 이벤트(BAR) 제거.
**대응**:

1. Queue Depth 패널에서 추이 확인
1. Handler Errors 증가 시 특정 핸들러 병목 점검
1. queue_size 설정 증가 검토

### WSDisconnected

**원인**: WebSocket 연결 끊김 (네트워크 문제 또는 거래소 유지보수).
**대응**:

1. 자동 재연결 대기 — exponential backoff (초기 1.0초 → 최대 60.0초)
1. 2분 이상 지속 시 봇 로그 확인
1. 전 심볼 동시 끊김이면 네트워크 문제
1. 연속 실패 10회(`_MAX_CONSECUTIVE_FAILURES`) 도달 시 CRITICAL RiskAlertEvent
   자동 발행 → Discord 알림 확인
1. CRITICAL 이후에도 복구 안 되면 프로세스 재시작 검토 (포지션 안전 확인 후)

### HighSlippage

**원인**: 주문 체결가가 기대가 대비 20bp 이상 차이.
**대응**:

1. 유동성 부족 코인인지 확인 (volume 패널)
1. 주문 크기가 시장 대비 과대한지 점검
1. 시장 급변 시 일시적 현상

### QueueCongestion

**원인**: EventBus 처리 속도 < 이벤트 발생 속도.
**대응**:

1. Bars Processed Rate 이상 없는지 확인
1. 핸들러 에러로 인한 지연 확인
1. 큐 크기 설정 검토

### HighEventLoopLag

**원인**: asyncio event loop 스케줄링 지연 > 1초.
**대응**:

1. CPU 사용률 확인 (CPU saturation)
1. Blocking I/O 호출이 없는지 로그 점검
1. 핸들러 처리 시간 확인

### HighMemoryUsage

**원인**: RSS 메모리 2GB 초과 (5분 지속).
**대응**:

1. 메모리 증가 추이 확인 (leak vs spike)
1. 대량 데이터 로드 여부 점검
1. GC 강제 실행 검토

### HighFDCount

**원인**: 열린 file descriptor 1000개 초과 (2분 지속).
**대응**:

1. `/proc/self/fd` (Linux) 또는 `/dev/fd` (macOS)에서 열린 FD 목록 확인
1. WebSocket 소켓 수 확인 — 심볼당 1개 기준, 비정상 누적 여부
1. 로그 파일 rotation 정상 동작 확인 (logrotate 설정)
1. SQLite DB 핸들 누수 여부 점검
1. FD count 추이 확인 — 단조 증가이면 leak, 주기적이면 burst

**참고**: 컨테이너 환경에서 `/proc/self/fd` 접근 불가 시 0 보고 + warning 로그 발생.
FD 고갈을 0으로 착각하지 않도록 로그 확인 필요.

### WSFrequentReconnects

**원인**: 5분 내 3회 이상 WS 재연결.
**대응**:

1. 네트워크 안정성 점검
1. 거래소 유지보수 공지 확인
1. 단일 심볼만이면 해당 마켓 이슈
1. 전 심볼 동시 재연결은 네트워크 장애 증거 — stagger delay(0.5초/심볼)로
   정상 시 재연결이 분산되므로, 동시 발생은 외부 요인 확실

### WSNoMessages

**원인**: 60초 이상 WS 메시지 수신 없음.
**대응**:

1. WS 연결 상태 확인 (`mcbot_exchange_ws_connected`)
1. 거래소 API 상태 확인
1. `rate(mcbot_ws_messages_received_total[5m])` 쿼리로 메시지 수신률 추이 확인
1. 데이터 피드 재시작 검토

**참고**: `ws_last_message_age_seconds` gauge는 30초 주기 갱신이므로 threshold 60초 +
for 1분 조건에서 실제 감지까지 60~90초 소요 가능. LiveDataFeed 내부
`_staleness_monitor()`(120초)가 별도 이중 감지 → CRITICAL 시 RiskAlertEvent → Discord

### DistributionDrift

**원인**: 라이브 수익률 분포가 백테스트 분포와 유의미하게 다름 (KS p < 0.05, 1일 지속).
**대응**:

1. 최근 시장 구조 변화 확인
1. 전략 파라미터 재최적화 검토
1. 1일 이상 지속 시 전략 교체 고려

### DistributionDriftCritical

**원인**: 라이브 수익률 분포가 백테스트 분포와 **매우** 유의미하게 다름 (KS p < 0.01, 1일 지속).
**대응**:

1. DistributionDrift WARNING 대응 항목 참조
1. p < 0.01은 1% 유의수준 이탈 — 시장 구조 변화 가능성 높음
1. 전략 파라미터 재최적화 또는 전략 교체 적극 검토

### StructuralDecay / StructuralDecayCritical

**원인**: RANSAC 기울기 <= 0 또는 conformal 하한 돌파 (1일 지속).

**2-tier PromQL:**

- **WARNING**: `mcbot_ransac_decay_detected == 1` (slope <= 0 **OR** level breach)
- **CRITICAL**: `mcbot_ransac_slope <= 0 AND mcbot_ransac_current_cumulative < mcbot_ransac_conformal_lower`
  (slope <= 0 **AND** level breach 동시 성립)

**대응**:

1. 전략 누적 수익률 추세 확인
1. GBM drawdown 결과와 교차 검증
1. WARNING 지속 시 파라미터 재최적화 검토
1. CRITICAL 도달 시 전략 retiring / 교체 검토

### GBMDrawdownWarning / GBMDrawdownCritical

**원인**: GBM 모형 기반 drawdown 깊이/지속기간이 95% CI를 초과.

- WARNING: depth **OR** duration 초과
- CRITICAL: depth **AND** duration 동시 초과

**대응**:

1. `mcbot_gbm_drawdown_depth` / `mcbot_gbm_drawdown_duration_days` gauge로 현재 상태 확인
1. 시장 전체 drawdown인지 단일 전략 문제인지 판단 (다른 Pod과 교차 비교)
1. DistributionDrift, StructuralDecay alert 동시 발생 여부 확인
1. CRITICAL 시 전략 파라미터 재최적화 또는 배분 축소 검토
1. GBM duration은 drift-ratio 기반 heuristic이므로, 극단 시장에서는
   Monte Carlo로 재검증 고려

### ExecutionLowFillRate / ExecutionConsecutiveRejections

**원인**: 실행 품질 이상 — fill rate 80% 미만 또는 연속 5건 이상 주문 거부.
**대응**:

1. `mcbot_execution_fill_rate` gauge로 현재 fill rate 확인
1. `mcbot_execution_consecutive_rejections` gauge로 연속 거부 횟수 확인
1. API 건강 상태 확인 (`APIUnhealthy` alert 동시 발생?)
1. 거부 사유 확인 — `mcbot_order_rejected_total` counter의 reason 라벨
1. MIN\_NOTIONAL 문제, 레버리지 초과, 심볼 상장폐지 등 원인별 대응
1. 모든 심볼의 fill/rejection이 단일 detector로 집계됨에 주의 —
   특정 심볼만 문제인지 전체인지 `mcbot_order_rejected_total{symbol=...}` 쿼리로 확인

### OnchainFetchHighFailureRate

**원인**: On-chain 데이터 수집 실패율 50% 초과 (24시간).
**대응**:

1. 외부 API 상태 확인 (DeFiLlama, CoinMetrics, Etherscan 등)
1. 네트워크 연결 상태 점검
1. Rate limit 초과 여부 확인
1. 자동 재시도 대기

### OnchainDataStale

**원인**: 특정 소스가 48시간 이상 데이터 미갱신.
**대응**:

1. `/onchain` Discord 명령으로 소스별 상태 확인
1. 해당 소스 API 정상 여부 수동 확인
1. `mcbot ingest onchain batch` 수동 실행 검토

### OnchainCacheEmpty

**원인**: Live 캐시에 on-chain 컬럼이 0개 (30분 지속).
**대응**:

1. Silver 데이터 존재 여부 확인 (`mcbot ingest onchain info`)
1. Silver 데이터 미존재 시 batch 수집 실행
1. LiveOnchainFeed 로그 확인

### OnchainFetchSlow

**원인**: On-chain fetch P95 지연시간 60초 초과 (INFO 수준).
**대응**:

1. 외부 API 응답 시간 확인 (DeFiLlama, CoinMetrics, Etherscan 등)
1. Rate limit 초과로 인한 throttle 여부 점검
1. `source` 라벨로 어떤 소스가 느린지 특정
1. Polling interval 조정 검토 (현재: DeFiLlama 6h, CoinMetrics 12h)
1. 네트워크 latency 점검

**참고**: INFO 수준이므로 운영 영향이 없으면 무시 가능. 지속 시 polling 간격 조정 또는 timeout 설정 검토.

### APIReadFailures

**원인**: 읽기 API(fetch_positions, fetch_balance 등) 실패율 0.1/s 초과 (2분 지속).
**대응**:

1. Binance 상태 페이지 확인
1. 네트워크 연결 상태 점검
1. `rate(mcbot_exchange_api_calls_total{endpoint=~"fetch_.*"}[5m])` 쿼리로 endpoint별 실패율 확인
1. 포지션 동기화 지연 여부 확인 (PositionReconciler drift)

### APILatencyHigh

**원인**: API P95 지연시간 5초 초과 (2분 지속). retry backoff 시간 포함.
**대응**:

1. Binance API 상태 확인
1. 네트워크 latency 점검 (ping, traceroute)
1. Exchange Health 대시보드에서 endpoint별 지연시간 분포 확인
1. retry 빈도 확인 — latency 급증은 retry backoff 포함 가능성

### APIRateLimitApproaching

**원인**: API 호출률이 900 req/min 초과 (Binance 한도 1200의 75%).
**대응**:

1. Exchange Health 대시보드에서 endpoint별 호출률 확인
1. 불필요한 fetch 호출 빈도 감소 검토
1. 심볼 수 과다 여부 확인 (16 심볼 × 30초 polling = 32 req/min 기준)
1. 한도 초과 시 IP 밴 위험 — 즉시 대응 필요

### BotRestarted

**원인**: `mcbot_uptime_seconds` gauge가 리셋됨 (OOM kill, crash, 수동 재시작).
**대응**:

1. 시스템 로그에서 OOM kill 여부 확인 (`dmesg`, `journalctl`)
1. 봇 로그에서 crash traceback 확인
1. 자동 재시작(systemd 등) 동작 확인
1. 반복 발생 시 메모리 제한 조정 또는 원인 디버깅

### HeartbeatStale

**원인**: 120초 이상 HeartbeatEvent 미수신. Event loop hung 또는 프로세스 unresponsive.
**대응**:

1. 프로세스 상태 확인 (`ps`, CPU 사용률)
1. `HighEventLoopLag` alert 동시 발생 여부 확인
1. Blocking I/O 또는 deadlock 가능성 점검
1. 프로세스 재시작 검토 (포지션 안전 확인 후)

### HandlerErrorsHigh

**원인**: EventBus handler 에러율 0.01/s 초과 (2분 지속). 이벤트 체인 중단 위험.
**대응**:

1. 봇 로그에서 "Handler error" 로그 확인 (handler 이름 + event_type 포함)
1. 특정 이벤트 타입에 집중되는지 확인
1. 외부 의존성(API, DB) 장애 여부 점검
1. handler 코드 버그 시 핫픽스 배포

### CircuitBreakerTriggered

**원인**: RiskManager가 system_stop_loss 임계값 도달로 전량 청산 실행.
**대응**:

1. **즉시 확인**: 거래소에서 포지션 상태 확인 (전량 청산 완료 여부)
1. Grafana에서 drawdown 추이 확인 (급락 원인 분석)
1. 시장 전체 급락인지 전략 문제인지 판단
1. 봇 재시작 전 원인 분석 완료 필수 (CB는 세션당 1회만 발동)

### QueueCongestionCritical

**원인**: EventBus 큐 80% 이상 사용 (8,000/10,000). 이벤트 드롭 임박.
**대응**:

1. `QueueCongestion` (5,000 WARNING)에서 이미 경고되었는지 확인
1. handler 처리 지연 원인 점검 (CPU, 외부 API 지연)
1. DROPPABLE 이벤트 드롭 발생 여부 확인 (`EventsDropped` alert)
1. 큐 크기 설정 증가 또는 handler 최적화 검토

### StrategyPnlNegative

**원인**: 특정 전략 누적 realized PnL이 -$500 이하 (1시간 지속).
**대응**:

1. 전략별 PnL 추이 확인 (Strategy Performance 대시보드)
1. 시장 환경 변화 vs 전략 결함 판단
1. Orchestrator probation/retired 전환 검토

### StrategySignalSilent

**원인**: 24시간 시그널 미발생.
**대응**:

1. 데이터 피드 정상 여부 확인 (`mcbot_last_bar_age_seconds`)
1. bar 수신 여부 확인 (`mcbot_bars_total` rate)
1. 전략 로직 데드락 점검 (handler errors 확인)

### HighCPUUsage

**원인**: CPU 사용률 80% 초과 (2분 지속). Compute-bound handler 또는 blocking I/O.
**대응**:

1. `HighEventLoopLag` alert 동시 발생 여부 확인 (CPU saturation → loop lag)
1. GC pressure 확인 (`HighGCPressure` alert)
1. handler 처리 시간 로그 점검 (특정 이벤트에 과도한 연산)
1. ThreadPoolExecutor 사용 여부 확인 (CPU-bound 작업은 executor 분리)

### HighTaskCount

**원인**: asyncio Task 200개 초과 (5분 지속). Task 누수 가능성.
**대응**:

1. `asyncio.all_tasks()` 목록 확인 — 어떤 coroutine이 다수인지
1. Task 생성 후 `await`/`cancel()` 없이 참조 소실된 건 없는지 점검
1. WS 재연결 시 이전 Task cancel 확인
1. 추이 확인: 단조 증가이면 leak, 주기적이면 burst

### HighGCPressure

**원인**: Gen2 GC 수집 빈도 0.1/s 초과 (5분 지속). 대량 객체 생성/소멸.
**대응**:

1. `mcbot_event_loop_lag_seconds`와 상관관계 확인 (gen2 GC 동안 event loop stall)
1. 대량 DataFrame 생성/파기 패턴 점검 (strategy collect 주기)
1. 메모리 프로파일링 검토 (`tracemalloc`, `objgraph`)
1. 장기 지속 시 객체 풀링 또는 데이터 파이프라인 최적화

### NettingLowOffset

**원인**: Pod간 포지션 상쇄 비율 10% 미만 (1시간 지속). 방향성 편중 위험.
**대응**:

1. Pod별 포지션 방향 확인 (Orchestrator 대시보드)
1. 시장 추세 vs 의도치 않은 편중 판단
1. 의도된 동일 방향이면 alert 무시 가능
1. 의도치 않은 편중이면 Pod 전략 구성 검토

### PodInProbation

**원인**: Pod가 probation 상태에 진입 (성과 하락 감지).
**대응**:

1. Pod 성과/Sharpe 확인 (Strategy Performance 대시보드)
1. `DistributionDrift`, `StructuralDecay` alert 동시 발생 여부 확인
1. 파라미터 재최적화 검토
1. 유예 기간 내 미회복 시 자동 RETIRED 전환 — 수동 개입 불필요

### HighGrossExposure

**원인**: Pod간 총 gross exposure 3.0 초과 (5분 지속). 레버리지 과다 위험.
**대응**:

1. Pod별 allocation 확인 (`mcbot_pod_allocation_fraction`)
1. 레버리지 배수 점검 (`mcbot_aggregate_leverage`)
1. offset\_ratio 교차 확인 — 높은 gross + 높은 offset이면 실질 risk 낮음
1. 지속 시 allocation 상한 조정 검토

### HighAggregateLeverage

**원인**: 포트폴리오 총 레버리지 5x 초과 (1분 지속). 강제 청산 위험.
**대응**:

1. `mcbot_aggregate_leverage` gauge로 현재 레버리지 확인
1. 포지션별 notional 확인 — 어떤 심볼이 과도한지 특정
1. allocation 축소 또는 포지션 부분 청산 검토
1. `HighGrossExposure` alert 동시 발생 여부 확인
1. Binance 유지보수 마진율 확인 — 강제 청산 임계값까지 여유 점검

### LiveAPIBlocked

**원인**: API unhealthy 상태에서 주문이 무음 차단됨.
**대응**:

1. `APIUnhealthy` alert 동시 발생 확인 — API 연속 실패 5회 이상
1. 거래소 API 상태 페이지 확인 (Binance status)
1. 네트워크 연결 상태 점검
1. 차단된 주문 수 확인 (`mcbot_live_api_blocked_total` counter)
1. API 복구 시 자동 해제 — 수동 개입 불필요 (consecutive_failures 리셋)

**주의**: 이 alert는 주문이 **무음으로** 차단됨을 의미합니다. 시그널은 정상 생성되지만
실행되지 않으므로, 복구 후 포지션 상태 확인이 중요합니다.

### HighMarginUtilization

**원인**: 마진 사용률 80% 초과 (2분 지속). Margin call 사전 경고.
**대응**:

1. `mcbot_margin_used_usdt` / `mcbot_equity_usdt` 비율 확인
1. 포지션 크기 축소 검토
1. `mcbot_aggregate_leverage` 교차 확인 — 레버리지와 마진 사용률 상관관계
1. 추가 마진(자금 이체) 검토
1. 90% 초과 시 긴급 포지션 축소 권장

### HighThreadCount

**원인**: 활성 스레드 50개 초과 (5분 지속). Thread leak 가능성.
**대응**:

1. `threading.active_count()` 및 `threading.enumerate()`로 스레드 목록 확인
1. ThreadPoolExecutor 상태 점검 — worker 미회수 여부
1. Blocking I/O 호출 점검 (동기 API 호출 등)
1. 추이 확인: 단조 증가이면 leak, 주기적이면 burst
1. `HighCPUUsage`, `HighEventLoopLag`와 상관관계 확인

### LiveFillParseFailure

**원인**: 거래소 fill 응답 파싱 실패. 포지션 추적 부정확 위험.
**대응**:

1. 봇 로그에서 파싱 에러 traceback 확인
1. 거래소 API 응답 형식 변경 여부 점검 (CCXT 라이브러리 업데이트 필요?)
1. 실제 포지션과 내부 추적 상태 비교 (reconciliation drift 확인)
1. 지속 시 CCXT 버전 업데이트 또는 파서 수정 필요

### LiveFrequentMinNotionalSkip

**원인**: MIN\_NOTIONAL 미달로 주문 스킵이 1시간 내 3회 이상 발생.
**대응**:

1. 해당 심볼 확인 (`mcbot_live_min_notional_skip_total{symbol=...}`)
1. 심볼별 최소 주문 크기(MIN\_NOTIONAL) 확인
1. allocation 금액이 최소 주문 크기 대비 충분한지 검토
1. 소액 포지션 심볼 제거 또는 allocation 상향 검토

**참고**: INFO 수준이므로 즉시 대응 불필요. 빈번 발생 시 자본 효율성 저하 원인.

---

## Infrastructure

### Docker Compose

```
docker-compose.yaml
├── mc-bot          봇 본체 (:8000/metrics)
├── prometheus       메트릭 수집 (:9090)
├── grafana          대시보드 (:3000)
└── node-exporter    호스트 메트릭 (:9100)
```

### File Structure

```
monitoring/
├── prometheus.yml                   Scrape 설정 (10s/15s)
├── alerts.yml                       45개 Alert Rule
├── Dockerfile.prometheus
├── Dockerfile.grafana
└── grafana/
    ├── dashboards/
    │   ├── trading.json             Overview 대시보드
    │   ├── strategy.json            전략 성과
    │   ├── execution.json           실행 품질
    │   ├── exchange.json            거래소 API 건강
    │   ├── system.json              시스템 건강
    │   ├── regime.json              마켓 레짐
    │   ├── orchestrator.json        Pod/Portfolio 관리
    │   ├── anomaly.json             이상 탐지 (KS/RANSAC/GBM)
    │   └── onchain.json             On-chain 데이터 건강
    └── provisioning/
        ├── dashboards/default.yml   대시보드 자동 프로비저닝
        └── datasources/prometheus.yml
```

---

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
| **GBM** | Geometric Brownian Motion. 수익률을 확률 과정으로 모델링 |
| **KS Test** | Kolmogorov-Smirnov test. 두 분포의 동일성 검정 |
| **RANSAC** | Random Sample Consensus. Outlier-robust 회귀 추정 |
| **Conformal Prediction** | 비모수적 예측 구간 생성 기법 |

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
