# MC Coin Bot — Monitoring Reference

Prometheus + Grafana 기반 모니터링 시스템 레퍼런스.

## Architecture

```
mc-bot (:8000/metrics)
    -> Prometheus (scrape 10s)
        -> Grafana (6 dashboards)
        -> Alertmanager (alerts.yml, 13 rules)

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

---

## Metric Layers

### Layer 1: Order Execution

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_orders_total` | Counter | symbol, side, order_type, status | 주문 건수 (status: ack/filled/rejected) |
| `mcbot_order_latency_seconds` | Histogram | symbol | 주문요청->체결 지연시간 |
| `mcbot_slippage_bps` | Histogram | symbol, side | 슬리피지 (basis points) |
| `mcbot_order_rejected_total` | Counter | symbol, reason | 거부 주문 |
| `mcbot_fees_usdt_total` | Counter | symbol | 누적 수수료 (USDT) |
| `mcbot_fills` | Counter | symbol, side | 체결 건수 |
| `mcbot_signals` | Counter | symbol | 시그널 건수 |
| `mcbot_bars` | Counter | timeframe | 처리된 bar 수 |

### Layer 2: Position & PnL

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_equity_usdt` | Gauge | -- | 포트폴리오 총 가치 |
| `mcbot_drawdown_pct` | Gauge | -- | 고점 대비 낙폭 |
| `mcbot_cash_usdt` | Gauge | -- | 가용 현금 |
| `mcbot_open_positions` | Gauge | -- | 열린 포지션 수 |
| `mcbot_position_size` | Gauge | symbol | 포지션 수량 |
| `mcbot_position_notional_usdt` | Gauge | symbol | 포지션 명목 금액 |
| `mcbot_unrealized_pnl_usdt` | Gauge | symbol | 미실현 손익 |
| `mcbot_realized_pnl_usdt_total` | Counter | symbol | 누적 실현 손익 |
| `mcbot_aggregate_leverage` | Gauge | -- | 포트폴리오 총 레버리지 |
| `mcbot_margin_used_usdt` | Gauge | -- | 사용 중인 마진 |

### Layer 3: Exchange API

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_exchange_api_calls_total` | Counter | endpoint, status | API 호출 수 |
| `mcbot_exchange_api_latency_seconds` | Histogram | endpoint | API 응답 시간 |
| `mcbot_exchange_ws_connected` | Gauge | symbol | WS 연결 상태 (1/0) |
| `mcbot_exchange_consecutive_failures` | Gauge | -- | 연속 API 실패 횟수 |

### Layer 4: Bot Health

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_uptime_seconds` | Gauge | -- | 봇 가동 시간 |
| `mcbot_heartbeat_timestamp` | Gauge | -- | 마지막 heartbeat |
| `mcbot_errors_total` | Counter | component, error_type | 에러 수 |
| `mcbot_eventbus_queue_depth` | Gauge | -- | 큐 대기 이벤트 수 |
| `mcbot_eventbus_events_dropped_total` | Counter | -- | 이벤트 드롭 수 |
| `mcbot_eventbus_handler_errors_total` | Counter | -- | 핸들러 에러 수 |
| `mcbot_circuit_breaker` | Counter | -- | 서킷 브레이커 발동 |
| `mcbot_risk_alerts` | Counter | level | 리스크 알림 |

### Layer 5: Per-Strategy

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_strategy_pnl_usdt` | Gauge | strategy | 전략별 PnL (USDT) |
| `mcbot_strategy_drawdown_pct` | Gauge | strategy | 전략별 drawdown (%) |
| `mcbot_strategy_signals_total` | Counter | strategy, side | 전략별 시그널 수 |
| `mcbot_strategy_fills_total` | Counter | strategy, side | 전략별 체결 수 |
| `mcbot_strategy_win_rate` | Gauge | strategy | 전략별 승률 (rolling 20) |
| `mcbot_strategy_sharpe_rolling` | Gauge | strategy | Rolling Sharpe (30d) |

### Layer 6: WebSocket

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_ws_reconnects_total` | Counter | symbol | WS 재연결 횟수 |
| `mcbot_ws_last_message_age_seconds` | Gauge | symbol | 마지막 메시지 후 경과 시간 |
| `mcbot_ws_messages_received_total` | Counter | symbol | 수신 메시지 수 |
| `mcbot_last_bar_age_seconds` | Gauge | symbol | 마지막 bar 수신 후 경과 시간 |

### Layer 7: Process & Event Loop

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_event_loop_lag_seconds` | Gauge | -- | Event loop 스케줄링 지연 |
| `mcbot_active_tasks` | Gauge | -- | 활성 asyncio Task 수 |
| `mcbot_process_memory_rss_bytes` | Gauge | -- | RSS 메모리 사용량 |
| `mcbot_process_cpu_percent` | Gauge | -- | CPU 사용률 (%) |
| `mcbot_process_open_fds` | Gauge | -- | 열린 file descriptor 수 |

### Layer 8: Anomaly Detection

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_gbm_drawdown_severity` | Gauge | strategy | GBM 심각도 (0=NORMAL, 1=WARNING, 2=CRITICAL) |
| `mcbot_distribution_p_value` | Gauge | strategy | KS test p-value |
| `mcbot_ransac_decay_detected` | Gauge | strategy | RANSAC 구조적 쇠퇴 감지 (0/1) |
| `mcbot_ransac_slope` | Gauge | strategy | RANSAC 추정 기울기 |

### Layer 9: On-chain Data

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mcbot_onchain_fetch_total` | Counter | source, status | On-chain fetch 시도 (success/failure/empty) |
| `mcbot_onchain_fetch_latency_seconds` | Histogram | source | Fetch 소요 시간 |
| `mcbot_onchain_fetch_rows` | Gauge | source, name | 마지막 fetch 반환 행 수 |
| `mcbot_onchain_last_success_timestamp` | Gauge | source | 마지막 성공 Unix timestamp |
| `mcbot_onchain_cache_size` | Gauge | symbol | 심볼별 캐시된 on-chain 컬럼 수 |
| `mcbot_onchain_cache_refresh_total` | Counter | status | 캐시 refresh 횟수 (success/failure) |

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
| EventsDropped | `rate(events_dropped[5m]) > 0` | 2m | WARNING |
| WSDisconnected | `mcbot_exchange_ws_connected == 0` | 2m | CRITICAL |
| HighSlippage | `histogram_quantile(0.95, slippage_bps) > 20` | 5m | WARNING |
| QueueCongestion | `mcbot_eventbus_queue_depth > 5000` | 1m | WARNING |
| HighEventLoopLag | `mcbot_event_loop_lag_seconds > 1.0` | 1m | WARNING |
| HighMemoryUsage | `mcbot_process_memory_rss_bytes > 2GB` | 5m | WARNING |
| HighFDCount | `mcbot_process_open_fds > 1000` | 2m | WARNING |
| WSFrequentReconnects | `increase(mcbot_ws_reconnects_total[5m]) >= 3` | -- | WARNING |
| WSNoMessages | `mcbot_ws_last_message_age_seconds > 60` | 1m | CRITICAL |
| DistributionDrift | `mcbot_distribution_p_value < 0.05` | 1d | WARNING |
| StructuralDecay | `mcbot_ransac_decay_detected == 1` | 1d | CRITICAL |
| OnchainFetchHighFailureRate | `rate(mcbot_onchain_fetch_total{status="failure"}[24h]) / rate(mcbot_onchain_fetch_total[24h]) > 0.5` | 1h | WARNING |
| OnchainDataStale | `time() - mcbot_onchain_last_success_timestamp > 172800` | 1h | WARNING |
| OnchainCacheEmpty | `mcbot_onchain_cache_size == 0` | 30m | WARNING |
| OnchainFetchSlow | `histogram_quantile(0.95, mcbot_onchain_fetch_latency_seconds) > 60` | 5m | INFO |

---

## Grafana Dashboards

`monitoring/grafana/dashboards/` 에 JSON으로 버전 관리.

| Dashboard | File | 주요 패널 |
|-----------|------|----------|
| **Trading Overview** | `trading.json` | Equity curve, Drawdown, Open positions, Today PnL, Uptime |
| **Strategy Performance** | `strategy.json` | Per-strategy PnL, Drawdown, Signal frequency, Win rate |
| **Execution Quality** | `execution.json` | Fill latency, Slippage distribution, Fee accumulation |
| **Exchange Health** | `exchange.json` | API latency, WS connection, Rate limit, Consecutive failures |
| **System Health** | `system.json` | Event loop lag, Queue depth, Memory, Active tasks, FD count |
| **Market Regime** | `regime.json` | Funding rate, OI changes, Regime score |

---

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
**원인**: EventBus 큐 가득 -> 드롭 가능 이벤트(BAR) 제거.
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

### HighEventLoopLag
**원인**: asyncio event loop 스케줄링 지연 > 1초.
**대응**:
1. CPU 사용률 확인 (CPU saturation)
2. Blocking I/O 호출이 없는지 로그 점검
3. 핸들러 처리 시간 확인

### HighMemoryUsage
**원인**: RSS 메모리 2GB 초과 (5분 지속).
**대응**:
1. 메모리 증가 추이 확인 (leak vs spike)
2. 대량 데이터 로드 여부 점검
3. GC 강제 실행 검토

### WSFrequentReconnects
**원인**: 5분 내 3회 이상 WS 재연결.
**대응**:
1. 네트워크 안정성 점검
2. 거래소 유지보수 공지 확인
3. 단일 심볼만이면 해당 마켓 이슈

### WSNoMessages
**원인**: 60초 이상 WS 메시지 수신 없음.
**대응**:
1. WS 연결 상태 확인 (`mcbot_exchange_ws_connected`)
2. 거래소 API 상태 확인
3. 데이터 피드 재시작 검토

### DistributionDrift
**원인**: 라이브 수익률 분포가 백테스트 분포와 유의미하게 다름 (KS p < 0.05, 1일 지속).
**대응**:
1. 최근 시장 구조 변화 확인
2. 전략 파라미터 재최적화 검토
3. 1일 이상 지속 시 전략 교체 고려

### StructuralDecay
**원인**: RANSAC 기울기 <= 0 또는 conformal 하한 돌파 (1일 지속).
**대응**:
1. 전략 누적 수익률 추세 확인
2. GBM drawdown 결과와 교차 검증
3. 전략 retiring / 교체 검토

### OnchainFetchHighFailureRate
**원인**: On-chain 데이터 수집 실패율 50% 초과 (24시간).
**대응**:
1. 외부 API 상태 확인 (DeFiLlama, CoinMetrics, Etherscan 등)
2. 네트워크 연결 상태 점검
3. Rate limit 초과 여부 확인
4. 자동 재시도 대기

### OnchainDataStale
**원인**: 특정 소스가 48시간 이상 데이터 미갱신.
**대응**:
1. `/onchain` Discord 명령으로 소스별 상태 확인
2. 해당 소스 API 정상 여부 수동 확인
3. `mcbot ingest onchain batch` 수동 실행 검토

### OnchainCacheEmpty
**원인**: Live 캐시에 on-chain 컬럼이 0개 (30분 지속).
**대응**:
1. Silver 데이터 존재 여부 확인 (`mcbot ingest onchain info`)
2. Silver 데이터 미존재 시 batch 수집 실행
3. LiveOnchainFeed 로그 확인

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
├── alerts.yml                       13개 Alert Rule
├── Dockerfile.prometheus
├── Dockerfile.grafana
└── grafana/
    ├── dashboards/
    │   ├── trading.json             Overview 대시보드
    │   ├── strategy.json            전략 성과
    │   ├── execution.json           실행 품질
    │   ├── exchange.json            거래소 API 건강
    │   ├── system.json              시스템 건강
    │   └── regime.json              마켓 레짐
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
