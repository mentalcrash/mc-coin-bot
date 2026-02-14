# Live Hardening Plan — 모니터링 & 알림 고도화

> **목적:** 라이브 트레이딩 운용에 필요한 관측성(Observability), 알림, 이상 감지 시스템을 고도화하여
> "시스템이 잘 작동하고 있는가?"와 "뭔가 잘못되고 있는가?"를 실시간으로 파악할 수 있도록 한다.

---

## 1. 현재 상태 (As-Is)

### 1.1 이미 구현된 것

| 영역 | 구현 상태 | 핵심 파일 |
|------|----------|----------|
| **Prometheus Metrics** (40+ metrics, 5 Layer) | ✅ 완료 | `src/monitoring/metrics.py` |
| **Discord Bot** (slash commands: /status, /balance, /kill) | ✅ 완료 | `src/notification/bot.py` |
| **Notification Pipeline** (Queue + SpamGuard + Retry) | ✅ 완료 | `src/notification/queue.py`, `engine.py` |
| **Health Check** (1h Heartbeat / 4h Regime / 8h Strategy) | ✅ 완료 | `src/notification/health_scheduler.py` |
| **Daily/Weekly Reports** (Equity curve + Charts → Discord) | ✅ 완료 | `src/notification/report_scheduler.py` |
| **Orchestrator Notifications** (Lifecycle / Rebalance / Risk) | ✅ 완료 | `src/notification/orchestrator_engine.py` |
| **PageHinkley Degradation** (CUSUM drift detection) | ✅ 완료 | `src/orchestrator/degradation.py` |
| **OTel Logging** (trace_id/span_id injection) | ✅ 완료 | `src/logging/sinks/otel.py` |
| **CircuitBreaker** (System stop-loss → 전 포지션 청산) | ✅ 완료 | `src/eda/risk_manager.py` |
| **Graceful Shutdown** (SIGTERM → feed/bot/queue drain) | ✅ 완료 | `src/eda/live_runner.py` |

### 1.2 Gap 분석 — 누락된 것

| Gap | 설명 | 영향 |
|-----|------|------|
| **G1. Per-Strategy Metrics** | 전략별 PnL/Sharpe/drawdown 분리 추적 없음 | 멀티 전략 시 어떤 전략이 문제인지 불명 |
| **G2. 봇 기동/종료 알림** | 서버 on/off 시 Discord 알림 없음 | 예상치 못한 종료를 감지 못함 |
| **G3. Event Loop Health** | asyncio lag, 활성 Task 수 미추적 | 시스템 병목 감지 불가 |
| **G4. WS 상세 Metrics** | 재연결 횟수, 메시지 lag 미추적 | 데이터 지연/유실 감지 불가 |
| **G5. 전략별 성과 알림** | 8h StrategyHealth는 시스템 레벨 요약만 | 개별 전략 degradation 즉시 알림 없음 |
| **G6. Slippage/Fee 임계치 알림** | Prometheus에만 기록, Discord 알림 없음 | 실행 품질 저하 시 대응 지연 |
| **G7. Data Freshness 알림** | EventBus stale symbol 감지하나 알림 없음 | 데이터 지연에 의한 잘못된 시그널 |
| **G8. Grafana Dashboard** | 메트릭 있으나 dashboard 코드 없음 | 시각적 모니터링 불가 |
| **G9. GBM Drawdown Monitor** | PageHinkley만 있음, 이론적 한계 검증 없음 | 전략 교체 시점 판단 부정확 |
| **G10. Process Metrics** | Memory, GC, CPU 미추적 | OOM/메모리 누수 감지 불가 |
| **G11. Interactive Discord** | 조회 명령만 있음, 전략 on/off 불가 | 즉각적 대응 불가 |
| **G12. Reconciliation 알림** | PositionReconciler drift 감지하나 알림 미흡 | 포지션 불일치 감지 지연 |

---

## 2. 고도화 로드맵

### Phase L1: 필수 알림 강화 (Critical Path)

> 라이브 운용 전 반드시 필요한 항목

#### L1-1. 봇 Lifecycle 알림 (`G2`)

봇이 **시작/종료/비정상 종료** 시 Discord 알림 전송.

```
시작 알림:
┌────────────────────────────────┐
│ 🟢 MC Coin Bot Started        │
│ Mode:     LIVE                 │
│ Strategies: CTREND, Anchor-Mom │
│ Symbols:  8 assets             │
│ Capital:  $10,000              │
│ Time:     2026-02-14 09:00 UTC │
└────────────────────────────────┘

종료 알림:
┌────────────────────────────────┐
│ 🔴 MC Coin Bot Stopped        │
│ Reason:   SIGTERM (graceful)   │
│ Uptime:   12h 34m              │
│ Final Equity: $10,150          │
│ Today PnL: +$150 (+1.5%)      │
│ Open Positions: 0 (all closed) │
└────────────────────────────────┘
```

**구현 위치:** `LiveRunner.run()` 시작/종료 지점에 embed 전송
**Crash 감지:** Python `atexit` + `sys.excepthook` 등록 → 비정상 종료 시 "CRASH" 알림
**노력:** 낮음 | **가치:** 매우 높음

#### L1-2. Per-Strategy Performance Tracking (`G1`, `G5`)

Prometheus metrics에 `strategy` label 추가 + 전략별 성과 Discord 알림.

**추가 Metrics:**

```python
# 전략별 메트릭 (strategy label)
mcbot_strategy_pnl_usdt           {strategy="ctrend"}        # Gauge: 전략별 PnL
mcbot_strategy_drawdown_pct       {strategy="ctrend"}        # Gauge: 전략별 drawdown
mcbot_strategy_signals_total      {strategy="ctrend", side}  # Counter: 시그널 수
mcbot_strategy_fills_total        {strategy="ctrend", side}  # Counter: 체결 수
mcbot_strategy_win_rate           {strategy="ctrend"}        # Gauge: 승률 (rolling 20)
mcbot_strategy_sharpe_rolling     {strategy="ctrend"}        # Gauge: Rolling Sharpe (30d)
```

**전략 성과 알림 (매 8h):**

```
┌──────────────────────────────────┐
│ 📊 Strategy Health Report        │
├──────────────────────────────────┤
│ CTREND                           │
│  Sharpe: 1.42  DD: 3.2%  WR: 62%│
│  Today: +$85 (+0.85%)           │
│  Status: ✅ HEALTHY              │
├──────────────────────────────────┤
│ Anchor-Mom                       │
│  Sharpe: 0.31  DD: 8.7%  WR: 48%│
│  Today: -$42 (-0.42%)           │
│  Status: ⚠️ DEGRADING (PH: 2/3) │
└──────────────────────────────────┘
```

**노력:** 중간 | **가치:** 매우 높음

#### L1-3. Execution Quality 알림 (`G6`)

슬리피지/수수료/체결 지연이 임계값 초과 시 Discord WARNING.

**임계값:**

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Slippage (P95) | > 15 bps | WARNING |
| Slippage (단건) | > 30 bps | CRITICAL |
| Fill Latency (P95) | > 5s | WARNING |
| Fill Latency (단건) | > 10s | CRITICAL |
| Hourly Fee / Equity | > 0.1% | WARNING |

**구현:** `MetricsExporter`의 fill 핸들러에서 임계값 검사 → `RiskAlertEvent` 발행
**노력:** 낮음 | **가치:** 높음

#### L1-4. Data Freshness 알림 (`G7`)

특정 symbol의 마지막 bar 수신으로부터 `2 × timeframe` 이상 경과 시 알림.

```python
mcbot_last_bar_age_seconds{symbol="BTC/USDT"}  # Gauge: 마지막 bar 이후 경과 시간
```

**임계값:**

| Timeframe | Stale Threshold | Severity |
|-----------|----------------|----------|
| 1m | > 3m | WARNING |
| 1h | > 2h | WARNING |
| 1d | > 25h | WARNING |
| Any | > 3 × TF | CRITICAL |

**구현:** `HealthCheckScheduler` heartbeat 루프에 stale symbol 검사 추가
**노력:** 낮음 | **가치:** 높음

---

### Phase L2: 관측성 강화 (Observability)

> 시스템이 잘 동작하는지 깊이 있게 파악

#### L2-1. Event Loop & Process Metrics (`G3`, `G10`)

asyncio event loop과 프로세스 상태 모니터링.

**추가 Metrics:**

```python
# Event Loop
mcbot_event_loop_lag_seconds      # Gauge: event loop 지연
mcbot_active_tasks                # Gauge: 활성 asyncio Task 수

# Process
mcbot_process_memory_rss_bytes    # Gauge: RSS 메모리
mcbot_process_cpu_percent         # Gauge: CPU 사용률
mcbot_process_open_fds            # Gauge: 열린 file descriptor 수
```

**Event Loop Lag 측정:**

```python
async def _monitor_event_loop(interval: float = 5.0) -> None:
    while True:
        t0 = time.monotonic()
        await asyncio.sleep(interval)
        lag = time.monotonic() - t0 - interval
        event_loop_lag_gauge.set(max(lag, 0.0))
        active_tasks_gauge.set(len(asyncio.all_tasks()))
```

**Alert:**
- Event loop lag > 1s → WARNING
- RSS memory > 2GB → WARNING
- Open FDs > 1000 → WARNING

**노력:** 낮음 | **가치:** 중간

#### L2-2. WebSocket 상세 Metrics (`G4`)

WS 연결 상태를 더 세밀하게 추적.

```python
mcbot_ws_reconnects_total{symbol}           # Counter: 재연결 횟수
mcbot_ws_last_message_age_seconds{symbol}   # Gauge: 마지막 메시지 후 경과
mcbot_ws_messages_received_total{symbol}    # Counter: 수신 메시지 수
```

**Alert:**
- 5분 내 3회 이상 재연결 → WARNING
- 메시지 수신 0건 (1분간) → CRITICAL

**구현:** `LiveDataFeed`의 WebSocket callback에 계측 추가
**노력:** 낮음 | **가치:** 중간

#### L2-3. Grafana Dashboard as Code (`G8`)

Prometheus metrics를 시각화하는 Grafana dashboard를 JSON으로 버전 관리.

**Dashboard 구성:**

```
1. Overview
   - Equity curve (timeseries)
   - Current drawdown (gauge)
   - Open positions (table)
   - Bot status / uptime (stat)
   - Today PnL (stat)

2. Strategy Performance
   - Per-strategy PnL (timeseries)
   - Per-strategy drawdown (timeseries)
   - Signal frequency by strategy (bar chart)
   - Win rate trend (timeseries)

3. Execution Quality
   - Fill latency percentiles (heatmap)
   - Slippage distribution (histogram)
   - Fee accumulation (timeseries)
   - Order status breakdown (pie)

4. Exchange Health
   - API latency (timeseries)
   - WS connection status (state timeline)
   - Rate limit headroom (gauge)
   - Consecutive failures (timeseries)

5. System Health
   - Event loop lag (timeseries)
   - EventBus queue depth (timeseries)
   - Memory usage (timeseries)
   - Active async tasks (timeseries)

6. Market Regime
   - Funding rate per symbol (timeseries)
   - OI changes (timeseries)
   - Regime score (gauge)
```

**파일 구조:**

```
infra/grafana/
├── dashboards/
│   ├── overview.json
│   ├── strategy.json
│   ├── execution.json
│   ├── exchange.json
│   ├── system.json
│   └── regime.json
└── provisioning/
    └── dashboards.yaml
```

**노력:** 중간 | **가치:** 높음

#### L2-4. Position Reconciliation 알림 강화 (`G12`)

`PositionReconciler`의 drift 감지 결과를 Discord에 전송.

```
┌──────────────────────────────────┐
│ ⚠️ Position Drift Detected       │
├──────────────────────────────────┤
│ BTC/USDT                         │
│  Expected: 0.050 BTC (LONG)     │
│  Actual:   0.048 BTC (LONG)     │
│  Drift:    4.0%                 │
│  Action:   Auto-corrected       │
├──────────────────────────────────┤
│ ETH/USDT                         │
│  Expected: 0.00 (FLAT)          │
│  Actual:   0.10 ETH (LONG)     │
│  Drift:    ORPHAN POSITION      │
│  Action:   Manual review needed │
└──────────────────────────────────┘
```

**노력:** 낮음 | **가치:** 높음

---

### Phase L3: 고급 이상 감지 (Anomaly Detection)

> 의도하지 않은 방향으로 가고 있을 때 감지

#### L3-1. GBM Drawdown Monitor (`G9`)

전략의 PnL을 Geometric Brownian Motion으로 모델링하여 현재 drawdown이 **통계적으로 비정상**인지 검증.

**원리:**
1. 백테스트 기간의 일일 수익률로 drift(μ)와 volatility(σ) 추정
2. 95% CI에서 예상 최대 drawdown depth와 duration 산출
3. 실제 drawdown이 이론적 한계 초과 → WARNING/CRITICAL

**PageHinkley와의 차이:**
- PageHinkley: mean-shift 감지 (방향 변화)
- GBM Monitor: depth/duration 정상 범위 검증 (크기 판단)

**구현 구조:**

```python
class GBMDrawdownMonitor:
    """GBM 기반 drawdown 정상 범위 검증."""

    def __init__(self, mu: float, sigma: float, confidence: float = 0.95) -> None:
        self.mu = mu          # 일일 drift
        self.sigma = sigma    # 일일 volatility
        self.confidence = confidence

    def expected_max_drawdown(self, n_days: int) -> float:
        """N일 동안 95% CI 최대 drawdown 추정."""
        ...

    def is_drawdown_abnormal(
        self, current_dd: float, dd_duration_days: int
    ) -> bool:
        """현재 drawdown이 GBM 95% CI를 벗어났는지."""
        ...
```

**Alert:**

| 조건 | Severity |
|------|----------|
| DD depth > 95% CI expected max | WARNING |
| DD duration > 95% CI expected max | WARNING |
| Both depth AND duration exceed | CRITICAL → 전략 점검 권고 |

**노력:** 중간 | **가치:** 높음

#### L3-2. Execution Anomaly Detection

실행 품질의 이상 패턴 감지.

**검사 항목:**

| 항목 | 정상 기준 | 이상 판정 |
|------|----------|----------|
| Signal → Fill 시간 | < 2 × avg | > 3 × avg |
| 연속 Rejection | < 2건 | ≥ 3건 연속 |
| Fill Rate (1h) | > 95% | < 80% |
| Slippage 추세 | Stable | 3건 연속 증가 |

**구현:** `AnalyticsEngine`에 rolling window 통계 + 임계값 검사
**노력:** 중간 | **가치:** 중간

#### L3-3. Interactive Discord Commands (`G11`)

조회를 넘어 **대응**까지 가능한 Discord 명령어 확장.

**추가 Slash Commands:**

| Command | 설명 | 확인 필요 |
|---------|------|----------|
| `/strategies` | 전략별 현재 상태 + 성과 요약 | No |
| `/strategy <name>` | 특정 전략 상세 (포지션, PnL, signals) | No |
| `/pause <strategy>` | 특정 전략 시그널 생성 중지 | Yes (확인 버튼) |
| `/resume <strategy>` | 중지된 전략 재개 | Yes |
| `/reduce <symbol> <pct>` | 특정 심볼 포지션 축소 | Yes |
| `/report` | 즉시 일일 리포트 생성 | No |
| `/health` | 시스템 헬스 즉시 조회 | No |
| `/metrics` | 핵심 Prometheus 메트릭 요약 | No |

**구현:** `DiscordBotService`에 command 추가 + `TradingContext` 확장
**노력:** 중간 | **가치:** 높음

---

### Phase L4: Observability 통합 (Optional)

> 장기적 운용 안정성을 위한 고급 기능

#### L4-1. OTel Full Tracing

주문 lifecycle을 trace로 추적하여 병목 지점 파악.

**Order Lifecycle Span:**

```
[trade_cycle] ─── duration: 1.2s
  ├── [strategy.generate_signal]    42ms
  ├── [pm.process_signal]           15ms
  ├── [rm.pre_trade_check]          8ms
  ├── [oms.submit_order]            23ms
  └── [exchange.create_order]       1100ms  ← 병목!
```

**Backend:**

```
OTel SDK → OTel Collector → Grafana Tempo
Loguru → Loki
Prometheus → Grafana
                    ↓
         Grafana (단일 UI에서 3 pillars 통합)
```

**노력:** 중간 | **가치:** 장기적으로 높음

#### L4-2. Conformal-RANSAC Kill Switch

PageHinkley + GBM에 추가로, 구조적 전략 쇠퇴를 감지하는 robust한 kill switch.

**원리:**
1. **Slope Condition:** RANSAC으로 robust trend 추정 → 기울기 ≤ 0이면 양의 drift 소멸
2. **Level Condition:** Conformal prediction lower bound 아래면 비정상 drawdown

**기존 대비 이점:**
- 단일 outlier에 의한 왜곡 방지 (RANSAC high breakdown point)
- "False dawns" 거부 — 일시적 수익 급등이 구조적 쇠퇴를 마스킹하는 것 방지

**노력:** 높음 | **가치:** 높음 (장기)

#### L4-3. Distribution Drift Detection

백테스트 기간 return 분포 vs 최근 N일 return 분포를 KS test로 비교.

```python
from scipy.stats import ks_2samp

stat, p_value = ks_2samp(backtest_returns, recent_returns)
if p_value < 0.05:
    # 수익률 분포가 유의미하게 변화 → WARNING
```

**Alert:** p-value < 0.05 시 "Return distribution has shifted significantly" 알림
**노력:** 낮음 | **가치:** 중간

---

## 3. 구현 우선순위 요약

```
                   가치
                    ↑
         높음  │ L1-1  L1-2  L1-3  L1-4  L3-1  L3-3
               │ L2-3  L2-4  L4-2
               │
         중간  │ L2-1  L2-2  L3-2  L4-3
               │ L4-1
               │
         낮음  │
               └──────────────────────────────→ 노력
                   낮음       중간       높음
```

### 권장 구현 순서

| 순서 | Phase | 항목 | 노력 | 핵심 이유 |
|------|-------|------|------|----------|
| 1 | L1-1 | 봇 Lifecycle 알림 | 낮음 | 서버 on/off를 즉시 인지 |
| 2 | L1-4 | Data Freshness 알림 | 낮음 | 잘못된 시그널 방지 |
| 3 | L1-3 | Execution Quality 알림 | 낮음 | 실행 품질 저하 즉시 감지 |
| 4 | L1-2 | Per-Strategy Metrics + 알림 | 중간 | 멀티 전략 운용의 핵심 |
| 5 | L2-4 | Reconciliation 알림 강화 | 낮음 | 포지션 불일치 빠른 감지 |
| 6 | L2-1 | Event Loop & Process Metrics | 낮음 | 시스템 병목 감지 |
| 7 | L2-2 | WS 상세 Metrics | 낮음 | 데이터 안정성 확보 |
| 8 | L3-3 | Interactive Discord Commands | 중간 | 즉각적 대응 능력 |
| 9 | L3-1 | GBM Drawdown Monitor | 중간 | 전략 교체 시점 판단 |
| 10 | L2-3 | Grafana Dashboard as Code | 중간 | 시각적 모니터링 |
| 11 | L3-2 | Execution Anomaly Detection | 중간 | 실행 이상 패턴 감지 |
| 12 | L4-1 | OTel Full Tracing | 중간 | 3 pillars 통합 |
| 13 | L4-2 | Conformal-RANSAC Kill Switch | 높음 | 고급 전략 쇠퇴 감지 |
| 14 | L4-3 | Distribution Drift Detection | 낮음 | 수익률 분포 변화 감지 |

---

## 4. 아키텍처 원칙

### 4.1 Fire-and-Forget (기존 유지)

모니터링/알림은 **절대로** 트레이딩 로직을 블로킹하지 않는다.

```python
# Good: fire-and-forget
bus.publish(RiskAlertEvent(level="WARNING", message="..."))

# Bad: await in critical path
await discord.send_embed(...)  # ← 절대 금지
```

### 4.2 Alert Fatigue 방지

| 원칙 | 구현 |
|------|------|
| Severity 분리 | INFO(heartbeat채널), WARNING(alerts채널), CRITICAL(alerts채널 + @mention) |
| Cooldown | SpamGuard 300s (기존 유지) |
| Aggregation | 동일 이벤트 5건 이상 → 1건으로 요약 |
| Escalation | WARNING 30분 지속 → CRITICAL 승격 |

### 4.3 Graceful Degradation (기존 강화)

```
Discord 정상 → Embed 전송
Discord 5회 실패 → loguru CRITICAL 전환 (기존)
Discord 10회 실패 → Webhook fallback 시도 (신규)
Prometheus 장애 → metrics 수집 중단, 트레이딩 계속 (기존)
```

### 4.4 Metrics as Code

- Grafana dashboard는 `infra/grafana/dashboards/*.json`으로 버전 관리
- Prometheus alert rules는 `infra/prometheus/alerts.yml`로 버전 관리
- Docker Compose로 전체 스택 원클릭 배포

---

## 5. 기술 스택

| Component | Technology | 역할 |
|-----------|-----------|------|
| Metrics | `prometheus_client` (기존) | 메트릭 수집/노출 |
| Dashboard | Grafana (추가) | 시각화 |
| Alerting | Prometheus Alertmanager + Discord webhook | 임계값 기반 알림 |
| Tracing | OpenTelemetry SDK → Tempo (L4) | 주문 lifecycle 추적 |
| Logging | loguru → Loki (L4) | 로그 집중화 |
| Notifications | discord.py (기존) | 실시간 알림 + 명령 |

---

## 6. 디렉토리 구조 변경

```
src/monitoring/
├── __init__.py              # 기존
├── metrics.py               # 기존 — strategy label 추가 (L1-2)
├── chart_generator.py       # 기존
├── process_monitor.py       # 신규 — Event loop lag, memory, CPU (L2-1)
└── anomaly/                 # 신규 — 이상 감지 모듈
    ├── __init__.py
    ├── gbm_drawdown.py      # GBM drawdown monitor (L3-1)
    ├── execution_quality.py # Execution anomaly detector (L3-2)
    └── distribution.py      # KS test drift detection (L4-3)

src/notification/
├── ...                      # 기존 파일 유지
├── lifecycle.py             # 신규 — 봇 시작/종료/crash 알림 (L1-1)
└── strategy_report.py       # 신규 — 전략별 성과 리포트 (L1-2)

infra/                       # 신규 — 인프라 코드
├── docker-compose.yml       # Prometheus + Grafana + Alertmanager
├── grafana/
│   ├── dashboards/          # Dashboard JSON (L2-3)
│   └── provisioning/
└── prometheus/
    ├── prometheus.yml       # Scrape config
    └── alerts.yml           # Alert rules
```

---

## 7. 성공 기준

| Phase | 성공 기준 |
|-------|----------|
| L1 | 봇 on/off 시 10초 이내 Discord 알림 수신 |
| L1 | 전략별 PnL/drawdown Discord 리포트에 표시 |
| L1 | 슬리피지 > 15bps 시 자동 알림 |
| L1 | 데이터 지연 시 2 × TF 내 알림 |
| L2 | Grafana에서 전체 시스템 상태 한눈에 파악 가능 |
| L2 | Event loop lag > 1s 시 감지 |
| L3 | GBM 기반으로 drawdown이 비정상인지 정량 판단 |
| L3 | Discord에서 전략 pause/resume 가능 |
| L4 | Log → Trace → Metric 상호 참조 가능 |

---

## 참고 자료

### 2026 트렌드 & Best Practices

- [Prometheus Naming Conventions](https://prometheus.io/docs/practices/naming/)
- [Freqtrade Grafana Dashboard](https://github.com/thraizz/freqtrade-dashboard)
- [Trading Strategy Monitoring via GBM](https://portfoliooptimizer.io/blog/trading-strategy-monitoring-modeling-the-pnl-as-a-geometric-brownian-motion/)
- [Conformal-RANSAC Kill Switch](https://www.quantbeckman.com/p/with-code-switch-off-conformal-ransac)
- [OpenTelemetry asyncio Instrumentation](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/asyncio/asyncio.html)
- [prometheus-async for Python](https://prometheus-async.readthedocs.io/en/stable/asyncio.html)
- [Grafana Observability Stack](https://grafana.com/docs/opentelemetry/)
