# MC Coin Bot — Notification Reference

Discord 기반 실시간 알림 + 양방향 명령 시스템 레퍼런스.

## Architecture

```
Trading Events (EventBus)
    -> NotificationEngine (subscriber)
        -> Formatters (event -> embed dict)
            -> NotificationQueue (async, fire-and-forget)
                -> SpamGuard (dedup)
                -> Retry (exponential backoff)
                -> DiscordBotService.send_embed()
                    -> Discord Channels

Scheduled Tasks (asyncio loops)
    -> HealthCheckScheduler (1h / 4h / 8h)
    -> ReportScheduler (daily / weekly)
    -> OrchestratorNotificationEngine (daily 00:05 UTC)

Slash Commands (interactive)
    -> DiscordBotService (discord.py CommandTree)
        -> TradingContext (PM, RM, Analytics, Orchestrator)
```

### Key Files

| File | Description |
|------|-------------|
| `src/notification/models.py` | Severity, ChannelRoute, NotificationItem |
| `src/notification/config.py` | DiscordBotConfig (env 설정) |
| `src/notification/queue.py` | NotificationQueue + SpamGuard |
| `src/notification/engine.py` | NotificationEngine (EventBus subscriber) |
| `src/notification/bot.py` | DiscordBotService + Slash Commands |
| `src/notification/formatters.py` | 트레이딩 이벤트 embed 포매터 |
| `src/notification/lifecycle.py` | 봇 시작/종료/crash embed |
| `src/notification/health_models.py` | Health check 데이터 모델 |
| `src/notification/health_scheduler.py` | 주기적 헬스체크 스케줄러 |
| `src/notification/health_formatters.py` | 헬스체크 embed 포매터 |
| `src/notification/report_scheduler.py` | Daily/Weekly 리포트 스케줄러 |
| `src/notification/orchestrator_engine.py` | Orchestrator 이벤트 알림 |
| `src/notification/orchestrator_formatters.py` | Orchestrator embed 포매터 |
| `src/notification/reconciler_formatters.py` | 포지션 reconciliation embed |

---

## Channel Routing

| Channel | Env Variable | 용도 |
|---------|-------------|------|
| `TRADE_LOG` | `DISCORD_TRADE_LOG_CHANNEL_ID` | 체결, 잔고, 포지션 업데이트 |
| `ALERTS` | `DISCORD_ALERTS_CHANNEL_ID` | Circuit Breaker, Risk Alert, Lifecycle, Orchestrator, Safety Stop, Drift, On-chain |
| `DAILY_REPORT` | `DISCORD_DAILY_REPORT_CHANNEL_ID` | Daily/Weekly 리포트, Strategy Health (8h) |
| `HEARTBEAT` | `DISCORD_HEARTBEAT_CHANNEL_ID` | System Health (1h) |
| `MARKET_REGIME` | `DISCORD_REGIME_CHANNEL_ID` | Market Regime (4h) |

### Severity Levels

| Severity | 용도 | Routing |
|----------|------|---------|
| `INFO` | 일상적 이벤트 (체결, heartbeat) | 해당 채널에 전송 |
| `WARNING` | 주의 필요 (shutdown, degradation) | ALERTS 채널 |
| `CRITICAL` | 즉시 대응 (crash, circuit breaker) | ALERTS 채널 |
| `EMERGENCY` | 시스템 비상 | ALERTS 채널 |

---

## Event-Driven Notifications

NotificationEngine이 EventBus를 구독하여 자동 전송.

| Event | Formatter | Channel | Severity |
|-------|-----------|---------|----------|
| `FillEvent` + `PositionUpdateEvent` | `format_fill_with_position_embed()` | TRADE_LOG | INFO |
| `PositionUpdateEvent` (단독) | `format_position_embed()` | TRADE_LOG | INFO |
| `CircuitBreakerEvent` | `format_circuit_breaker_embed()` | ALERTS | CRITICAL |
| `RiskAlertEvent` | `format_risk_alert_embed()` | ALERTS | WARNING/CRITICAL |
| `BalanceUpdateEvent` | `format_balance_embed()` | TRADE_LOG | INFO |

> **Fill + Position 병합:** `FillEvent`는 `_pending_fills`에 버퍼링 → 다음 `PositionUpdateEvent`에서 병합 전송. Fill 없이 Position만 오면 `format_position_embed()`으로 단독 전송.

### Embed Color Palette

| Color | Hex | Decimal | Use Cases |
|-------|-----|---------|-----------|
| Green | #57F287 | 5763719 | BUY, startup, healthy, profit |
| Red | #ED4245 | 15548997 | SELL, crash, critical, loss |
| Blue | #3498DB | 3447003 | Info, reports, balance |
| Yellow | #FFFF00 | 16776960 | Warning, shutdown, decay |
| Orange | #E67E22 | 15105570 | Circuit breaker, critical drift |

---

## Lifecycle Notifications

봇 시작/종료/crash 시 ALERTS 채널에 자동 전송. `LiveRunner.run()`에 통합.

### Startup (GREEN)

```
MC Coin Bot Started
  Mode:       LIVE
  Strategy:   Orchestrator (2 pods)
  Timeframe:  1D
  Capital:    $100,000
  Symbols:    BTC/USDT, ETH/USDT, SOL/USDT, DOGE/USDT, BNB/USDT
  Pods:
    Pod              State       Alloc
    ──────────────────────────────────
    anchor-mom       production  50.0%
    ctrend           incubation  30.0%
```

### Graceful Shutdown (YELLOW)

```
MC Coin Bot Stopped
  Reason:         Graceful shutdown
  Uptime:         12h 34m
  Final Equity:   $10,150
  Today PnL:      $+500.00 (+5.00%)
  Realized:       $+300.00
  Unrealized:     $+200.00
  Open Positions: 2
  Pods:
    Pod              State       Alloc
    ──────────────────────────────────
    anchor-mom       production  50.0%
    ctrend           incubation  30.0%
```

### Crash (RED)

```
MC Coin Bot CRASHED
  Error Type:      ConnectionError
  Uptime:          2h 15m
  Error:           Connection to Binance lost after 3 retries...
  Final Equity:    $9,500       (조건부: PM 접근 가능 시)
  Open Positions:  2            (조건부)
  Unrealized PnL:  $-150.50     (조건부)
```

---

## Scheduled Reports

### Health Check Scheduler

3개의 독립 asyncio loop.

#### Heartbeat (1h) -> HEARTBEAT

시스템 상태 요약.

**Color Logic:**

- GREEN: DD < 5%, stale symbols = 0, CB off, queue normal, safety_stop_failures < 5
- YELLOW: DD 5-8%, stale > 0, queue depth > 50, on-chain sources degraded (ok < total)
- RED: DD > 8%, CB active, all stale, notification degraded, safety_stop_failures >= 5

**Fields:**

- Uptime, Equity, Drawdown, WS Status, Positions, Leverage
- Today PnL (trade count), Queue Depth, CB Status
- Safety Stops (active count + failures), On-chain (조건부: sources OK/total + columns)

#### Market Regime (4h) -> MARKET_REGIME

시장 체제 + 파생 데이터 요약.

**Fields:**

- Regime Label & Score (-1.0 ~ +1.0)
- Per-symbol: Price, Funding Rate (annualized), LS Ratio, Taker Ratio

#### Strategy Health (8h) -> DAILY_REPORT

전략별 성과 분석 + Alpha Decay 감지.

**Fields:**

- Rolling Sharpe (30d), Win Rate (recent 20), Profit Factor
- Trades Total, Open Positions, CB Status
- Alpha Decay (3 consecutive Sharpe declines × 2 confirmations = 16h window)
- Per-strategy breakdown: Sharpe, WR, PnL, trade count, status (HEALTHY/WATCH/DEGRADING)

### Report Scheduler

| Schedule | Trigger | Embed Fields | Charts |
|----------|---------|-------------|--------|
| Daily (00:00 UTC) | Auto | Today's Trades, Today's PnL, Total Equity, Max Drawdown, Open Positions, Sharpe Ratio | 4종 (아래 참조) |
| Weekly (Mon 00:00 UTC) | Auto | Weekly Trades, Weekly PnL, Sharpe Ratio, Max Drawdown, Best Trade, Worst Trade | 4종 (아래 참조) |
| Manual | `/report` command | 즉시 daily report 생성 | 4종 |

**Charts (PNG, matplotlib) — Daily & Weekly 공통:**

- Equity Curve (timeseries with fill)
- Drawdown (% with fill)
- Monthly Return Heatmap (RdYlGn colormap)
- PnL Distribution (win/loss histogram)

### Orchestrator Report

| Schedule | Content | Channel |
|----------|---------|---------|
| Daily (00:05 UTC) | Pod table (State/Alloc/Days), Total Equity, Effective N, Correlation, Drawdown, Gross Leverage, Active Pods | DAILY_REPORT |
| On Event | Lifecycle transition, Capital rebalance, Risk alerts | ALERTS / TRADE_LOG |

**Lifecycle Color:**

- Incubation -> BLUE
- Production -> GREEN
- Warning -> YELLOW
- Probation -> ORANGE
- Retired -> RED

---

## Position Reconciliation Alerts

`PositionReconciler` drift 감지 시 ALERTS 채널 전송.

> **Note:** 현재 `_setup_reconciler`에서 `PositionReconciler(auto_correct=False)`로 생성되므로,
> Action은 항상 `Manual review needed`으로 표시됩니다.

### Position Drift

```
Position Drift Detected (2 symbols)

  BTC/USDT
    PM: 0.050000 (LONG)
    Exchange: 0.048000 (LONG)
    Drift: 4.0% | Manual review needed

  ORPHAN ETH/USDT
    PM: 0.000000 (FLAT)
    Exchange: 0.100000 (LONG)
    Drift: 100.0% | Manual review needed
```

**Color:** ORANGE (drift < 10%), RED (orphan or drift >= 10%)

### Balance Drift

```
Balance Drift WARNING
  PM Equity:        $10,000
  Exchange Equity:  $10,350
  Drift:            3.5%
```

**Color:** YELLOW (2~5%), RED (>= 5%) — drift >= 2% 시 알림 트리거

---

## Safety Stop Alerts

`ExchangeStopManager`가 안전 정지 주문 실패/누락 감지 시 ALERTS 채널 전송.

### Safety Stop Failure (RED)

연속 5회 이상 배치 실패 시 CRITICAL 알림.

```
SAFETY STOP FAILURE
  Symbol:    BTC/USDT
  Failures:  5
  Exchange safety net may be INACTIVE — manual intervention required.
```

### Safety Stop Stale (ORANGE)

재시작 후 거래소에 안전 정지 주문이 없을 때 WARNING 알림.

```
SAFETY STOP STALE
  Symbol:    BTC/USDT
  Restored from state but not found on exchange. Will be re-placed on next bar.
```

---

## On-chain Data Alerts

`LiveOnchainFeed` 캐시 갱신 실패 시 ALERTS 채널 전송.

```
On-chain Alert — LiveOnchainFeed
  Cache refresh failed
```

**Color:** YELLOW, **Severity:** WARNING, **Spam Key:** `onchain_refresh_fail` (300s 쿨다운)

---

## Discord Slash Commands

### Read-Only

| Command | Description | Response |
|---------|-------------|----------|
| `/status` | Open positions + equity + drawdown + CB | Embed |
| `/balance` | Account balance + leverage + margin | Embed |
| `/health` | System health (triggers manual heartbeat) | Embed |
| `/metrics` | Prometheus metrics summary | Embed |
| `/strategies` | Orchestrator pod overview (state/sharpe/DD/WR) | Embed |
| `/report` | Trigger daily report immediately | Charts + Embed |
| `/strategy <name>` | Individual pod details (state/capital/performance/GBM) | Embed |
| `/onchain` | On-chain 데이터 소스 상태 (cache + per-source last fetch) | Embed |

### Action Commands

| Command | Description | Confirmation |
|---------|-------------|-------------|
| `/kill` | Emergency shutdown + 시스템 비활성화 | **없음** (즉시 실행) |
| `/pause <name>` | Pod 시그널 생성 중지 | Button (30s timeout) |
| `/resume <name>` | 중지된 pod 재개 | Button (30s timeout) |

> **주의:** `/kill`은 확인 없이 즉시 실행됩니다. `/pause`, `/resume`만 `_ConfirmView` (Confirm/Cancel 버튼, 30초 timeout) 사용.

---

## Queue System

### NotificationQueue

Fire-and-forget 비동기 큐. 트레이딩 로직을 절대 블로킹하지 않음.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `queue_size` | 500 | asyncio.Queue maxsize |
| `max_retries` | 3 | 재시도 횟수 |
| `base_backoff` | 1.0s | 지수 백오프 기본 (1s, 2s, 4s) |
| `cooldown_seconds` | 300s | SpamGuard 기본 쿨다운 |

### SpamGuard

동일 이벤트 반복 전송 방지.

| Spam Key | Source | 용도 |
|----------|--------|------|
| `balance_update` | engine.py | 잔고 업데이트 |
| `risk_alert:{level}` | engine.py | Risk alert (레벨별) |
| `heartbeat` | health_scheduler.py | System heartbeat |
| `regime_report` | health_scheduler.py | Market regime |
| `strategy_health` | health_scheduler.py | Strategy health |
| `orchestrator_rebalance` | orchestrator_engine.py | Capital rebalance |
| `orch_risk:{type}` | orchestrator_engine.py | Orchestrator risk (유형별) |
| `startup_drift` | live_runner.py | 시작 시 포지션 drift |
| `position_drift` | live_runner.py | 주기적 포지션 drift |
| `balance_drift` | live_runner.py | 잔고 drift |
| `onchain_refresh_fail` | onchain_feed.py | On-chain 갱신 실패 |

모든 spam key의 기본 쿨다운은 300초 (5분).

### Graceful Degradation

```
Discord 정상        -> Embed 전송
5회 연속 실패       -> Degraded mode (loguru CRITICAL fallback)
1회 성공            -> 자동 복구
Queue full          -> put_nowait() drop (loguru WARNING)
```

---

## Configuration

### Environment Variables

```bash
# Bot Mode (Primary)
DISCORD_BOT_TOKEN=<bot_token>
DISCORD_GUILD_ID=<guild_id>
DISCORD_TRADE_LOG_CHANNEL_ID=<channel_id>
DISCORD_ALERTS_CHANNEL_ID=<channel_id>
DISCORD_DAILY_REPORT_CHANNEL_ID=<channel_id>
DISCORD_HEARTBEAT_CHANNEL_ID=<channel_id>
DISCORD_REGIME_CHANNEL_ID=<channel_id>

# Webhook Mode (Legacy fallback)
DISCORD_TRADE_WEBHOOK_URL=<webhook_url>
DISCORD_ERROR_WEBHOOK_URL=<webhook_url>
DISCORD_REPORT_WEBHOOK_URL=<webhook_url>
```

### DiscordBotConfig

`src/notification/config.py` — Pydantic `BaseSettings`, env prefix `DISCORD_`.

```python
config = DiscordBotConfig()     # .env 자동 로드
config.is_bot_configured        # True if bot_token + guild_id set
```

---

## Integration with LiveRunner

`src/eda/live_runner.py` 에서 전체 알림 시스템 초기화/종료.

### Startup Flow

```python
# 1. Discord 셋업
discord_tasks = await self._setup_discord(bus, pm, rm, analytics)
    # -> DiscordBotService + NotificationQueue
    # -> NotificationEngine.register(bus)
    # -> TradingContext + set on bot
    # -> ReportScheduler.start()
    # -> HealthCheckScheduler.start()
    # -> OrchestratorNotificationEngine (if orchestrator)

# 2. Lifecycle 알림
await self._send_lifecycle_startup(discord_tasks, capital)

# 3. 트레이딩 루프 (이벤트 자동 전송)
...

# 4. 정상 종료
await self._send_lifecycle_shutdown(discord_tasks, pm, analytics)

# 5. (또는) Crash
await self._send_lifecycle_crash(discord_tasks, exc)
await asyncio.sleep(2)  # drain 보장

# 6. 셧다운
await self._shutdown_discord(discord_tasks)
    # -> health/report scheduler stop
    # -> queue drain + stop
    # -> bot close
```

### _DiscordTasks

```python
@dataclass
class _DiscordTasks:
    bot_service: DiscordBotService
    notification_queue: NotificationQueue
    queue_task: asyncio.Task[None]
    bot_task: asyncio.Task[None]
    report_scheduler: ReportScheduler | None
    health_scheduler: HealthCheckScheduler | None
```

---

## Design Principles

### 1. Fire-and-Forget

모니터링/알림은 트레이딩 로직을 블로킹하지 않음.

```python
# Good: fire-and-forget
bus.publish(RiskAlertEvent(level="WARNING", message="..."))

# Bad: await in critical path
await discord.send_embed(...)  # 절대 금지
```

### 2. Alert Fatigue 방지

| 원칙 | 구현 |
|------|------|
| Severity 분리 | INFO(heartbeat), WARNING(alerts), CRITICAL(alerts + 강조) |
| Cooldown | SpamGuard 300s 기본 |
| Channel 분리 | 5개 독립 채널로 노이즈 격리 |

### 3. Graceful Degradation

Discord 장애 시 트레이딩은 계속, 알림은 loguru로 fallback.

### 4. Executor for Blocking

차트 생성(matplotlib)은 `loop.run_in_executor()`로 event loop 보호.
