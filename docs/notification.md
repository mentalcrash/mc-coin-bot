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
| `ALERTS` | `DISCORD_ALERTS_CHANNEL_ID` | Circuit Breaker, Risk Alert, Lifecycle, Orchestrator |
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
| `FillEvent` | `format_fill_embed()` | TRADE_LOG | INFO |
| `CircuitBreakerEvent` | `format_circuit_breaker_embed()` | ALERTS | CRITICAL |
| `RiskAlertEvent` | `format_risk_alert_embed()` | ALERTS | WARNING/CRITICAL |
| `BalanceUpdateEvent` | `format_balance_embed()` | TRADE_LOG | INFO |
| `PositionUpdateEvent` | `format_position_embed()` | TRADE_LOG | INFO |

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
  Strategy:   anchor-mom
  Timeframe:  12h
  Capital:    $10,000.00
  Symbols:    DOGE/USDT, SOL/USDT, BTC/USDT, ETH/USDT, BNB/USDT
```

### Graceful Shutdown (YELLOW)

```
MC Coin Bot Stopped
  Reason:         Graceful shutdown
  Uptime:         12h 34m
  Final Equity:   $10,150.00
  Today PnL:      +$150.00 (+1.50%)
  Open Positions: 0
```

### Crash (RED)

```
MC Coin Bot CRASHED
  Error Type:  ConnectionError
  Uptime:      2h 15m
  Error:       Connection to Binance lost after 3 retries...
```

---

## Scheduled Reports

### Health Check Scheduler

3개의 독립 asyncio loop.

#### Heartbeat (1h) -> HEARTBEAT

시스템 상태 요약.

**Color Logic:**
- GREEN: DD < 5%, stale symbols = 0, CB off, queue normal
- YELLOW: DD 5-8%, stale > 0, queue depth > 50
- RED: DD > 8%, CB active, all stale, notification degraded

**Fields:**
- Uptime, Equity, Cash, Leverage, Positions
- Drawdown, Regime, WS Health
- Queue Depth, Events Dropped

#### Market Regime (4h) -> MARKET_REGIME

시장 체제 + 파생 데이터 요약.

**Fields:**
- Regime Label & Score (-1.0 ~ +1.0)
- Per-symbol: Funding Rate, LS Ratio, Taker Ratio

#### Strategy Health (8h) -> DAILY_REPORT

전략별 성과 분석 + Alpha Decay 감지.

**Fields:**
- Rolling Sharpe (30d), Win Rate (recent 20), Profit Factor
- Open Positions, Alpha Decay (3 consecutive Sharpe declines)
- Per-strategy breakdown: PnL, trade count, status (HEALTHY/WATCH/DEGRADING)

### Report Scheduler

| Schedule | Trigger | Content |
|----------|---------|---------|
| Daily (00:00 UTC) | Auto | Equity curve + drawdown + monthly heatmap + PnL distribution |
| Weekly (Mon 00:00 UTC) | Auto | 동일 (주간 집계) |
| Manual | `/report` command | 즉시 daily report 생성 |

**Charts (PNG, matplotlib):**
- Equity Curve (timeseries with fill)
- Drawdown (% with fill)
- Monthly Return Heatmap (RdYlGn colormap)
- PnL Distribution (win/loss histogram)

### Orchestrator Report

| Schedule | Content | Channel |
|----------|---------|---------|
| Daily (00:05 UTC) | Pod summaries, Total equity, Diversification, Correlation | DAILY_REPORT |
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

### Position Drift

```
Position Drift Detected
  BTC/USDT
    Expected: 0.050 BTC (LONG)
    Actual:   0.048 BTC (LONG)
    Drift:    4.0%
    Action:   Auto-corrected

  ETH/USDT
    Expected: 0.00 (FLAT)
    Actual:   0.10 ETH (LONG)
    Drift:    ORPHAN POSITION
    Action:   Manual review needed
```

**Color:** ORANGE (drift < 10%), RED (orphan or drift >= 10%)

### Balance Drift

**Color:** YELLOW (2-5%), RED (>= 5%)

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

### Action Commands (Confirmation Required)

| Command | Description | Confirmation |
|---------|-------------|-------------|
| `/kill` | Emergency shutdown + 전 포지션 청산 | Button (30s timeout) |
| `/pause <name>` | Pod 시그널 생성 중지 | Button (30s timeout) |
| `/resume <name>` | 중지된 pod 재개 | Button (30s timeout) |

**Confirmation UI:** `_ConfirmView` (discord.ui.View) - Confirm/Cancel 버튼, 30초 timeout.

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

| Spam Key | 용도 |
|----------|------|
| `balance_update` | 잔고 업데이트 (5분 쿨다운) |
| `risk_alert:{level}` | Risk alert (레벨별 쿨다운) |
| `heartbeat` | Heartbeat (5분 쿨다운) |
| `orchestrator_rebalance` | Rebalance 알림 |

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
