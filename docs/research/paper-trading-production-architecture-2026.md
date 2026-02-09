# Paper Trading & Production Architecture Guide (2026)

> MC Coin Bot의 EDA 백테스트 → 공유 인프라 구축 → Paper Trading → Live 배포 전환을 위한 종합 리서치 문서
>
> 작성일: 2026-02-09 (v2: Shadow/Paper 분석 추가) | Python 3.13 + asyncio + Pydantic V2 + CCXT Pro 기준

---

## Table of Contents

1. [현재 아키텍처 분석](#1-현재-아키텍처-분석)
2. [데이터 저장소 설계](#2-데이터-저장소-설계)
3. [알림 시스템 설계](#3-알림-시스템-설계)
4. [Production 배포 아키텍처](#4-production-배포-아키텍처)
5. [Shadow vs Paper 분석 & 전환 파이프라인](#5-shadow-vs-paper-분석--전환-파이프라인) **(v2 신규)**
6. [종합 구현 로드맵](#6-종합-구현-로드맵) **(v2 재구성)**

---

## 1. 현재 아키텍처 분석

### 1.1 이미 구축된 것 (Strengths)

| 컴포넌트 | 상태 | 핵심 파일 |
|----------|------|-----------|
| **EventBus** | ✅ 완료 | `src/core/event_bus.py` — async Queue, JSONL audit, backpressure |
| **11 Event Types** | ✅ 완료 | `src/core/events.py` — BAR, SIGNAL, FILL, ORDER_*, POSITION/BALANCE_UPDATE 등 |
| **ExecutorPort Protocol** | ✅ 완료 | `src/eda/ports.py` — `execute(order) -> FillEvent | None` |
| **BacktestExecutor** | ✅ 완료 | `src/eda/executors.py` — next-open fill + CostModel |
| **ShadowExecutor** | ✅ 완료 | `src/eda/executors.py` — signal logging only |
| **PM/RM/OMS** | ✅ 완료 | Position tracking, SL/TS, risk validation, idempotent orders |
| **AnalyticsEngine** | ✅ 완료 | Equity curve, trade records, PerformanceMetrics |
| **EDARunner** | ✅ 완료 | `.backtest()` / `.shadow()` factory methods |
| **CandleAggregator** | ✅ 완료 | 1m → target TF 집계, UTC boundary alignment |
| **Discord Notifier** | ✅ 기본 | `src/notification/discord.py` — Embed 기반 webhook |
| **BinanceClient** | ✅ 완료 | `src/exchange/binance_client.py` — CCXT Pro async wrapper |
| **23 Strategies** | ✅ 완료 | Tier 1~3, 1623 tests |

### 1.2 구현 필요 사항 (Gaps)

> Phase 번호는 Section 6의 구현 로드맵과 대응합니다.

| 컴포넌트 | 상태 | Phase | 비고 |
|----------|------|-------|------|
| **LiveDataFeed** (WebSocket 실시간) | ❌ 미구현 | 6-A | Shadow/Paper/Live 모두 필요 |
| **LiveRunner** (Graceful shutdown) | ❌ 미구현 | 6-A | 24/7 프로세스 관리 |
| **Trade 데이터 영속화** (SQLite) | ❌ 미구현 | 6-B | 거래 기록, equity curve |
| **State Recovery** (재시작 복구) | ❌ 미구현 | 6-B | VPS 재시작 시 포지션 유지 |
| **Dockerfile / docker-compose** | ❌ 미구현 | 6-C | VPS 배포 필수 |
| **환경별 설정** (paper/live yaml) | ❌ 미구현 | 6-C | 모드 전환 용이 |
| **NotificationEngine** (EventBus 연동) | ❌ 미구현 | 6-D | 알림 기반 |
| **Telegram 알림** (aiogram 3.x) | ❌ 미구현 | 6-D | CRITICAL alert |
| **Chart 생성/첨부** | ❌ 미구현 | 7.5 | Paper 운영 중 추가 |
| **ReportScheduler** (주기적 리포트) | ❌ 미구현 | 7.5 | Paper 운영 중 추가 |
| **Prometheus Metrics** | ❌ 미구현 | 7.5 | Paper 운영 중 추가 |
| **Telegram 양방향** (/status, /kill) | ❌ 미구현 | 7.5 | Paper 운영 중 추가 |
| **LiveExecutor** (실제 주문 제출) | ❌ 미구현 | 8 | Live 전환 시 구현 |
| **Reconciliation** (거래소 vs 로컬) | ❌ 미구현 | 8 | Live 전환 시 필수 |
| **Kill Switch** (긴급 정지) | ❌ 미구현 | 8 | Live 전환 시 필수 |
| **CI/CD Pipeline** | ❌ 미구현 | 7.5 | Paper 안정화 후 |

### 1.3 ExecutionMode (이미 정의됨)

`src/models/eda.py`에 5단계가 이미 설계되어 있음:

```python
class ExecutionMode(StrEnum):
    BACKTEST = "backtest"  # ✅ 구현 완료 (HistoricalDataFeed + BacktestExecutor)
    SHADOW   = "shadow"    # ⚠️ Executor만 구현 (ShadowExecutor), LiveDataFeed 미구현
    PAPER    = "paper"     # ❌ LiveDataFeed + BacktestExecutor (시뮬레이션 체결)
    CANARY   = "canary"    # ❌ LiveDataFeed + LiveExecutor (소액 실제 주문)
    LIVE     = "live"      # ❌ LiveDataFeed + LiveExecutor (전액 실제 주문)
```

> **핵심 인사이트**: Shadow/Paper/Canary/Live 모두 **동일한 공유 인프라**가 필요합니다.
> LiveDataFeed, DB, 알림, 배포 환경이 갖춰져야 어떤 모드든 실행 가능.
> 차이는 **Executor 한 줄 교체**뿐입니다.

### 1.4 이벤트 흐름 (현재 vs 목표)

```
[Backtest Mode — ✅ 완료]
HistoricalDataFeed → BarEvent(1m) → CandleAggregator → BarEvent(target_tf)
  → StrategyEngine → SignalEvent
  → PM → OrderRequestEvent → RM → OMS → BacktestExecutor → FillEvent
  → PM → PositionUpdateEvent + BalanceUpdateEvent
  → AnalyticsEngine → PerformanceMetrics

[Shadow/Paper/Live Mode — 공유 인프라 필요]
LiveDataFeed (WebSocket) → BarEvent(1m) → CandleAggregator → BarEvent(target_tf)
  → StrategyEngine → SignalEvent
  → PM → OrderRequestEvent → RM → OMS → [Executor 교체] → FillEvent
  → PM → PositionUpdateEvent + BalanceUpdateEvent
  → AnalyticsEngine → PerformanceMetrics
  → NotificationEngine → Discord/Telegram      ← 신규 (공유)
  → PersistenceEngine → SQLite/JSONL           ← 신규 (공유)
  → StateManager → 재시작 복구                  ← 신규 (공유)
```

**Executor 교체 매트릭스:**

| Mode | DataFeed | Executor | 체결 | 실제 자금 |
|------|----------|----------|------|----------|
| **Shadow** | LiveDataFeed | ShadowExecutor (✅있음) | ❌ 없음 (로깅만) | ❌ |
| **Paper** | LiveDataFeed | BacktestExecutor (✅있음) | ✅ 시뮬레이션 | ❌ |
| **Canary** | LiveDataFeed | LiveExecutor (❌미구현) | ✅ 실제 체결 | ✅ 소액 |
| **Live** | LiveDataFeed | LiveExecutor (❌미구현) | ✅ 실제 체결 | ✅ 전액 |

핵심: **Strategy, PM, RM 코드는 모든 모드에서 동일**. DataFeed와 Executor만 교체하면 됨 (EDA의 강점).
공유 인프라(LiveDataFeed, DB, 알림, 배포)를 한 번 구축하면 모든 모드를 즉시 전환 가능.

---

## 2. 데이터 저장소 설계

### 2.1 Database 비교 (MC Coin Bot 규모: 10-50 trades/day, 5-10 assets)

| DB | 유형 | 장점 | 단점 | 적합도 |
|---|---|---|---|---|
| **SQLite** | OLTP embedded | 설정 제로, 단일 파일, Python 내장, `aiosqlite` async | 동시 쓰기 제한, 분석 쿼리 느림 | **최적** (trade journal) |
| **DuckDB** | OLAP embedded | Parquet 직접 쿼리, 10-100x 집계 성능, 메모리 분석 | 동시 쓰기 불가, 서버 모드 없음 | **최적** (분석/리서치) |
| **PostgreSQL** | 범용 RDBMS | 에코시스템 최강, 풍부한 확장 | 운영 부담, 단일 봇에 과함 | 과잉 |
| **TimescaleDB** | PG 확장 | PG 호환, 자동 파티셔닝, 연속 집계 | PG 운영 필요, TSL 라이선스 | 확장 시 고려 |
| **QuestDB** | 시계열 전용 | 최고 ingestion 속도 (4M rows/sec) | 단일 노드, JVM 기반, 생태계 작음 | 틱 데이터 시 |
| **InfluxDB 3.0** | 시계열 전용 | Rust 재작성, Parquet 저장 | 2026년 아직 초기, 금융 레퍼런스 부족 | 비추천 |
| **ClickHouse** | OLAP 분산 | 대규모 분석 최강 | 소규모에 완전 overkill | 비추천 |

#### 2026년 트렌드

- **DuckDB 급성장**: 월 600만+ PyPI 다운로드. "From SQLite to DuckDB: Embedded Analytics Is Here"가 2026년 트렌드를 대변. Parquet + SQLite를 단일 SQL로 JOIN 가능.
- **TimescaleDB**: 하이브리드 row-columnar 엔진 (최근 데이터 row, 오래된 데이터 columnar 자동 전환).
- **InfluxDB 3.0**: 2025년 4월 Rust 재작성 출시. Apache Parquet + Arrow + DataFusion 기반. 아직 초기.
- **QuestDB**: TimescaleDB 대비 ingestion 6-13x, 쿼리 16-20x 빠름. 단일 노드 한정.

### 2.2 권장 아키텍처: 3-Tier Hybrid

```
[Data Architecture]

┌─────────────────────────────────────────────────────────────┐
│                     HOT (In-Memory)                         │
│  Pydantic models (현재 유지)                                 │
│  - PM._positions: dict[symbol, Position]                    │
│  - PM._total_equity, PM._available_cash                     │
│  - OMS._processed_orders: set[str]                          │
│  - RM._peak_equity, RM._current_drawdown                   │
│  접근 패턴: 매 bar마다 실시간 읽기/쓰기                         │
└─────────────────────────────────────────────────────────────┘
                          │ (FillEvent, BalanceUpdateEvent)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   WARM (SQLite via aiosqlite)                │
│  구조화된 쿼리가 필요한 운영 데이터                              │
│  - trades: 체결 내역 (fill_price, fee, pnl, strategy)       │
│  - equity_snapshots: 시간별 자산 곡선                         │
│  - positions_history: 포지션 변경 이력                        │
│  - risk_events: 리스크 알림, 서킷 브레이커 이력                 │
│  - bot_state: 재시작 복구용 스냅샷                             │
│  접근 패턴: 10-50 writes/day, daily/weekly 쿼리              │
│  보관: 1년, 이후 Parquet로 아카이브                            │
│  의존성: uv add aiosqlite                                    │
│  백업: 파일 복사 (cp trading.db backups/)                     │
└─────────────────────────────────────────────────────────────┘
                          │ (분석 시)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   COLD (Parquet + JSONL)                     │
│  장기 보관, 분석 전용                                         │
│  - OHLCV: data/bronze/, data/silver/ (현재 유지)             │
│  - Event audit: JSONL (EventBus event_log_path, 현재 유지)  │
│  - Backtest results: Parquet                                │
│  접근 패턴: 리서치/백테스트 시에만                              │
│  도구: DuckDB로 Parquet + SQLite를 단일 SQL로 JOIN           │
│  의존성: uv add duckdb (분석 전용)                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 저장 대상 데이터 상세

#### Event 저장 정책

| 이벤트 | SQLite | JSONL (현재) | 이유 |
|--------|--------|-------------|------|
| FILL | ✅ 구조화 저장 | ✅ audit | 거래 기록, PnL 분석 |
| ORDER_REQUEST | ✅ | ✅ audit | 주문 이력 추적 |
| ORDER_ACK / ORDER_REJECTED | ✅ | ✅ audit | 체결률 분석, 거부 원인 |
| POSITION_UPDATE | ✅ | ✅ audit | 포지션 이력 |
| BALANCE_UPDATE | ✅ (hourly) | ✅ audit | Equity curve |
| SIGNAL | 선택적 | ✅ audit | 전략 분석 시에만 |
| CIRCUIT_BREAKER, RISK_ALERT | ✅ | ✅ audit | 리스크 이벤트 사후 분석 |
| BAR | ❌ (Parquet에 이미 존재) | 선택적 | 중복 저장 방지 |
| HEARTBEAT | ❌ | ❌ | 운영 메트릭, 별도 Prometheus |

#### SQLite 스키마 설계

```sql
-- trade lifecycle: Signal → Order → Fill
-- correlation_id로 이벤트 체인 연결

CREATE TABLE trades (
    client_order_id  TEXT PRIMARY KEY,
    correlation_id   TEXT,
    symbol           TEXT NOT NULL,
    strategy_name    TEXT NOT NULL,
    direction        TEXT NOT NULL,      -- LONG/SHORT
    signal_strength  REAL,
    signal_ts        TEXT,               -- ISO 8601
    order_side       TEXT,               -- BUY/SELL
    target_weight    REAL,
    notional_usd     REAL,
    fill_price       REAL,
    fill_qty         REAL,
    fee              REAL DEFAULT 0.0,
    fill_ts          TEXT,
    realized_pnl     REAL,
    status           TEXT NOT NULL,      -- FILLED/REJECTED/CANCELLED
    created_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE equity_snapshots (
    timestamp        TEXT PRIMARY KEY,   -- ISO 8601
    total_equity     REAL NOT NULL,
    available_cash   REAL NOT NULL,
    unrealized_pnl   REAL DEFAULT 0.0,
    drawdown_pct     REAL DEFAULT 0.0,
    peak_equity      REAL
);

CREATE TABLE positions_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    direction        TEXT NOT NULL,
    size             REAL NOT NULL,
    avg_entry_price  REAL NOT NULL,
    current_price    REAL,
    unrealized_pnl   REAL DEFAULT 0.0,
    realized_pnl     REAL DEFAULT 0.0
);

CREATE TABLE risk_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    event_type       TEXT NOT NULL,      -- RISK_ALERT/CIRCUIT_BREAKER
    alert_level      TEXT,               -- WARNING/CRITICAL
    message          TEXT NOT NULL,
    metadata         TEXT                -- JSON blob
);

CREATE TABLE bot_state (
    key              TEXT PRIMARY KEY,
    value            TEXT NOT NULL,      -- JSON serialized
    updated_at       TEXT DEFAULT (datetime('now'))
);

-- Indexes
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_strategy ON trades(strategy_name);
CREATE INDEX idx_trades_fill_ts ON trades(fill_ts);
CREATE INDEX idx_equity_ts ON equity_snapshots(timestamp);
```

### 2.4 Pydantic V2 + aiosqlite 통합 패턴

기존 Event 모델의 `model_dump()` / `model_validate()`를 활용하여 ORM 없이 직접 매핑:

```python
import aiosqlite
from src.core.events import FillEvent

class TradePersistence:
    """거래 기록 영속화 (aiosqlite)."""

    def __init__(self, db_path: str = "data/trading.db") -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        """테이블 생성 (멱등)."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(SCHEMA_SQL)

    async def save_fill(self, fill: FillEvent, strategy: str, pnl: float | None) -> None:
        """FillEvent → trades 테이블 저장."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO trades
                   (client_order_id, correlation_id, symbol, strategy_name,
                    direction, order_side, fill_price, fill_qty, fee,
                    fill_ts, realized_pnl, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fill.client_order_id,
                    str(fill.correlation_id) if fill.correlation_id else None,
                    fill.symbol,
                    strategy,
                    "LONG" if fill.side == "BUY" else "SHORT",
                    fill.side,
                    fill.fill_price,
                    fill.fill_qty,
                    fill.fee,
                    fill.fill_timestamp.isoformat(),
                    pnl,
                    "FILLED",
                ),
            )
            await db.commit()

    async def get_trades(self, symbol: str | None = None, days: int = 30) -> list[dict]:
        """최근 N일간 거래 조회."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            query = "SELECT * FROM trades WHERE fill_ts > datetime('now', ?)"
            params: list = [f"-{days} days"]
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            query += " ORDER BY fill_ts DESC"
            cursor = await db.execute(query, params)
            return [dict(row) for row in await cursor.fetchall()]
```

### 2.5 DuckDB 분석 쿼리 예시

```python
import duckdb

# Parquet (OHLCV) + SQLite (trades) 크로스 JOIN
con = duckdb.connect()

# "SOL/USDT의 최근 30일 승률과 시장 대비 수익률은?"
result = con.sql("""
    WITH trade_pnl AS (
        SELECT symbol, realized_pnl, fill_ts
        FROM sqlite_scan('data/trading.db', 'trades')
        WHERE symbol = 'SOL/USDT'
          AND fill_ts > current_date - INTERVAL 30 DAY
    ),
    market AS (
        SELECT timestamp, close
        FROM read_parquet('data/silver/SOL_USDT/2026.parquet')
        WHERE timestamp > current_date - INTERVAL 30 DAY
    )
    SELECT
        COUNT(*) AS total_trades,
        SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
        ROUND(100.0 * SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS win_rate,
        ROUND(SUM(realized_pnl), 2) AS total_pnl,
        ROUND((LAST(market.close) / FIRST(market.close) - 1) * 100, 2) AS market_return_pct
    FROM trade_pnl
    CROSS JOIN market
""")
```

### 2.6 피해야 할 것

- **ClickHouse / Kafka / Redis Streams**: 10-50 trades/day 단일 프로세스 봇에 overkill
- **InfluxDB 3.0**: 아직 초기, 금융 레퍼런스 부족
- **무거운 ORM** (SQLAlchemy full, Django ORM): `aiosqlite` + `model_dump()`로 충분
- **PostgreSQL 단독 사용**: Docker 운영 부담 대비 SQLite로 충분한 규모

---

## 3. 알림 시스템 설계

### 3.1 Discord vs Telegram 비교

| 항목 | Discord Webhook | Telegram Bot API |
|------|----------------|-----------------|
| **설정** | URL 하나로 동작 | BotFather → 토큰 |
| **양방향** | ❌ (webhook은 단방향) | ✅ (명령어 수신: `/status`, `/kill`) |
| **Rate Limit** | 5 req / 2초 per webhook | 30 msg/sec (group), 개인 제한 없음 |
| **Rich Format** | Embed (25 fields, 6000자) | Markdown/HTML + InlineKeyboard |
| **이미지** | multipart/form-data | `sendPhoto` (BufferedInputFile) |
| **모바일 푸시** | 양호 (멘션 시 강함) | **매우 우수** (즉시 도착) |
| **양방향 명령** | Bot 별도 구현 필요 | 기본 지원 (`/status`, `/kill`) |

#### 2026 커뮤니티 선호

- **Telegram 압도적 우위**: Freqtrade (가장 인기 OSS crypto bot)는 Telegram 전용 RPC. QuantConnect도 Telegram webhook 공식 지원.
- **Discord는 보완적**: 커뮤니티 관리/토론에 강점, signal 자동화는 Telegram 대비 부족.
- **동시 사용 권장**: Severity 기반 라우팅으로 두 플랫폼 병행.

### 3.2 권장: Dual-Channel 아키텍처

```
[NotificationEngine] ← EventBus subscriber (FillEvent, CircuitBreakerEvent, BalanceUpdateEvent)
        │
        ▼
[NotificationQueue] ← asyncio.Queue, retry/backoff, SpamGuard
        │
        ▼
[NotificationRouter] ← Severity 기반 라우팅
        │
        ├── [Discord Webhook] (aiohttp, 현재 코드 확장)
        │     ├── #trade-log     ← INFO: 모든 거래 기록
        │     ├── #errors        ← WARNING+: 에러/경고
        │     └── #daily-report  ← 일일/주간 리포트
        │
        └── [Telegram Bot] (aiogram 3.x, 신규)
              ├── Alert Chat   ← CRITICAL: 서킷 브레이커, 긴급 알림
              ├── Trade Chat   ← INFO: 실시간 포지션 변경
              └── 양방향 명령   ← /status, /kill, /report
```

#### Severity 라우팅 규칙

| Severity | Discord | Telegram | 예시 |
|----------|---------|----------|------|
| **INFO** | ✅ trade-log | 선택적 | Trade entry/exit, signal 생성 |
| **WARNING** | ✅ errors | ✅ alert | Drawdown 경고, 연결 불안정 |
| **CRITICAL** | ✅ errors | ✅ alert (즉시) | Circuit breaker, 시스템 정지 |
| **EMERGENCY** | ✅ errors | ✅ alert + 반복 | 자금 이상, 보안 위협 |

### 3.3 알림 종류 상세

#### Trade Execution Alert (INFO)

```
--- ENTRY ---
BTC/USDT | LONG | TSMOM
Entry:     $95,230.50
Size:      0.052 BTC ($4,952)
Leverage:  1.0x
Stop-Loss: $91,421 (ATR 3.0x)
Signal:    +0.85

--- EXIT ---
BTC/USDT | CLOSE LONG | TSMOM
Exit:    $98,100.00
PnL:     +$149.22 (+3.01%)
Hold:    3d 14h
Fee:     $3.96
```

#### Daily Summary (INFO, 매일 00:00 UTC)

```
========== Daily Report: 2026-02-09 ==========
NAV:     $10,523.40 (+$124.30 / +1.20%)
Trades:  5 (W:3 / L:2 / WR: 60.0%)

[Top/Bottom]
 Best:  SOL/USDT  +$82.30 (+4.1%)
 Worst: DOGE/USDT -$23.10 (-2.3%)

[Open Positions]
 BTC/USDT  LONG  0.05 @ $95,200  (uPnL: +$140)
 ETH/USDT  LONG  1.20 @ $3,410   (uPnL: -$22)

[Risk]
 DD Today: -0.8%  |  System DD: -3.2% / -15.0%

[Rolling 30d]
 Sharpe: 1.42  |  Sortino: 2.10  |  Win Rate: 58%
==============================================
```

첨부: Equity curve + Drawdown chart (PNG 이미지)

#### Weekly Digest (INFO, 매주 월요일)

포함 항목:
1. 주간 수익률 (절대값 + %)
2. vs BTC / vs ETH benchmark 비교
3. 전략별 P&L breakdown table
4. Drawdown 상태
5. 거래 통계 (건수, 평균 hold time)
6. Portfolio allocation 변화

첨부 차트 2개:
- 주간 equity curve vs BTC benchmark (line chart)
- 전략별 수익 기여도 (stacked bar chart)

#### Monthly Performance Report (INFO, 매월 1일)

포함 항목:
1. 월간 수익률 vs benchmark
2. 전략별 상세 성과 (Sharpe, CAGR, MDD, Win Rate)
3. Risk-adjusted return analysis
4. Calendar heatmap (일별 PnL)

첨부: Equity curve + Drawdown + Allocation pie chart

#### Risk Alert (WARNING/CRITICAL)

```
[WARNING] Drawdown Alert
Current DD: -12.5% (Threshold: -15%)
Peak: $10,500 | Current: $9,187.50

[CRITICAL] Circuit Breaker Triggered
System DD: -18.2% (Limit: -15%)
Action: All positions closed, entries blocked
Recovery: Manual reset required
```

#### System Health (WARNING/CRITICAL)

- Exchange connectivity loss/restore
- API rate limit 근접 (> 1000/1200 req/min)
- Order fill latency spike (> 5초)
- Data feed gap 감지
- Position drift (로컬 vs 거래소 불일치)

### 3.4 Visualization

#### 라이브러리 선택

| 라이브러리 | 용도 | 권장도 |
|-----------|------|--------|
| **matplotlib** | Equity curve, Drawdown, Bar chart | **최우선** — 서버 안정적, VBT 통해 이미 의존성 |
| **mplfinance** | 캔들스틱 차트 | 보조 — OHLCV 시각화 시 |
| **Plotly + Kaleido 1.0** | 고품질 인터랙티브 차트 | 월간 리포트 전용 — Chrome headless 필요 |
| **Pillow** | 텍스트 테이블 → 이미지 | 보조 — 복잡한 테이블 시 |

#### Chart 생성 패턴

```python
import matplotlib
matplotlib.use("Agg")  # 서버: GUI 없이 렌더링
import matplotlib.pyplot as plt
from io import BytesIO

def generate_equity_chart(
    dates: list,
    equity: list[float],
    benchmark: list[float] | None = None,
    drawdown: list[float] | None = None,
) -> bytes:
    """Equity curve + Drawdown 차트를 PNG bytes로 생성."""
    fig, axes = plt.subplots(
        2 if drawdown else 1, 1,
        figsize=(10, 6),
        height_ratios=[3, 1] if drawdown else [1],
        sharex=True,
    )
    ax1 = axes[0] if drawdown else axes

    # Equity curve
    ax1.plot(dates, equity, label="Portfolio", color="#3498DB", linewidth=1.5)
    if benchmark is not None:
        ax1.plot(dates, benchmark, label="BTC", color="#95A5A6", linewidth=1.0, alpha=0.7)
    ax1.set_ylabel("Equity ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown
    if drawdown:
        ax2 = axes[1]
        ax2.fill_between(dates, drawdown, 0, color="#E74C3C", alpha=0.3)
        ax2.set_ylabel("DD (%)")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
```

#### Discord 이미지 전송

```python
import aiohttp
import json

async def send_discord_embed_with_chart(
    webhook_url: str,
    embed: dict,
    chart_bytes: bytes,
    filename: str = "chart.png",
) -> bool:
    """Discord webhook으로 Embed + 이미지 전송 (multipart/form-data)."""
    embed["image"] = {"url": f"attachment://{filename}"}
    form = aiohttp.FormData()
    form.add_field("payload_json", json.dumps({"embeds": [embed]}), content_type="application/json")
    form.add_field("files[0]", chart_bytes, filename=filename, content_type="image/png")

    async with aiohttp.ClientSession() as session:
        async with session.post(webhook_url, data=form) as resp:
            return resp.status in (200, 204)
```

#### Telegram 이미지 전송 (aiogram 3.x)

```python
from aiogram import Bot
from aiogram.types import BufferedInputFile

async def send_telegram_chart(
    bot: Bot,
    chat_id: int,
    chart_bytes: bytes,
    caption: str,
) -> None:
    """Telegram으로 차트 이미지 전송."""
    photo = BufferedInputFile(chart_bytes, filename="chart.png")
    await bot.send_photo(chat_id=chat_id, photo=photo, caption=caption, parse_mode="HTML")
```

#### 모바일 최적화 가이드

- **해상도**: 150 DPI, 10x6 인치 (1500x900px) — 모바일에서 선명
- **폰트**: 12pt 이상 (작은 화면 가독성)
- **색상**: 고대비 팔레트 (다크 모드 대응)
- **레이아웃**: 2-panel (equity + drawdown) 이 모바일 최적

### 3.5 Python 라이브러리 권장

#### Discord

| 라이브러리 | 권장도 | 비고 |
|-----------|--------|------|
| **aiohttp 직접 webhook** | **최우선** | 이미 프로젝트에서 사용, 의존성 추가 불필요 |
| **disnake** | 양방향 필요 시 | discord.py 최고 fork, 활발히 유지보수 |

#### Telegram

| 라이브러리 | 권장도 | 비고 |
|-----------|--------|------|
| **aiogram 3.x** | **최우선** | 완전 asyncio native, trading bot에 최적 |
| **python-telegram-bot** | 대안 | 문서 우수하나 sync 기반, asyncio 프로젝트에 부적합 |

#### Freqtrade 참고 알림 기능

Freqtrade (가장 인기 있는 OSS crypto trading bot)의 Telegram 인터페이스:
- `/status`: 모든 오픈 트레이드 상태 (Trade ID, 방향, 미실현 PnL, 레버리지)
- `/profit`: ROI 요약, 전체 거래 수, 평균 duration, 최고 성과 페어, profit factor, 승률, max drawdown
- `/daily`: 최근 7일 일별 손익
- `/performance`: 코인별 수익 성과 순위
- `/balance`: 계좌 잔고
- `/stop`: 봇 즉시 중지

→ MC Coin Bot에서도 동일한 양방향 명령어 세트 구현 권장.

### 3.6 구현 패턴: Async Queue + Retry + SpamGuard

```python
class NotificationQueue:
    """비동기 알림 큐 — 거래 로직을 block하지 않음."""

    def __init__(self, max_retries: int = 3) -> None:
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._max_retries = max_retries

    async def enqueue(self, item: NotificationItem) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning("Notification queue full, dropping: {}", item.severity)

    async def _worker(self) -> None:
        while True:
            item = await self._queue.get()
            for attempt in range(self._max_retries):
                try:
                    await item.sender.send(item.payload)
                    break
                except Exception:
                    delay = 1.0 * (2 ** attempt)
                    await asyncio.sleep(delay)
            self._queue.task_done()


class SpamGuard:
    """동일 알림 반복 전송 방지 (cooldown 기반)."""

    def __init__(self, cooldown_seconds: int = 300) -> None:
        self._cooldown = cooldown_seconds
        self._sent: dict[str, float] = {}

    def should_send(self, key: str) -> bool:
        now = time.monotonic()
        if key in self._sent and (now - self._sent[key]) < self._cooldown:
            return False
        self._sent[key] = now
        return True
```

---

## 4. Production 배포 아키텍처

### 4.1 인프라 선택

| Provider | Spec | 비용 | Binance 지연 | 추천 |
|----------|------|------|-------------|------|
| **Oracle Cloud Always Free** | ARM 4 vCPU, 24GB RAM | **$0/month** | 도쿄 ~3ms | **개발/Paper** |
| **Hetzner CX22** | 2 vCPU, 4GB RAM | ~$4/month | 유럽 ~30ms | **Live (가성비)** |
| **AWS t3.micro** (Seoul) | 2 vCPU, 1GB RAM | ~$8/month | 서울 ~5ms | Live (저지연) |
| **DigitalOcean Basic** | 1 vCPU, 1GB RAM | $6/month | 싱가포르 ~10ms | 대안 |

일봉 TSMOM 전략은 ms 단위 지연이 무의미하므로 **Oracle Free Tier → Hetzner** 순으로 권장.

> **Spot Instance 사용 절대 금지**: 예고 없이 종료되면 포지션이 방치됨.

### 4.2 Docker Compose 구성

```yaml
# docker-compose.yml
services:
  bot:
    build: .
    restart: unless-stopped
    stop_grace_period: 30s
    env_file: .env
    volumes:
      - ./data:/app/data          # SQLite, Parquet, JSONL
      - ./logs:/app/logs          # loguru 로그
      - ./config:/app/config      # YAML 설정
    secrets:
      - binance_api_key
      - binance_secret
    depends_on:
      - prometheus
    networks:
      - trading

  prometheus:
    image: prom/prometheus:v2.52.0
    restart: unless-stopped
    volumes:
      - ./infra/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "127.0.0.1:9090:9090"    # 로컬만 노출
    networks:
      - trading

  grafana:
    image: grafana/grafana:11.0.0
    restart: unless-stopped
    volumes:
      - grafana_data:/var/lib/grafana
      - ./infra/grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "127.0.0.1:3000:3000"    # 로컬만 노출
    environment:
      - GF_SECURITY_ADMIN_PASSWORD__FILE=/run/secrets/grafana_password
    secrets:
      - grafana_password
    networks:
      - trading

volumes:
  prometheus_data:
  grafana_data:

secrets:
  binance_api_key:
    file: ./secrets/binance_api_key.txt
  binance_secret:
    file: ./secrets/binance_secret.txt
  grafana_password:
    file: ./secrets/grafana_password.txt

networks:
  trading:
    driver: bridge
```

### 4.3 Production Dockerfile (uv 최적화)

```dockerfile
# ---- Stage 1: Builder ----
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 의존성 레이어 (코드보다 먼저 — Docker cache 활용)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 소스 코드 + 프로젝트 설치
COPY src/ src/
COPY config/ config/
COPY main.py ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- Stage 2: Runtime ----
FROM python:3.13-slim-bookworm

WORKDIR /app

# 보안: non-root user
RUN groupadd -r bot && useradd -r -g bot bot

# Builder에서 venv만 복사
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/config /app/config
COPY --from=builder /app/main.py /app/main.py

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER bot

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

ENTRYPOINT ["python", "-m", "src.live.main"]
```

핵심:
- **Multi-stage build**: builder에서 `uv sync`, runtime에는 .venv만 복사
- **`UV_COMPILE_BYTECODE=1`**: .pyc 사전 컴파일로 startup 단축
- **의존성 레이어 분리**: Docker cache 극대화
- **Non-root user**: `bot` 사용자로 실행
- **PID 1 직접 실행**: shell form 금지 → signal 수신 보장

### 4.4 Graceful Shutdown

```python
import asyncio
import signal
from loguru import logger

class LiveRunner:
    """Production EDA Runner with graceful shutdown."""

    def __init__(self) -> None:
        self._shutdown_event = asyncio.Event()

    def _setup_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown, sig)

    def _handle_shutdown(self, sig: signal.Signals) -> None:
        logger.warning("Received {}, initiating graceful shutdown...", sig.name)
        self._shutdown_event.set()

    async def run(self) -> None:
        self._setup_signal_handlers()
        try:
            # ... 컴포넌트 초기화 ...
            while not self._shutdown_event.is_set():
                await asyncio.sleep(1)
        finally:
            await self._graceful_shutdown()

    async def _graceful_shutdown(self) -> None:
        logger.info("Saving state...")
        # 1. 신규 주문 중지
        # 2. 미체결 주문 취소 (exchange API)
        # 3. 포지션 상태 SQLite 저장
        # 4. EventBus flush + stop
        # 5. 로그 flush
        logger.info("Shutdown complete.")
```

**Position 처리 정책**: 1D timeframe이므로 **Keep Positions + State 저장** 권장.

### 4.5 Monitoring & Observability

#### Prometheus Metrics

```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Trading Metrics
ORDERS_TOTAL = Counter("trading_orders_total", "Total orders", ["symbol", "side", "strategy"])
FILLS_TOTAL = Counter("trading_fills_total", "Total fills", ["symbol", "side"])
SLIPPAGE_BPS = Histogram("trading_slippage_bps", "Slippage in bps", ["symbol"],
                         buckets=[0, 1, 2, 5, 10, 20, 50, 100])
ORDER_LATENCY = Histogram("trading_order_latency_seconds", "Order-to-fill latency", ["symbol"],
                          buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10])

# Portfolio Metrics
EQUITY = Gauge("trading_equity_usd", "Total portfolio equity")
DRAWDOWN = Gauge("trading_drawdown_pct", "Current drawdown %")
POSITION_COUNT = Gauge("trading_open_positions", "Open position count")

# System Metrics
EVENTBUS_QUEUE = Gauge("eda_eventbus_queue_size", "EventBus queue depth")
HEARTBEAT_TS = Gauge("trading_last_heartbeat", "Last heartbeat unix timestamp")
```

#### 핵심 모니터링 지표 & Alert

| Category | Metric | Alert Threshold |
|----------|--------|-----------------|
| **Uptime** | Last heartbeat | > 5분 무응답 |
| **Latency** | Order-to-fill | > 5초 (daily) |
| **Slippage** | Expected vs actual | > 20 bps |
| **Fill Rate** | Filled / submitted | < 90% |
| **Drawdown** | Current DD % | > 15% (W), > 25% (C) |
| **Queue** | EventBus size | > 80% capacity |
| **Drift** | Local vs exchange | Any mismatch |
| **API Rate** | Binance calls/min | > 1000 (limit: 1200) |

#### Logging (loguru, 현재 유지 + JSON 추가)

```python
# Production: JSON stdout + file rotation
logger.remove()
logger.add(sys.stdout, format=json_format, level="INFO")
logger.add("/app/logs/bot.log", rotation="100 MB", retention="30 days")
logger.add("/app/logs/trades.log",
           filter=lambda r: "trade" in r["extra"],
           rotation="50 MB", retention="90 days")
```

### 4.6 Security

#### API Key 관리

| 수준 | 방법 | 적합 상황 |
|------|------|-----------|
| Level 1 | `.env` file (gitignored) | 로컬 개발 |
| Level 2 | **Docker Secrets (file mount)** | 단일 VPS Docker — **권장** |
| Level 3 | SOPS (암호화된 git) | 소규모 팀 |
| Level 4 | HashiCorp Vault | 다중 서비스, key rotation |

#### Binance API Key 보안 체크리스트

```
[x] Enable Spot & Margin Trading
[ ] Enable Futures (필요시만)
[ ] Enable Withdrawals (절대 비활성화!)
[x] Restrict access to trusted IPs only
[x] VPS 고정 IP 등록
[x] 데이터 조회용/거래용 API key 분리
```

#### Binance Rate Limits (2026)

| Limit | Spot | Futures |
|-------|------|---------|
| Request Weight | 6,000/min per IP | 2,400/min per IP |
| Order Count | 50/10s, 160,000/24h | 1,200/min |
| WebSocket | 5 msg/s per connection | 10 connections per IP |

429 → exponential backoff. 반복 위반 → 418 (IP ban).

#### VPS Hardening

```bash
# SSH key-only
sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Firewall
ufw default deny incoming
ufw allow 22/tcp       # SSH
ufw allow 127.0.0.1:3000  # Grafana (로컬만)
ufw allow 127.0.0.1:9090  # Prometheus (로컬만)
ufw enable

# Fail2ban + Unattended upgrades
apt install fail2ban unattended-upgrades
```

### 4.7 Risk Management for Live

#### Circuit Breaker 계층

| Level | Trigger | Action |
|-------|---------|--------|
| **Order-level** | 단일 주문 > 자본의 30% | Reject |
| **Daily Loss** | 일일 손실 > 5% | 신규 주문 중지, 알림 |
| **Drawdown** | MDD > 15% | 전체 포지션 청산, 봇 중지 |
| **System** | API 연속 실패 > 5회 | 봇 중지, 긴급 알림 |
| **Manual Kill** | `/kill` 명령 or KILL_SWITCH 파일 | 즉시 전량 청산 |

현재 `EDARiskManager`에 `enable_circuit_breaker` 플래그가 이미 존재. Production에서 `True`로 활성화.

#### Kill Switch 트리거 방법

1. **Telegram 명령**: `/kill` 메시지 수신 시 활성화
2. **파일 기반**: `/app/KILL_SWITCH` 파일 존재 시 (가장 간단)
3. **HTTP endpoint**: `POST /api/kill` (인증 필요)

#### Reconciliation (교차 검증)

```python
async def reconcile(pm: EDAPortfolioManager, exchange: ccxt.binance) -> None:
    """로컬 상태와 거래소 상태 비교."""
    exchange_positions = await exchange.fetch_positions()
    local_positions = pm.positions

    for symbol, local_pos in local_positions.items():
        exchange_pos = next(
            (p for p in exchange_positions if p["symbol"] == symbol), None
        )
        if exchange_pos is None:
            logger.error("DRIFT: {} exists locally but not on exchange", symbol)
            continue

        size_diff = abs(local_pos.size - exchange_pos["contracts"])
        if size_diff > 0.001:
            logger.error("POSITION DRIFT: {} local={} exchange={}",
                        symbol, local_pos.size, exchange_pos["contracts"])
            # 거래소 상태를 truth로 사용
```

주기: **매 bar 처리 후** + **매 시간 1회** 정기 reconciliation.

#### Exchange Disconnection 대응

| 시나리오 | 대응 |
|---------|------|
| WebSocket 끊김 | 자동 재연결 (exponential backoff, max 5회) |
| REST API 타임아웃 | 재시도 (tenacity, 3회) |
| 장기 disconnection (> 5분) | 봇 일시 중지, 알림, 재연결 후 reconciliation |
| Exchange 점검 | 사전 감지 (Binance 공지), 봇 예약 중지 |

### 4.8 State Recovery (재시작 복구)

```python
class StateManager:
    """봇 상태 저장/복구 — SQLite bot_state 테이블 활용."""

    async def save(self, pm: EDAPortfolioManager) -> None:
        state = {
            "positions": {s: p.to_dict() for s, p in pm.positions.items()},
            "cash": pm.cash,
            "last_bar_timestamp": pm.last_bar_ts.isoformat(),
            "peak_equity": rm.peak_equity,
        }
        await db.execute(
            "INSERT OR REPLACE INTO bot_state (key, value) VALUES (?, ?)",
            ("pm_state", json.dumps(state))
        )

    async def load(self) -> dict | None:
        row = await db.execute("SELECT value FROM bot_state WHERE key = ?", ("pm_state",))
        return json.loads(row["value"]) if row else None
```

복구 순서:
1. State file/DB 로드
2. Exchange position 조회 (`fetch_positions`)
3. **Reconciliation** (state vs exchange) — exchange를 truth로 사용
4. 놓친 bar 보충 (`fetch_ohlcv` since last_processed_ts)
5. 정상 운영 재개

---

## 5. Shadow vs Paper 분석 & 전환 파이프라인

### 5.1 Shadow Trading은 건너뛸 수 있는가?

**결론: 건너뛸 수 있다. 단, Paper 초기에 Shadow 모드로 짧게 검증하는 것을 권장.**

#### 비교 분석

| 항목 | Shadow | Paper | 비고 |
|------|--------|-------|------|
| **DataFeed** | LiveDataFeed (WebSocket) | LiveDataFeed (WebSocket) | **동일** |
| **Executor** | ShadowExecutor (로깅만) | BacktestExecutor (시뮬 체결) | Executor만 다름 |
| **Strategy/PM/RM** | 동일 | 동일 | **동일** |
| **DB** | 필요 (시그널 저장) | 필요 (거래 저장) | **동일** |
| **알림** | 필요 (시그널 알림) | 필요 (거래 알림) | **동일** |
| **배포** | 필요 (VPS + Docker) | 필요 (VPS + Docker) | **동일** |
| **구현 비용** | 공유 인프라 구축 | 공유 인프라 구축 + 0 | **0 추가 비용** |
| **검증 범위** | 데이터 파이프라인 + 시그널 | 데이터 + 시그널 + 체결 + PnL | Paper가 상위집합 |
| **위험** | 없음 | 없음 (시뮬레이션) | **동일** |

#### 왜 건너뛸 수 있는가?

1. **Paper는 Shadow의 상위집합**: Paper에서 시그널 생성(Shadow의 역할) + 시뮬레이션 체결까지 모두 검증
2. **공유 인프라 비용이 99%**: LiveDataFeed, DB, 알림, Docker 등이 양쪽 동일. Executor 교체는 설정 한 줄
3. **BacktestExecutor는 이미 검증됨**: 1623개 테스트를 통과한 EDA BacktestExecutor를 그대로 사용
4. **Paper도 실제 자금 위험 없음**: 시뮬레이션 체결이므로 Shadow와 리스크 동일

#### 권장 전략: Paper-First with Shadow Warmup

```
[권장 전환 경로]

1. 공유 인프라 구축 (LiveDataFeed, DB, 알림, Docker, 배포)
      ↓
2. VPS에 Paper 모드로 배포
      ↓
3. 첫 2-4시간: Shadow 모드 (--mode shadow)  ← Executor만 교체
   → 데이터 수신 정상? WebSocket 안정? 시그널 생성 정상?
      ↓ (확인되면)
4. Paper 모드로 전환 (--mode paper)  ← 설정 변경만
   → 2-4주 운영: 시뮬 체결, PnL 추적, 알림, 리포트 검증
      ↓ (결과 만족하면)
5. Canary/Live 모드로 전환 (LiveExecutor 구현 후)
```

Shadow는 **별도 Phase가 아니라 Paper 배포 후 최초 검증 단계**로 취급합니다.

### 5.2 전체 전환 파이프라인

```
Phase 1-5: VBT/EDA Backtest (✅ 완료)
      ↓
Phase 6: 공유 인프라 구축                          ← 핵심 작업
      │  LiveDataFeed, DB(SQLite), 알림,
      │  Dockerfile, docker-compose, LiveRunner
      ↓
Phase 7: Paper Trading 배포 (VPS)                  ← 첫 번째 실전
      │  Shadow warmup (2-4h) → Paper 전환
      │  2-4주 운영, 시그널/체결/PnL 검증
      ↓
Phase 8: Live Trading (LiveExecutor 구현)           ← 실제 자금
      │  Alpha (1 asset, 10%) → Beta (3, 30%) → Full (8, 100%)
      ↓
운영 & 모니터링
```

### 5.3 Phase별 상세

#### Phase 6: 공유 인프라 구축 (모든 모드의 기반)

이 Phase가 가장 많은 작업량을 포함합니다. Shadow든 Paper든 Live든 이 인프라 없이는 시작할 수 없습니다.

| 컴포넌트 | 설명 | 필요한 이유 |
|----------|------|------------|
| **LiveDataFeed** | WebSocket 1m 실시간 수신 + CandleAggregator | 모든 실시간 모드의 데이터 소스 |
| **LiveRunner** | Graceful shutdown + signal handler + TaskGroup | 프로세스 관리, 24/7 안정성 |
| **TradePersistence** | SQLite 거래/시그널/equity 기록 | 결과 분석, 재시작 복구 |
| **StateManager** | 봇 상태 저장/복구 | VPS 재시작 시 포지션 유지 |
| **NotificationEngine** | EventBus subscriber → Discord/Telegram | 실시간 알림 수신 |
| **Dockerfile + docker-compose** | 컨테이너화, Prometheus/Grafana | VPS 배포 필수 |
| **환경별 설정** | paper.yaml / live.yaml 분리 | 모드 전환 용이 |

#### Phase 7: Paper Trading 배포

| 단계 | 기간 | 모드 | 검증 항목 |
|------|------|------|-----------|
| **7-A: Shadow Warmup** | 2-4시간 | `--mode shadow` | WebSocket 안정성, 1m bar 수신, CandleAggregator 정상, 시그널 생성 일관성 |
| **7-B: Paper 전환** | 2-4주 | `--mode paper` | 시뮬 체결, PnL 계산, equity curve, 알림 정상, 리포트 생성, 재시작 복구 |
| **7-C: Paper 평가** | — | 분석 | 백테스트 대비 parity (수익 부호, 거래 수, Sharpe 유사) |

**Paper 모드 성공 기준:**
- WebSocket 24시간 무중단 연결 유지
- 시그널이 백테스트와 일관 (동일 bar에서 동일 방향)
- 시뮬 PnL 부호가 백테스트와 일치
- 알림 정상 수신 (Discord + Telegram)
- 재시작 후 상태 정상 복구
- Daily report 자동 생성

#### Phase 8: Live Trading

| 단계 | Assets | Capital | 기간 | 성공 기준 |
|------|--------|---------|------|-----------|
| **8-A: LiveExecutor 구현** | — | — | — | TestnetExecutor → LiveExecutor |
| **8-B: Testnet** | 1 (BTC) | 가상 | 1주 | API 주문 체결 정상 |
| **8-C: Alpha** | 1 (BTC) | $1,000 (10%) | 1개월 | Sharpe > 0, MDD < 10% |
| **8-D: Beta** | 3 (BTC,ETH,SOL) | $3,000 (30%) | 1개월 | 멀티에셋 정상 |
| **8-E: Full** | 8 (전체) | $10,000 (100%) | 계속 | 백테스트 parity |

### 5.4 Binance Testnet 참고사항

Binance Testnet은 제약이 있으므로 Paper 모드가 더 현실적입니다:

| 항목 | Binance Testnet | Paper (BacktestExecutor) |
|------|----------------|------------------------|
| **가격 데이터** | Testnet 전용 (실제와 다름) | 실제 WebSocket 가격 사용 |
| **유동성** | 제한적, 비현실적 체결 | 시뮬 체결 (CostModel 반영) |
| **지원 페어** | 제한적 | 전체 페어 가능 |
| **목적** | API 연동/에러 핸들링 테스트 | 전략 성과 검증 |

**결론**: Paper 모드가 전략 검증에 더 적합합니다. Testnet은 LiveExecutor 구현 시 API 호출 테스트용으로만 사용.

### 5.3 LiveDataFeed 설계

```python
class LiveDataFeed:
    """실시간 WebSocket 데이터 피드 — HistoricalDataFeed 대체."""

    def __init__(
        self,
        client: BinanceClient,
        symbols: list[str],
        aggregator: CandleAggregator,
        target_timeframe: str,
    ) -> None:
        self._client = client
        self._symbols = symbols
        self._aggregator = aggregator
        self._target_tf = target_timeframe

    async def start(self, bus: EventBus) -> None:
        """WebSocket 스트림 시작 — 1m 캔들 실시간 수신."""
        while True:
            try:
                for symbol in self._symbols:
                    ohlcv = await self._client._exchange.watch_ohlcv(symbol, "1m")
                    for candle in ohlcv:
                        bar = self._to_bar_event(symbol, candle)
                        await bus.publish(bar)

                        # CandleAggregator로 target TF 집계
                        completed = self._aggregator.update(bar)
                        if completed:
                            await bus.publish(completed)
                            await bus.flush()  # bar-by-bar 이벤트 체인 보장
            except NetworkError:
                logger.warning("WebSocket disconnected, reconnecting...")
                await asyncio.sleep(5)
```

### 5.4 LiveExecutor 설계

```python
class LiveExecutor:
    """Production Executor — ExecutorPort 구현."""

    def __init__(self, client: BinanceClient) -> None:
        self._client = client

    async def execute(self, order: OrderRequestEvent) -> FillEvent | None:
        try:
            result = await self._client._exchange.create_order(
                symbol=order.symbol,
                type=order.order_type.lower(),
                side=order.side.lower(),
                amount=self._client._exchange.amount_to_precision(
                    order.symbol, abs(order.notional_usd / self._last_price)
                ),
                params={"newClientOrderId": order.client_order_id},
            )
            return self._to_fill_event(order, result)
        except InsufficientFunds:
            logger.error("Insufficient funds for {}", order.symbol)
            return None
        except NetworkError:
            logger.error("Network error executing {}", order.client_order_id)
            # Retry는 OMS 레벨에서 처리
            raise
```

### 5.5 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --group dev --group research
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pyright src/
      - run: uv run pytest --cov=src --cov-report=xml

  build-image:
    needs: lint-and-test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}

  deploy-paper:
    needs: build-image
    environment: paper-trading
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to paper trading server
        run: |
          ssh ${{ secrets.VPS_HOST }} \
            "docker compose -f docker-compose.paper.yml up -d"

  deploy-live:
    needs: deploy-paper
    environment:
      name: production
      # Required reviewers + 24h wait timer
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production (manual approval required)
        run: |
          ssh ${{ secrets.VPS_HOST }} \
            "docker compose -f docker-compose.live.yml up -d"
```

### 5.6 환경별 설정 분리

```
config/
  default.yaml        # 공통 (전략 파라미터, timeframe)
  paper.yaml           # Paper: enable_circuit_breaker=false, testnet=true
  production.yaml      # Live: enable_circuit_breaker=true, capital=10000
```

```python
# Pydantic Settings (환경별 오버라이드)
from pydantic_settings import BaseSettings, SettingsConfigDict

class TradingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MC_",
        env_nested_delimiter="__",
        secrets_dir="/run/secrets",
    )

    env: str = "development"
    binance_api_key: SecretStr
    binance_secret: SecretStr
    initial_capital: float = 10000.0
    enable_circuit_breaker: bool = True
    max_daily_loss_pct: float = 5.0
    execution_mode: ExecutionMode = ExecutionMode.PAPER
```

---

## 6. 종합 구현 로드맵

### 전체 흐름도

```
┌──────────────────────────────────────────────────────────────────┐
│  Phase 6: 공유 인프라 구축 (P0) ← 가장 큰 작업량                   │
│                                                                  │
│  6-A: 실시간 데이터       6-B: 데이터 영속화    6-C: 배포 환경     │
│  ├ LiveDataFeed           ├ TradePersistence    ├ Dockerfile      │
│  ├ LiveRunner             ├ StateManager        ├ docker-compose  │
│  └ CLI run-live           └ SQLite schema       └ 환경별 설정     │
│                                                                  │
│  6-D: 알림 시스템 (최소)                                          │
│  ├ NotificationEngine (EventBus subscriber)                      │
│  ├ Discord 거래 알림 (기존 확장)                                   │
│  └ Telegram Critical alert (aiogram)                             │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Phase 7: Paper Trading 배포 (P1)                                │
│                                                                  │
│  7-A: Shadow Warmup (2-4h)  → 데이터 파이프라인 검증              │
│  7-B: Paper 운영 (2-4주)    → 시뮬 체결, PnL, 알림, 리포트 검증   │
│  7-C: 결과 평가             → 백테스트 parity 확인                │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Phase 7.5: 알림 고도화 + 모니터링 (P2, Paper 운영 중 병행)       │
│                                                                  │
│  ├ ChartGenerator (matplotlib equity/drawdown PNG)               │
│  ├ ReportScheduler (daily/weekly/monthly)                        │
│  ├ Telegram 양방향 명령 (/status, /kill, /report)                │
│  ├ Prometheus + Grafana 대시보드                                  │
│  └ Watchdog + KillSwitch                                        │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Phase 8: Live Trading (P3)                                      │
│                                                                  │
│  8-A: LiveExecutor 구현 + Testnet 검증                           │
│  8-B: Reconciliation (거래소 vs 로컬 교차 검증)                   │
│  8-C: Alpha (BTC 1개, $1,000) → Beta (3개, $3,000)              │
│  8-D: Full (8개, $10,000)                                        │
└──────────────────────────────────────────────────────────────────┘
```

### Phase 6: 공유 인프라 구축 (P0 — Paper/Shadow/Live 공통)

> Shadow든 Paper든 Live든, 이 인프라 없이는 시작할 수 없습니다.
> **이 Phase가 전체 작업의 70%를 차지합니다.**

#### 6-A: 실시간 데이터 + 프로세스 관리

| Task | 설명 | 의존성 | 테스트 |
|------|------|--------|--------|
| **6-A-1** | `LiveDataFeed` — WebSocket 1m 수신 + CandleAggregator 연동 | BinanceClient, CandleAggregator | 단위 + 통합 |
| **6-A-2** | `LiveRunner` — Graceful shutdown + signal handler + asyncio.TaskGroup | LiveDataFeed | 단위 |
| **6-A-3** | CLI `run-live` 명령 — `--mode shadow|paper|canary|live` 플래그 | LiveRunner | 통합 |

#### 6-B: 데이터 영속화

| Task | 설명 | 의존성 | 테스트 |
|------|------|--------|--------|
| **6-B-1** | SQLite 스키마 생성 (trades, equity_snapshots, positions_history, bot_state) | — | 단위 |
| **6-B-2** | `TradePersistence` — EventBus subscriber, FillEvent/BalanceUpdate 저장 | aiosqlite, EventBus | 단위 |
| **6-B-3** | `StateManager` — 재시작 복구 (save/load/reconcile) | SQLite | 단위 |

#### 6-C: 배포 환경

| Task | 설명 | 의존성 |
|------|------|--------|
| **6-C-1** | `Dockerfile` (uv multi-stage build) | — |
| **6-C-2** | `docker-compose.yml` (bot + prometheus + grafana) | Dockerfile |
| **6-C-3** | 환경별 설정 분리 (paper.yaml / live.yaml) + Pydantic Settings | — |
| **6-C-4** | VPS 프로비저닝 (Oracle Free / Hetzner) + Docker Secrets | — |

#### 6-D: 알림 시스템 (최소 MVP)

| Task | 설명 | 의존성 |
|------|------|--------|
| **6-D-1** | `NotificationEngine` — EventBus subscriber (Fill, CircuitBreaker, Balance) | EventBus |
| **6-D-2** | Discord 거래 알림 확장 (기존 DiscordNotifier + 이미지 첨부) | aiohttp |
| **6-D-3** | `TelegramNotifier` — aiogram 3.x, CRITICAL alert 전용 | `uv add aiogram` |
| **6-D-4** | `NotificationQueue` — async retry + SpamGuard | — |

### Phase 7: Paper Trading 배포 (P1)

> Phase 6 완료 후, VPS에 배포하여 실시간 검증을 시작합니다.

| Task | 설명 |
|------|------|
| **7-A** | VPS에 Docker 배포 (`--mode shadow`로 시작) |
| **7-B** | Shadow Warmup (2-4시간): WebSocket 안정성, 시그널 생성 확인 |
| **7-C** | Paper 전환 (`--mode paper`): 2-4주 운영 |
| **7-D** | 결과 분석: 백테스트 parity (수익 부호, 거래 수, Sharpe) |

### Phase 7.5: 알림 고도화 + 모니터링 (P2, Paper 운영 중 병행)

> Paper 운영이 안정화되면, 운영 중에 점진적으로 추가합니다.

| Task | 설명 | 의존성 |
|------|------|--------|
| **7.5-A** | `ChartGenerator` — matplotlib equity/drawdown PNG | matplotlib |
| **7.5-B** | Discord/Telegram 차트 이미지 첨부 | ChartGenerator |
| **7.5-C** | `ReportScheduler` — daily/weekly/monthly 자동 리포트 | ChartGenerator, DB |
| **7.5-D** | Telegram 양방향 명령 (`/status`, `/kill`, `/report`) | aiogram |
| **7.5-E** | Prometheus metrics 노출 + Grafana 대시보드 | `uv add prometheus-client` |
| **7.5-F** | `Watchdog` — heartbeat 감시 + 알림 | NotificationEngine |
| **7.5-G** | CI/CD Pipeline (GitHub Actions: lint → test → build → deploy) | Dockerfile |

### Phase 8: Live Trading (P3)

> Paper 결과가 만족스러우면, LiveExecutor를 구현하고 실제 자금을 투입합니다.

| Task | 설명 | 의존성 |
|------|------|--------|
| **8-A** | `LiveExecutor` — CCXT 주문 제출 (ExecutorPort 구현) | BinanceClient |
| **8-B** | `Reconciliation` — 거래소 vs 로컬 포지션 교차 검증 | LiveExecutor |
| **8-C** | `KillSwitch` — 긴급 정지 (Telegram `/kill` + 파일 기반) | Telegram, OMS |
| **8-D** | Testnet 검증 (1주, API 호출 테스트) | LiveExecutor |
| **8-E** | Alpha rollout — BTC 1개, $1,000 (10%), 1개월 | 8-A~D |
| **8-F** | Beta rollout — 3개 (BTC,ETH,SOL), $3,000 (30%), 1개월 | Alpha 성공 |
| **8-G** | Full rollout — 8개 전체, $10,000 (100%) | Beta 성공 |
| **8-H** | Feature flags (전략 활성화/비활성화 YAML) | — |

### 의존성 요약

```
Phase 6에서 추가할 Python 의존성 (3개만):
  uv add aiosqlite          # SQLite async (trade persistence, state manager)
  uv add aiogram            # Telegram bot (알림, 양방향 명령)
  uv add prometheus-client   # Prometheus metrics (Phase 7.5)

기존 의존성 활용 (추가 설치 불필요):
  aiohttp                   # Discord webhook (이미 있음)
  matplotlib                # Chart 생성 (VBT 통해 이미 있음)
  ccxt                      # Binance WebSocket + REST API (이미 있음)
  loguru                    # Logging (이미 있음)
  pydantic-settings          # 환경별 설정 (이미 있음)

분석 전용 (선택, Phase 7 이후):
  uv add --group research duckdb  # Parquet+SQLite 크로스 쿼리
```

### Phase 6 구현 순서 (권장)

```
Week 1:  6-C-1 Dockerfile + 6-C-3 환경 설정  (배포 기반)
         6-B-1 SQLite 스키마                    (데이터 기반)

Week 2:  6-A-1 LiveDataFeed                    (핵심 컴포넌트)
         6-B-2 TradePersistence                (EventBus 연동)

Week 3:  6-A-2 LiveRunner                      (프로세스 관리)
         6-B-3 StateManager                    (재시작 복구)
         6-D-1 NotificationEngine              (알림 연동)

Week 4:  6-D-2~4 Discord/Telegram 알림         (알림 완성)
         6-A-3 CLI run-live                     (통합)
         6-C-2 docker-compose                  (배포 완성)
         6-C-4 VPS 프로비저닝                   (배포 실행)

Week 5:  Phase 7 시작 — VPS 배포 + Shadow Warmup + Paper 전환
```

---

## Sources

### Database
- [QuestDB vs TimescaleDB Benchmark](https://questdb.com/blog/timescaledb-vs-questdb-comparison/)
- [DuckDB vs SQLite: Comprehensive Comparison 2026](https://www.analyticsvidhya.com/blog/2026/01/duckdb-vs-sqlite/)
- [From SQLite to DuckDB: Embedded Analytics Is Here](https://medium.com/@Quaxel/from-sqlite-to-duckdb-embedded-analytics-is-here-da79263a7fea)
- [Trading Data Analytics with DuckDB and Parquet](https://medium.com/quant-factory/trading-data-analytics-part-1-first-steps-with-duckdb-and-parquet-files-f74fd2869372)
- [TimescaleDB for Algorithmic Trading](https://siddharthqs.com/introduction-to-timescaledb-for-algorithmic-trading)
- [InfluxDB 3.0 Release - VentureBeat](https://venturebeat.com/enterprise-analytics/influxdata-releases-influxdb-3-0-product-suite-for-time-series-analytics)
- [Python eventsourcing Library](https://github.com/pyeventsourcing/eventsourcing)
- [Event Sourcing Database Architecture - Redpanda](https://www.redpanda.com/guides/event-stream-processing-event-sourcing-database)
- [Freqtrade - SQLite-based Crypto Trading Bot](https://github.com/freqtrade/freqtrade)

### Notification
- [Freqtrade Telegram Usage](https://www.freqtrade.io/en/stable/telegram-usage/)
- [aiogram 3.x Documentation](https://docs.aiogram.dev/en/dev-3.x/)
- [Discord Webhook Embeds Guide](https://birdie0.github.io/discord-webhooks-guide/)
- [Discord Rate Limits](https://discord.com/developers/docs/topics/rate-limits)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Plotly Kaleido 1.0](https://plotly.com/blog/kaleido-the-next-generation/)
- [99bitcoins - Best Crypto Trading Signals 2026](https://99bitcoins.com/cryptocurrency/best-crypto-signals/)

### Production Deployment
- [Step-by-Step Crypto Trading Bot Development 2026](https://appinventiv.com/blog/crypto-trading-bot-development/)
- [Trading Bot Crypto: Complete Guide 2026](https://tickerly.net/trading-bot-crypto-complete-guide-2026/)
- [Binance API Rate Limits](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/limits)
- [Using uv in Docker](https://docs.astral.sh/uv/guides/integration/docker/)
- [Production-ready Python Docker with uv](https://hynek.me/articles/docker-uv/)
- [Prometheus with Docker Compose](https://last9.io/blog/prometheus-with-docker-compose/)
- [Python Monitoring with Prometheus](https://betterstack.com/community/guides/monitoring/prometheus-python-metrics/)
- [Docker Compose Secrets Management](https://phase.dev/blog/docker-compose-secrets/)
- [Graceful Shutdowns with asyncio](https://roguelynn.com/words/asyncio-graceful-shutdowns/)
- [CCXT Documentation](https://docs.ccxt.com/)
- [Pydantic Settings v2](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [7 Best Crypto VPS Hosting 2026](https://vettedpropfirms.com/best-crypto-vps/)
- [FIA Automated Trading Risk Controls Best Practices](https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf)
