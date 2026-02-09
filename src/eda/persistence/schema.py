"""SQL DDL 상수 — SQLite 스키마 정의.

5개 테이블: trades, equity_snapshots, positions_history, risk_events, bot_state.
모두 IF NOT EXISTS로 멱등하게 생성됩니다.
"""

SCHEMA_SQL = """
-- 거래 기록
CREATE TABLE IF NOT EXISTS trades (
    client_order_id TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    fill_price      REAL,
    fill_qty        REAL,
    fee             REAL DEFAULT 0.0,
    target_weight   REAL,
    notional_usd    REAL,
    status          TEXT NOT NULL DEFAULT 'FILLED',
    strategy_name   TEXT,
    timestamp       TEXT NOT NULL,
    correlation_id  TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);

-- 자산 스냅샷
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    total_equity    REAL NOT NULL,
    available_cash  REAL NOT NULL,
    margin_used     REAL DEFAULT 0.0,
    timestamp       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_snapshots(timestamp);

-- 포지션 히스토리
CREATE TABLE IF NOT EXISTS positions_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    direction       INTEGER NOT NULL,
    size            REAL NOT NULL,
    avg_entry_price REAL NOT NULL,
    unrealized_pnl  REAL DEFAULT 0.0,
    realized_pnl    REAL DEFAULT 0.0,
    timestamp       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions_history(symbol);

-- 리스크 이벤트
CREATE TABLE IF NOT EXISTS risk_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT NOT NULL,
    reason          TEXT,
    message         TEXT,
    alert_level     TEXT,
    close_all       INTEGER DEFAULT 0,
    timestamp       TEXT NOT NULL
);

-- 봇 상태 (key-value 저장)
CREATE TABLE IF NOT EXISTS bot_state (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""
