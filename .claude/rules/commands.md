---
paths:
  - "**"
---

# CLI Commands

## Environment Setup

```bash
# Install dependencies
uv sync --group dev --group research

# Activate virtual environment
source .venv/bin/activate
```

## Code Quality

```bash
# Lint check
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix . && uv run ruff format .

# Type check (strict mode)
uv run pyright src/
```

## Testing

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=src --cov-report=html

# Specific test file
uv run pytest tests/unit/test_portfolio.py

# Pattern matching
uv run pytest -k "test_tsmom"
```

## Data Ingestion (Medallion Architecture)

```bash
# Bronze → Silver full pipeline (OHLCV 1m)
uv run mcbot ingest pipeline BTC/USDT --year 2024 --year 2025

# Individual layers
uv run mcbot ingest bronze BTC/USDT --year 2024 --year 2025
uv run mcbot ingest silver BTC/USDT --year 2024 --year 2025

# Validate data integrity
uv run mcbot ingest validate data/silver/BTC_USDT_1m_2025.parquet

# Bulk download (top N symbols)
uv run mcbot ingest bulk-download --top 100 --year 2024 --year 2025

# Data status
uv run mcbot ingest info

# Derivatives (Funding Rate, OI, LS Ratio, Taker Ratio)
uv run mcbot ingest derivatives pipeline BTC/USDT --year 2024 --year 2025
uv run mcbot ingest derivatives batch            # 8 Tier-1/2 assets
uv run mcbot ingest derivatives batch --dry-run  # Preview targets
uv run mcbot ingest derivatives info BTC/USDT --year 2024 --year 2025

# On-chain (DeFiLlama, Coin Metrics, Fear & Greed, Blockchain.com, Etherscan, mempool)
uv run mcbot ingest onchain pipeline defillama stablecoin_total       # Single dataset
uv run mcbot ingest onchain pipeline coinmetrics btc_metrics
uv run mcbot ingest onchain batch --type all                          # All 22 datasets
uv run mcbot ingest onchain batch --type stablecoin                   # By category
uv run mcbot ingest onchain batch --type coinmetrics
uv run mcbot ingest onchain batch --dry-run                           # Preview targets
uv run mcbot ingest onchain info                                      # Data inventory
uv run mcbot ingest onchain info --type sentiment                     # By category
```

## Backtest (VBT — Vectorized)

```bash
# List available strategies
uv run mcbot backtest strategies

# Strategy info
uv run mcbot backtest info

# Run backtest (YAML config 기반)
uv run mcbot backtest run config/default.yaml
uv run mcbot backtest run config/default.yaml --report    # QuantStats HTML
uv run mcbot backtest run config/default.yaml --advisor   # Strategy Advisor
uv run mcbot backtest run config/default.yaml -V          # Verbose

# Parameter optimization (VW-TSMOM specific)
uv run mcbot backtest optimize BTC/USDT

# Overfitting validation
uv run mcbot backtest validate -m quick       # IS/OOS Split
uv run mcbot backtest validate -m milestone   # Walk-Forward (5-fold)
uv run mcbot backtest validate -m final       # CPCV + DSR + PBO

# Signal diagnosis
uv run mcbot backtest diagnose BTC/USDT -s tsmom
```

## EDA Backtest (Event-Driven — Live Parity)

```bash
# EDA backtest (1m → target TF aggregation, single/multi auto-detect)
uv run mcbot eda run config/default.yaml
uv run mcbot eda run config/default.yaml --report         # QuantStats
uv run mcbot eda run config/default.yaml --mode shadow    # Signal logging only
uv run mcbot eda run config/default.yaml --fast           # Fast mode
uv run mcbot eda run config/default.yaml -V               # Verbose
```

## Live Trading

```bash
# Paper mode — WebSocket real-time data + simulated execution
uv run mcbot eda run-live config/paper.yaml --mode paper

# Shadow mode — signal logging only, no execution
uv run mcbot eda run-live config/paper.yaml --mode shadow

# Live mode — Binance USDT-M Futures real orders (Hedge Mode)
# ⚠️ Real funds! Confirmation prompt will appear.
uv run mcbot eda run-live config/paper.yaml --mode live

# Options
# --db-path <path>       SQLite path for state persistence
# -V / --verbose         Verbose output
```

## Pipeline Management

```bash
# Strategy status overview
uv run mcbot pipeline status
uv run mcbot pipeline table             # Full gate progress table
uv run mcbot pipeline report            # Auto-generate dashboard
uv run mcbot pipeline list              # Strategy list with filters
uv run mcbot pipeline show ctrend       # Strategy details

# Strategy lifecycle
uv run mcbot pipeline create            # Create new strategy YAML
uv run mcbot pipeline record            # Record gate result
uv run mcbot pipeline update-status     # Change strategy status

# Lessons management (30 lessons in lessons/*.yaml)
uv run mcbot pipeline lessons-list                      # All lessons
uv run mcbot pipeline lessons-list -c strategy-design   # Category filter
uv run mcbot pipeline lessons-list -t ML                # Tag filter
uv run mcbot pipeline lessons-list -s ctrend            # Strategy filter
uv run mcbot pipeline lessons-list --tf 1H              # Timeframe filter
uv run mcbot pipeline lessons-show 1                    # Lesson details
uv run mcbot pipeline lessons-add --title "제목" --body "설명" -c strategy-design -t tag1
```

## Audit

```bash
# Snapshots
uv run mcbot audit list                         # All snapshots
uv run mcbot audit show 2026-02-13              # Specific snapshot
uv run mcbot audit latest                       # Latest snapshot
uv run mcbot audit trend                        # Metric trends

# Findings
uv run mcbot audit findings                     # All findings
uv run mcbot audit findings --status open       # Open findings
uv run mcbot audit findings --severity critical # Critical findings
uv run mcbot audit finding-show 1               # Finding details
uv run mcbot audit resolve-finding 1            # Resolve finding

# Actions
uv run mcbot audit actions                      # All actions
uv run mcbot audit actions --priority P0        # Urgent actions
uv run mcbot audit action-show 1                # Action details
uv run mcbot audit update-action 1 --status completed  # Complete action

# Create new entries
uv run mcbot audit add-finding                  # Add finding
uv run mcbot audit add-action                   # Add action
uv run mcbot audit create-snapshot              # Create snapshot from YAML
```
