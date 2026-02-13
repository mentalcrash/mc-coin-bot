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
# Bronze layer: raw data from Binance
uv run mcbot ingest bronze BTC/USDT --year 2024 --year 2025

# Silver layer: gap filling
uv run mcbot ingest silver BTC/USDT --year 2024 --year 2025

# Full pipeline: Bronze → Silver
uv run mcbot ingest pipeline BTC/USDT --year 2024 --year 2025

# Validate data integrity
uv run mcbot ingest validate BTC/USDT --year 2025
```

## Backtest

```bash
# List available strategies
uv run mcbot backtest strategies

# Strategy info
uv run mcbot backtest info tsmom

# Run backtest
uv run mcbot backtest run tsmom BTC/USDT --start 2024-01-01 --end 2025-12-31

# Parameter sweep
uv run mcbot backtest sweep tsmom BTC/USDT --start 2024-01-01 --end 2025-12-31

# Generate QuantStats report
uv run mcbot backtest run tsmom BTC/USDT --start 2024-01-01 --end 2025-12-31 --report
```
