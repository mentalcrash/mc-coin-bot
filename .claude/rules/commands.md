---
paths:
  - "**"
---

# CLI Commands (Quick Reference)

## Setup

```bash
uv sync --group dev --group research
```

## Quality Gates

```bash
uv run ruff check --fix . && uv run ruff format .
uv run pyright src/
uv run pytest
```

## EDA Run

```bash
uv run mcbot eda run config/default.yaml          # Backtest
uv run mcbot eda run-live config/paper.yaml --mode paper  # Paper trading
uv run mcbot eda spot-live config/spot_supertrend.yaml     # Spot live
```

## Full CLI Reference

```bash
uv run mcbot --help                # Top-level commands
uv run mcbot ingest --help         # Data ingestion (Bronze/Silver/Derivatives/Onchain/Macro/Options/DerivExt)
uv run mcbot backtest --help       # VBT backtest + validation + optimization
uv run mcbot pipeline --help       # Strategy lifecycle + lessons
uv run mcbot catalog --help        # Data catalog (75 datasets)
uv run mcbot audit --help          # Architecture audit
```
