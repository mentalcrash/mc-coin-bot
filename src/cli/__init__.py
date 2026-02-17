"""CLI interface using Typer.

Available subcommands:
    - ingest: Data ingestion pipeline (Bronze -> Silver)
    - backtest: Strategy backtesting and optimization
    - eda: EDA (Event-Driven Architecture) backtesting
    - orchestrate: Strategy Orchestrator (multi-strategy)
    - pipeline: Strategy pipeline management (YAML)
    - audit: Architecture audit report system
    - catalog: Data catalog (dataset metadata)

Usage:
    uv run mcbot pipeline table
    uv run mcbot backtest run tsmom BTC/USDT
    uv run mcbot eda run tsmom BTC/USDT
    uv run mcbot orchestrate backtest config/orchestrator-example.yaml
    uv run mcbot ingest bronze BTC/USDT --year 2025
    uv run mcbot audit list
    uv run mcbot catalog list
"""

import typer


def create_app() -> typer.Typer:
    """Create the main CLI application with all sub-commands.

    Lazy import를 사용하여 각 서브커맨드 모듈을 필요할 때만 로드합니다.
    """
    from src.cli.audit import app as audit_app
    from src.cli.backtest import app as backtest_app
    from src.cli.catalog import app as catalog_app
    from src.cli.eda import app as eda_app
    from src.cli.ingest import app as ingest_app
    from src.cli.orchestrate import app as orchestrate_app
    from src.cli.pipeline import app as pipeline_app

    main_app = typer.Typer(
        name="mcbot",
        help="MC Coin Bot - 2026 Crypto Quant Trading System",
        no_args_is_help=True,
    )

    main_app.add_typer(ingest_app, name="ingest", help="Data ingestion pipeline (Bronze/Silver)")
    main_app.add_typer(backtest_app, name="backtest", help="Strategy backtesting (VectorBT)")
    main_app.add_typer(eda_app, name="eda", help="EDA (Event-Driven Architecture) backtesting")
    main_app.add_typer(
        orchestrate_app, name="orchestrate", help="Strategy Orchestrator (multi-strategy)"
    )
    main_app.add_typer(pipeline_app, name="pipeline", help="Strategy pipeline management (YAML)")
    main_app.add_typer(audit_app, name="audit", help="Architecture audit report system")
    main_app.add_typer(catalog_app, name="catalog", help="Data catalog (dataset metadata)")

    return main_app


def main() -> None:
    """Entry point for the ``mcbot`` console script."""
    app = create_app()
    app()
