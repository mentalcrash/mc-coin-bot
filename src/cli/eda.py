"""Typer CLI for EDA (Event-Driven Architecture) backtesting.

Commands:
    - run: EDA 백테스트 실행 (단일 심볼, --mode로 backtest/shadow 선택)
    - run-multi: EDA 멀티에셋 백테스트

Rules Applied:
    - #18 Typer CLI: Annotated syntax, Rich UI, async handling
    - #15 Logging Standards: Loguru structured logging
"""

from __future__ import annotations

import asyncio
import enum
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config.config_loader import build_strategy, load_config
from src.config.settings import get_settings
from src.core.exceptions import DataNotFoundError
from src.core.logger import setup_logger
from src.data.market_data import MarketDataRequest, MultiSymbolData
from src.data.service import MarketDataService
from src.eda.runner import EDARunner

if TYPE_CHECKING:
    from src.models.backtest import PerformanceMetrics


class RunMode(str, enum.Enum):
    """EDA 실행 모드."""

    BACKTEST = "backtest"
    SHADOW = "shadow"


console = Console()
app = typer.Typer(help="EDA (Event-Driven Architecture) backtesting")


def _display_metrics(
    title: str,
    metrics: PerformanceMetrics,
    extra_rows: list[tuple[str, str]] | None = None,
) -> None:
    """Rich Table로 PerformanceMetrics를 출력한다."""
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    if extra_rows:
        for label, value in extra_rows:
            table.add_row(label, value)

    table.add_row("Total Return", f"{metrics.total_return:.2f}%")
    table.add_row("CAGR", f"{metrics.cagr:.2f}%")
    table.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.4f}")
    table.add_row("Max Drawdown", f"{metrics.max_drawdown:.2f}%")
    table.add_row("Win Rate", f"{metrics.win_rate:.1f}%")
    table.add_row("Total Trades", str(metrics.total_trades))
    table.add_row("Winning Trades", str(metrics.winning_trades))
    table.add_row("Losing Trades", str(metrics.losing_trades))

    if metrics.volatility is not None:
        table.add_row("Volatility", f"{metrics.volatility:.2f}%")
    if metrics.profit_factor is not None:
        table.add_row("Profit Factor", f"{metrics.profit_factor:.4f}")

    console.print(table)


@app.command()
def run(
    config_path: Annotated[str, typer.Argument(help="YAML config file path")],
    mode: Annotated[RunMode, typer.Option(help="Execution mode")] = RunMode.BACKTEST,
    verbose: Annotated[bool, typer.Option("--verbose", "-V", help="Enable verbose output")] = False,
) -> None:
    """Run EDA backtest for a single symbol from config file.

    항상 1m 데이터를 로드하여 config의 timeframe으로 집계합니다.

    Modes:
        - backtest: 기본 백테스트
        - shadow: 시그널 로깅만 (체결 없음)
    """
    setup_logger(console_level="DEBUG" if verbose else "WARNING")

    cfg = load_config(config_path)
    settings = get_settings()

    strategy = build_strategy(cfg)
    symbol = cfg.backtest.symbols[0]

    start_dt = datetime.strptime(cfg.backtest.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(cfg.backtest.end, "%Y-%m-%d").replace(tzinfo=UTC)

    service = MarketDataService(settings)

    # 항상 1m 데이터 로드
    try:
        request_1m = MarketDataRequest(
            symbol=symbol,
            timeframe="1m",
            start=start_dt,
            end=end_dt,
        )
        data = service.get(request_1m)
    except DataNotFoundError as e:
        console.print(f"[red]1m data not found: {e}[/red]")
        raise typer.Exit(code=1) from e

    tf = cfg.backtest.timeframe
    target_tf = tf.upper() if tf.lower() == "1d" else tf

    if mode == RunMode.SHADOW:
        runner = EDARunner.shadow(
            strategy=strategy,
            data=data,
            target_timeframe=target_tf,
            config=cfg.portfolio,
            initial_capital=cfg.backtest.capital,
        )
        title = f"EDA Shadow: {cfg.strategy.name} / {symbol} (1m → {target_tf})"
    else:
        runner = EDARunner.backtest(
            strategy=strategy,
            data=data,
            target_timeframe=target_tf,
            config=cfg.portfolio,
            initial_capital=cfg.backtest.capital,
        )
        title = f"EDA Backtest: {cfg.strategy.name} / {symbol} (1m → {target_tf})"

    logger.info("Running EDA {}: {} {} (1m → {})", mode.value, cfg.strategy.name, symbol, target_tf)
    metrics = asyncio.run(runner.run())

    extra_rows = [("Mode", mode.value)] if mode != RunMode.BACKTEST else None
    _display_metrics(title, metrics, extra_rows=extra_rows)


@app.command(name="run-multi")
def run_multi(
    config_path: Annotated[str, typer.Argument(help="YAML config file path")],
    verbose: Annotated[bool, typer.Option("--verbose", "-V", help="Enable verbose output")] = False,
) -> None:
    """Run EDA backtest for multiple symbols (equal-weight portfolio) from config file."""
    setup_logger(console_level="DEBUG" if verbose else "WARNING")

    cfg = load_config(config_path)
    settings = get_settings()

    symbol_list = cfg.backtest.symbols
    _min_symbols = 2
    if len(symbol_list) < _min_symbols:
        console.print("[red]At least 2 symbols required in config.[/red]")
        raise typer.Exit(code=1)

    strategy = build_strategy(cfg)

    start_dt = datetime.strptime(cfg.backtest.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(cfg.backtest.end, "%Y-%m-%d").replace(tzinfo=UTC)

    # Load 1m data for each symbol
    service = MarketDataService(settings)
    ohlcv_dict: dict[str, object] = {}
    loaded_symbols: list[str] = []

    for sym in symbol_list:
        try:
            request = MarketDataRequest(
                symbol=sym,
                timeframe="1m",
                start=start_dt,
                end=end_dt,
            )
            data = service.get(request)
            ohlcv_dict[sym] = data.ohlcv
            loaded_symbols.append(sym)
            logger.info("Loaded {} bars for {}", len(data.ohlcv), sym)
        except DataNotFoundError:
            console.print(f"[yellow]Warning: Data not found for {sym}, skipping.[/yellow]")

    if len(loaded_symbols) < _min_symbols:
        console.print("[red]Need at least 2 symbols with data.[/red]")
        raise typer.Exit(code=1)

    tf = cfg.backtest.timeframe
    target_tf = tf.upper() if tf.lower() == "1d" else tf

    # Construct MultiSymbolData
    multi_data = MultiSymbolData(
        symbols=loaded_symbols,
        timeframe="1m",
        start=start_dt,
        end=end_dt,
        ohlcv=ohlcv_dict,  # type: ignore[arg-type]
    )

    # Equal weights
    weights = {s: 1.0 / len(loaded_symbols) for s in loaded_symbols}

    runner = EDARunner.backtest(
        strategy=strategy,
        data=multi_data,
        target_timeframe=target_tf,
        config=cfg.portfolio,
        initial_capital=cfg.backtest.capital,
        asset_weights=weights,
    )

    logger.info(
        "Running EDA multi-asset backtest: {} {} symbols (1m → {})",
        cfg.strategy.name,
        len(loaded_symbols),
        target_tf,
    )
    metrics = asyncio.run(runner.run())

    _display_metrics(
        f"EDA Multi-Asset: {cfg.strategy.name} / {len(loaded_symbols)} symbols",
        metrics,
        extra_rows=[("Symbols", ", ".join(loaded_symbols))],
    )
