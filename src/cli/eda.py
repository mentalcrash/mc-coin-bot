"""Typer CLI for EDA (Event-Driven Architecture) backtesting.

Commands:
    - run: EDA 백테스트 실행 (단일 심볼, --mode로 backtest/backtest-agg/shadow 선택)
    - run-agg: EDA 1m 집계 백테스트 (하위 호환)
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

from src.config.settings import get_settings
from src.core.exceptions import DataNotFoundError
from src.core.logger import setup_logger
from src.data.market_data import MarketDataRequest, MultiSymbolData
from src.data.service import MarketDataService
from src.eda.runner import EDARunner
from src.portfolio.config import PortfolioManagerConfig
from src.strategy import get_strategy

if TYPE_CHECKING:
    from src.models.backtest import PerformanceMetrics


class RunMode(str, enum.Enum):
    """EDA 실행 모드."""

    BACKTEST = "backtest"
    BACKTEST_AGG = "backtest-agg"
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
    strategy_name: Annotated[str, typer.Argument(help="Strategy name (e.g., tsmom)")],
    symbol: Annotated[str, typer.Argument(help="Trading symbol (e.g., BTC/USDT)")],
    start: Annotated[str, typer.Option(help="Start date (YYYY-MM-DD)")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="End date (YYYY-MM-DD)")] = "2025-12-31",
    capital: Annotated[float, typer.Option(help="Initial capital (USD)")] = 10000.0,
    leverage: Annotated[float, typer.Option(help="Max leverage cap")] = 2.0,
    rebalance: Annotated[float, typer.Option(help="Rebalance threshold")] = 0.05,
    timeframe: Annotated[str, typer.Option(help="Timeframe")] = "1d",
    mode: Annotated[RunMode, typer.Option(help="Execution mode")] = RunMode.BACKTEST,
) -> None:
    """Run EDA backtest for a single symbol.

    Modes:
        - backtest: 기본 백테스트 (--timeframe 해상도)
        - backtest-agg: 1m 데이터 로드 후 --timeframe으로 집계
        - shadow: 시그널 로깅만 (체결 없음)
    """
    setup_logger()
    settings = get_settings()

    try:
        strategy_cls = get_strategy(strategy_name)
        strategy = strategy_cls()
    except (KeyError, ValueError, TypeError) as e:
        console.print(f"[red]Strategy not found: {e}[/red]")
        raise typer.Exit(code=1) from e

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

    config = PortfolioManagerConfig(
        max_leverage_cap=leverage,
        rebalance_threshold=rebalance,
    )

    service = MarketDataService(settings)

    if mode == RunMode.BACKTEST_AGG:
        # 1m 데이터 로드 → target TF로 집계
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

        target_tf = timeframe.upper() if timeframe.lower() == "1d" else timeframe
        runner = EDARunner.backtest_agg(
            strategy=strategy,
            data=data,
            target_timeframe=target_tf,
            config=config,
            initial_capital=capital,
        )
        title = f"EDA Aggregation: {strategy_name} / {symbol} (1m → {target_tf})"
        logger.info(
            "Running EDA aggregation backtest: {} {} {}-{} (1m → {})",
            strategy_name,
            symbol,
            start,
            end,
            target_tf,
        )
    else:
        # backtest / shadow 공통: timeframe 해상도 데이터 로드
        try:
            request = MarketDataRequest(
                symbol=symbol,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt,
            )
            data = service.get(request)
        except DataNotFoundError as e:
            console.print(f"[red]Data not found: {e}[/red]")
            raise typer.Exit(code=1) from e

        if mode == RunMode.SHADOW:
            runner = EDARunner.shadow(
                strategy=strategy,
                data=data,
                config=config,
                initial_capital=capital,
            )
            title = f"EDA Shadow: {strategy_name} / {symbol}"
            logger.info("Running EDA shadow: {} {} {}-{}", strategy_name, symbol, start, end)
        else:
            runner = EDARunner.backtest(
                strategy=strategy,
                data=data,
                config=config,
                initial_capital=capital,
            )
            title = f"EDA Backtest: {strategy_name} / {symbol}"
            logger.info("Running EDA backtest: {} {} {}-{}", strategy_name, symbol, start, end)

    metrics = asyncio.run(runner.run())

    extra_rows = [("Mode", mode.value)] if mode != RunMode.BACKTEST else None
    _display_metrics(title, metrics, extra_rows=extra_rows)


@app.command(name="run-agg")
def run_agg(
    strategy_name: Annotated[str, typer.Argument(help="Strategy name (e.g., tsmom)")],
    symbol: Annotated[str, typer.Argument(help="Trading symbol (e.g., BTC/USDT)")],
    start: Annotated[str, typer.Option(help="Start date (YYYY-MM-DD)")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="End date (YYYY-MM-DD)")] = "2025-12-31",
    capital: Annotated[float, typer.Option(help="Initial capital (USD)")] = 10000.0,
    leverage: Annotated[float, typer.Option(help="Max leverage cap")] = 2.0,
    rebalance: Annotated[float, typer.Option(help="Rebalance threshold")] = 0.05,
    timeframe: Annotated[str, typer.Option(help="Target timeframe for aggregation")] = "1d",
) -> None:
    """Run EDA backtest with 1m aggregation mode.

    Silver 1m 데이터를 로드하여 CandleAggregator로 target TF에 집계합니다.
    라이브 환경과 동일한 데이터 흐름을 백테스트에서 재현합니다.
    """
    setup_logger()
    settings = get_settings()

    try:
        strategy_cls = get_strategy(strategy_name)
        strategy = strategy_cls()
    except (KeyError, ValueError, TypeError) as e:
        console.print(f"[red]Strategy not found: {e}[/red]")
        raise typer.Exit(code=1) from e

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

    # Load 1m data
    try:
        service = MarketDataService(settings)
        request_1m = MarketDataRequest(
            symbol=symbol,
            timeframe="1m",
            start=start_dt,
            end=end_dt,
        )
        data_1m = service.get(request_1m)
    except DataNotFoundError as e:
        console.print(f"[red]1m data not found: {e}[/red]")
        raise typer.Exit(code=1) from e

    config = PortfolioManagerConfig(
        max_leverage_cap=leverage,
        rebalance_threshold=rebalance,
    )

    # Normalize timeframe for aggregation target
    target_tf = timeframe.upper() if timeframe.lower() == "1d" else timeframe

    runner = EDARunner.backtest_agg(
        strategy=strategy,
        data=data_1m,
        target_timeframe=target_tf,
        config=config,
        initial_capital=capital,
    )

    logger.info(
        "Running EDA aggregation backtest: {} {} {}-{} (1m → {})",
        strategy_name,
        symbol,
        start,
        end,
        target_tf,
    )
    metrics = asyncio.run(runner.run())

    _display_metrics(f"EDA Aggregation: {strategy_name} / {symbol} (1m → {target_tf})", metrics)


_DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,DOGE/USDT,LINK/USDT,ADA/USDT,AVAX/USDT"


@app.command(name="run-multi")
def run_multi(
    strategy_name: Annotated[str, typer.Argument(help="Strategy name (e.g., tsmom)")],
    symbols: Annotated[str, typer.Option(help="Comma-separated symbols")] = _DEFAULT_SYMBOLS,
    start: Annotated[str, typer.Option(help="Start date (YYYY-MM-DD)")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="End date (YYYY-MM-DD)")] = "2025-12-31",
    capital: Annotated[float, typer.Option(help="Initial capital (USD)")] = 100000.0,
    leverage: Annotated[float, typer.Option(help="Max leverage cap")] = 2.0,
    rebalance: Annotated[float, typer.Option(help="Rebalance threshold")] = 0.05,
    timeframe: Annotated[str, typer.Option(help="Timeframe")] = "1d",
) -> None:
    """Run EDA backtest for multiple symbols (equal-weight portfolio)."""
    setup_logger()
    settings = get_settings()

    symbol_list = [s.strip() for s in symbols.split(",")]
    _min_symbols = 2
    if len(symbol_list) < _min_symbols:
        console.print("[red]At least 2 symbols required.[/red]")
        raise typer.Exit(code=1)

    try:
        strategy_cls = get_strategy(strategy_name)
        strategy = strategy_cls()
    except (KeyError, ValueError, TypeError) as e:
        console.print(f"[red]Strategy not found: {e}[/red]")
        raise typer.Exit(code=1) from e

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

    # Load data for each symbol
    service = MarketDataService(settings)
    ohlcv_dict: dict[str, object] = {}
    loaded_symbols: list[str] = []

    for sym in symbol_list:
        try:
            request = MarketDataRequest(
                symbol=sym,
                timeframe=timeframe,
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

    # Construct MultiSymbolData
    multi_data = MultiSymbolData(
        symbols=loaded_symbols,
        timeframe=timeframe,
        start=start_dt,
        end=end_dt,
        ohlcv=ohlcv_dict,  # type: ignore[arg-type]
    )

    # Equal weights
    weights = {s: 1.0 / len(loaded_symbols) for s in loaded_symbols}

    config = PortfolioManagerConfig(
        max_leverage_cap=leverage,
        rebalance_threshold=rebalance,
    )

    runner = EDARunner.backtest(
        strategy=strategy,
        data=multi_data,
        config=config,
        initial_capital=capital,
        asset_weights=weights,
    )

    logger.info(
        "Running EDA multi-asset backtest: {} {} symbols {}-{}",
        strategy_name,
        len(loaded_symbols),
        start,
        end,
    )
    metrics = asyncio.run(runner.run())

    _display_metrics(
        f"EDA Multi-Asset: {strategy_name} / {len(loaded_symbols)} symbols",
        metrics,
        extra_rows=[("Symbols", ", ".join(loaded_symbols))],
    )
