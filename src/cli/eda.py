"""Typer CLI for EDA (Event-Driven Architecture) backtesting and live trading.

Commands:
    - run: EDA 백테스트 실행 (config의 symbols 개수로 단일/멀티 자동 판별)
    - run-live: 실시간 WebSocket 데이터로 EDA 실행 (shadow/paper 모드)

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
from src.data.market_data import MarketDataRequest, MarketDataSet, MultiSymbolData
from src.data.service import MarketDataService
from src.eda.runner import EDARunner

if TYPE_CHECKING:
    import pandas as pd

    from src.config.config_loader import RunConfig
    from src.models.backtest import PerformanceMetrics


class RunMode(str, enum.Enum):
    """EDA 백테스트 실행 모드."""

    BACKTEST = "backtest"
    SHADOW = "shadow"


class LiveRunMode(str, enum.Enum):
    """EDA 라이브 실행 모드."""

    PAPER = "paper"
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
    report: Annotated[
        bool, typer.Option("--report/--no-report", help="Generate QuantStats HTML report")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-V", help="Enable verbose output")] = False,
) -> None:
    """Run EDA backtest from config file.

    config의 symbols 개수로 단일/멀티에셋을 자동 판별합니다.
    항상 1m 데이터를 로드하여 config의 timeframe으로 집계합니다.

    Modes:
        - backtest: 기본 백테스트
        - shadow: 시그널 로깅만 (체결 없음)
    """
    setup_logger(console_level="DEBUG" if verbose else "WARNING")

    cfg = load_config(config_path)
    settings = get_settings()

    strategy = build_strategy(cfg)
    symbol_list = cfg.backtest.symbols

    start_dt = datetime.strptime(cfg.backtest.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(cfg.backtest.end, "%Y-%m-%d").replace(tzinfo=UTC)

    tf = cfg.backtest.timeframe
    target_tf = tf.upper() if tf.lower() == "1d" else tf

    service = MarketDataService(settings)
    is_multi = len(symbol_list) > 1

    if is_multi:
        data, loaded_symbols = _load_multi_symbol_data(service, symbol_list, start_dt, end_dt)
        weights = {s: 1.0 / len(loaded_symbols) for s in loaded_symbols}
        asset_weights: dict[str, float] | None = weights
        label = f"{len(loaded_symbols)} symbols"
    else:
        data = _load_single_symbol_data(service, symbol_list[0], start_dt, end_dt)
        loaded_symbols = symbol_list
        asset_weights = None
        label = symbol_list[0]

    factory = EDARunner.shadow if mode == RunMode.SHADOW else EDARunner.backtest
    runner = factory(
        strategy=strategy,
        data=data,
        target_timeframe=target_tf,
        config=cfg.portfolio,
        initial_capital=cfg.backtest.capital,
        asset_weights=asset_weights,
    )

    mode_label = "Shadow" if mode == RunMode.SHADOW else "Backtest"
    title = f"EDA {mode_label}: {cfg.strategy.name} / {label} (1m → {target_tf})"

    logger.info("Running EDA {}: {} {} (1m → {})", mode.value, cfg.strategy.name, label, target_tf)
    metrics = asyncio.run(runner.run())

    extra_rows: list[tuple[str, str]] = []
    if mode != RunMode.BACKTEST:
        extra_rows.append(("Mode", mode.value))
    if is_multi:
        extra_rows.append(("Symbols", ", ".join(loaded_symbols)))
    _display_metrics(title, metrics, extra_rows=extra_rows or None)

    if report:
        _generate_report_for_run(
            runner=runner,
            cfg=cfg,
            is_multi=is_multi,
            loaded_symbols=loaded_symbols,
            target_tf=target_tf,
            data=data,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_single_symbol_data(
    service: MarketDataService,
    symbol: str,
    start: datetime,
    end: datetime,
) -> MarketDataSet:
    """단일 심볼 1m 데이터 로드."""
    try:
        request = MarketDataRequest(symbol=symbol, timeframe="1m", start=start, end=end)
        return service.get(request)
    except DataNotFoundError as e:
        console.print(f"[red]1m data not found: {e}[/red]")
        raise typer.Exit(code=1) from e


def _load_multi_symbol_data(
    service: MarketDataService,
    symbol_list: list[str],
    start: datetime,
    end: datetime,
) -> tuple[MultiSymbolData, list[str]]:
    """멀티 심볼 1m 데이터 로드. (MultiSymbolData, loaded_symbols) 반환."""
    ohlcv_dict: dict[str, object] = {}
    loaded_symbols: list[str] = []

    for sym in symbol_list:
        try:
            request = MarketDataRequest(symbol=sym, timeframe="1m", start=start, end=end)
            data = service.get(request)
            ohlcv_dict[sym] = data.ohlcv
            loaded_symbols.append(sym)
            logger.info("Loaded {} bars for {}", len(data.ohlcv), sym)
        except DataNotFoundError:
            console.print(f"[yellow]Warning: Data not found for {sym}, skipping.[/yellow]")

    _min_symbols = 2
    if len(loaded_symbols) < _min_symbols:
        console.print("[red]Need at least 2 symbols with data.[/red]")
        raise typer.Exit(code=1)

    multi_data = MultiSymbolData(
        symbols=loaded_symbols,
        timeframe="1m",
        start=start,
        end=end,
        ohlcv=ohlcv_dict,  # type: ignore[arg-type]
    )
    return multi_data, loaded_symbols


def _generate_report_for_run(
    runner: EDARunner,
    cfg: RunConfig,
    is_multi: bool,
    loaded_symbols: list[str],
    target_tf: str,
    data: MarketDataSet | MultiSymbolData,
) -> None:
    """run 커맨드의 --report 처리."""
    freq = _tf_to_pandas_freq(target_tf)

    if is_multi:
        import pandas as pd

        sym_returns: list[pd.Series] = []
        for sym in loaded_symbols:
            close_1m = data.ohlcv[sym]["close"]  # type: ignore[index]
            close_tf: pd.Series = close_1m.resample(freq).last().dropna()  # type: ignore[assignment]
            sym_returns.append(close_tf.pct_change().dropna())
        benchmark_returns: pd.Series = pd.concat(sym_returns, axis=1).mean(axis=1)  # type: ignore[assignment]
        title = f"EDA Multi-Asset: {cfg.strategy.name} / {len(loaded_symbols)} symbols ({cfg.backtest.start} ~ {cfg.backtest.end})"
    else:
        symbol = loaded_symbols[0]
        close_tf_s = data.ohlcv["close"].resample(freq).last().dropna()  # type: ignore[union-attr]
        benchmark_returns = close_tf_s.pct_change().dropna()
        title = f"EDA {cfg.strategy.name} - {symbol} ({cfg.backtest.start} ~ {cfg.backtest.end})"

    _generate_eda_report(runner=runner, benchmark_returns=benchmark_returns, title=title)


def _tf_to_pandas_freq(tf: str) -> str:
    """Target timeframe → pandas resample frequency."""
    from src.eda.analytics import tf_to_pandas_freq

    return tf_to_pandas_freq(tf)


def _generate_eda_report(
    runner: EDARunner,
    benchmark_returns: pd.Series,
    title: str,
) -> None:
    """EDA 백테스트 결과를 QuantStats HTML 리포트로 생성.

    Args:
        runner: EDA runner (config, target_timeframe, analytics 포함)
        benchmark_returns: 벤치마크 수익률 (target TF로 리샘플링 완료)
        title: 리포트 제목
    """
    from src.backtest.reporter import generate_quantstats_report

    analytics = runner.analytics
    if analytics is None:
        console.print("[yellow]Analytics not available — skipping report.[/yellow]")
        return

    # compute_metrics()와 동일한 returns 사용 (리샘플링 + funding drag)
    strategy_returns = analytics.get_strategy_returns(
        timeframe=runner.target_timeframe,
        cost_model=runner.config.cost_model,
    )
    _min_points = 2
    if len(strategy_returns) < _min_points:
        console.print("[yellow]Equity curve too short — skipping report.[/yellow]")
        return

    # 벤치마크와 인덱스 정렬 (공통 날짜만)
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < _min_points:
        console.print("[yellow]Not enough overlapping dates for report — skipping.[/yellow]")
        return
    strategy_returns = strategy_returns.loc[common_idx]
    benchmark_aligned = benchmark_returns.loc[common_idx]

    report_path = generate_quantstats_report(
        returns=strategy_returns,
        benchmark_returns=benchmark_aligned,
        title=title,
    )
    console.print(f"[green]Report saved: {report_path}[/green]")


# ---------------------------------------------------------------------------
# run-live command
# ---------------------------------------------------------------------------


@app.command("run-live")
def run_live(
    config_path: Annotated[str, typer.Argument(help="YAML config file path")],
    mode: Annotated[LiveRunMode, typer.Option(help="Execution mode")] = LiveRunMode.PAPER,
    verbose: Annotated[bool, typer.Option("--verbose", "-V", help="Enable verbose output")] = False,
) -> None:
    """Run EDA in live mode (WebSocket real-time data).

    Modes:
        - paper: 시뮬레이션 체결 (BacktestExecutor)
        - shadow: 시그널 로깅만 (ShadowExecutor, 체결 없음)
    """
    from src.eda.live_runner import LiveMode, LiveRunner
    from src.exchange.binance_client import BinanceClient

    setup_logger(console_level="DEBUG" if verbose else "INFO")

    cfg = load_config(config_path)
    strategy = build_strategy(cfg)
    symbol_list = cfg.backtest.symbols

    tf = cfg.backtest.timeframe
    target_tf = tf.upper() if tf.lower() == "1d" else tf

    is_multi = len(symbol_list) > 1
    asset_weights: dict[str, float] | None = None
    if is_multi:
        asset_weights = {s: 1.0 / len(symbol_list) for s in symbol_list}

    live_mode = LiveMode.SHADOW if mode == LiveRunMode.SHADOW else LiveMode.PAPER
    mode_label = "Shadow" if live_mode == LiveMode.SHADOW else "Paper"

    header = f"[bold cyan]EDA Live {mode_label}: {cfg.strategy.name} / {len(symbol_list)} symbols (1m → {target_tf})[/bold cyan]"
    console.print(header)
    console.print(f"[dim]Symbols: {', '.join(symbol_list)}[/dim]")
    console.print("[dim]Press Ctrl+C to stop gracefully.[/dim]")

    async def _run() -> None:
        async with BinanceClient() as client:
            factory = LiveRunner.shadow if live_mode == LiveMode.SHADOW else LiveRunner.paper
            runner = factory(
                strategy=strategy,
                symbols=symbol_list,
                target_timeframe=target_tf,
                config=cfg.portfolio,
                client=client,
                initial_capital=cfg.backtest.capital,
                asset_weights=asset_weights,
            )
            await runner.run()

    asyncio.run(_run())
