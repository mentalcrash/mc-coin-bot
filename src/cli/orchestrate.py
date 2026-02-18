"""Typer CLI for Strategy Orchestrator (multi-strategy).

Commands:
    - backtest: Orchestrator 멀티전략 백테스트 실행
    - paper: Paper 모드 실시간 실행
    - live: Live 모드 실주문 실행

Rules Applied:
    - #18 Typer CLI: Annotated syntax, Rich UI, async handling
    - #15 Logging Standards: Loguru structured logging
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config.orchestrator_loader import load_orchestrator_config
from src.config.settings import get_settings
from src.core.exceptions import DataNotFoundError
from src.core.logger import setup_logger
from src.data.market_data import MarketDataRequest, MultiSymbolData
from src.data.service import MarketDataService

if TYPE_CHECKING:
    from src.config.orchestrator_loader import OrchestratorRunConfig
    from src.orchestrator.config import OrchestratorConfig
    from src.orchestrator.result import OrchestratedResult

console = Console()
app = typer.Typer(help="Strategy Orchestrator (multi-strategy)")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_portfolio_metrics(title: str, result: OrchestratedResult) -> None:
    """포트폴리오 전체 메트릭을 Rich Table로 출력."""
    metrics = result.portfolio_metrics
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

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


def _display_pod_summary(result: OrchestratedResult) -> None:
    """Pod별 요약 테이블 출력."""
    if not result.pod_metrics:
        return

    table = Table(title="Pod Summary")
    table.add_column("Pod ID", style="cyan")
    table.add_column("Return", style="green", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("MDD", style="red", justify="right")
    table.add_column("Days", justify="right")

    for pod_id, metrics in result.pod_metrics.items():
        table.add_row(
            pod_id,
            f"{metrics.get('total_return', 0) * 100:.2f}%",
            f"{metrics.get('sharpe', 0):.4f}",
            f"{metrics.get('mdd', 0) * 100:.2f}%",
            str(metrics.get("n_days", 0)),
        )

    console.print(table)


def _display_pod_config_table(orch_config: OrchestratorConfig) -> None:
    """Live 시작 전 Pod 설정 테이블 출력."""
    table = Table(title="Pod Configuration")
    table.add_column("Pod ID", style="cyan")
    table.add_column("Strategy", style="green")
    table.add_column("TF")
    table.add_column("Symbols")
    table.add_column("Fraction", justify="right")
    table.add_column("Max DD", style="red", justify="right")

    for pod in orch_config.pods:
        table.add_row(
            pod.pod_id,
            pod.strategy_name,
            pod.timeframe,
            ", ".join(pod.symbols),
            f"{pod.initial_fraction:.0%}",
            f"{pod.max_drawdown:.0%}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_orchestrator_data(
    service: MarketDataService,
    symbols: tuple[str, ...],
    start: datetime,
    end: datetime,
) -> tuple[MultiSymbolData, list[str]]:
    """Orchestrator용 멀티 심볼 1m 데이터 로드.

    Args:
        service: MarketDataService
        symbols: 심볼 튜플
        start: 시작 시각
        end: 종료 시각

    Returns:
        (MultiSymbolData, loaded_symbols) 튜플

    Raises:
        typer.Exit: 데이터 로드 실패 시
    """
    ohlcv_dict: dict[str, object] = {}
    loaded_symbols: list[str] = []

    for sym in symbols:
        try:
            request = MarketDataRequest(symbol=sym, timeframe="1m", start=start, end=end)
            data = service.get(request)
            ohlcv_dict[sym] = data.ohlcv
            loaded_symbols.append(sym)
            logger.info("Loaded {} bars for {}", len(data.ohlcv), sym)
        except DataNotFoundError:
            console.print(f"[yellow]Warning: Data not found for {sym}, skipping.[/yellow]")

    if not loaded_symbols:
        console.print("[red]No data loaded for any symbol.[/red]")
        raise typer.Exit(code=1)

    multi_data = MultiSymbolData(
        symbols=loaded_symbols,
        timeframe="1m",
        start=start,
        end=end,
        ohlcv=ohlcv_dict,  # type: ignore[arg-type]
    )
    return multi_data, loaded_symbols


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------


def _generate_orchestrated_report(
    result: OrchestratedResult,
    cfg: OrchestratorRunConfig,
    symbols: list[str],
    target_tf: str,
) -> None:
    """QuantStats HTML 리포트 생성."""
    from src.backtest.reporter import generate_quantstats_report

    equity = result.portfolio_equity_curve
    _min_points = 2
    if len(equity) < _min_points:
        console.print("[yellow]Equity curve too short — skipping report.[/yellow]")
        return

    returns = equity.pct_change().dropna()
    if len(returns) < _min_points:
        console.print("[yellow]Not enough data for report — skipping.[/yellow]")
        return

    title = (
        f"Orchestrator: {len(cfg.orchestrator.pods)} pods / "
        f"{len(symbols)} symbols ({cfg.backtest.start} ~ {cfg.backtest.end})"
    )

    report_path = generate_quantstats_report(
        returns=returns,
        benchmark_returns=None,
        title=title,
    )
    console.print(f"[green]Report saved: {report_path}[/green]")


# ---------------------------------------------------------------------------
# Live launcher (shared)
# ---------------------------------------------------------------------------


def launch_orchestrated_live(
    config_path: str,
    *,
    mode: str = "paper",
    db_path: str | None = "data/trading.db",
) -> None:
    """Orchestrated LiveRunner 실행 공통 함수.

    CLI ``paper``/``live`` 커맨드와 Docker ENTRYPOINT 모두
    이 함수를 호출하여 중복을 제거합니다.

    Args:
        config_path: YAML config 파일 경로.
        mode: 실행 모드 (``"paper"`` | ``"live"``).
        db_path: SQLite 경로. ``None``이면 영속화 비활성.
    """
    from src.eda.live_runner import LiveRunner
    from src.exchange.binance_client import BinanceClient
    from src.notification.config import DiscordBotConfig

    run_cfg = load_orchestrator_config(config_path)
    orch_config = run_cfg.orchestrator

    tf = run_cfg.backtest.timeframe
    target_tf = tf.upper() if tf.lower() == "1d" else tf
    symbols = list(orch_config.all_symbols)
    capital = run_cfg.backtest.capital

    # Discord config
    discord_config = None
    dc = DiscordBotConfig()
    if dc.is_bot_configured:
        discord_config = dc
        console.print("[dim]Discord Bot enabled[/dim]")

    mode_label = "Paper" if mode == "paper" else "LIVE"
    header = (
        f"[bold cyan]Orchestrator {mode_label}: "
        f"{orch_config.n_pods} pods / {len(symbols)} symbols "
        f"(1m → {target_tf})[/bold cyan]"
    )
    console.print(header)
    console.print(f"[dim]Symbols: {', '.join(symbols)}[/dim]")
    console.print("[dim]Press Ctrl+C to stop gracefully.[/dim]")

    logger.info(
        "Starting Orchestrated LiveRunner: mode={}, pods={}, symbols={}, tf=1m→{}",
        mode,
        orch_config.n_pods,
        len(symbols),
        target_tf,
    )

    async def _run() -> None:
        from src.config.settings import get_deployment_config

        deploy_cfg = get_deployment_config()
        async with BinanceClient() as client:
            if mode == "live":
                from src.exchange.binance_futures_client import BinanceFuturesClient

                async with BinanceFuturesClient() as futures_client:
                    await futures_client.setup_account(
                        symbols,
                        leverage=round(orch_config.max_gross_leverage),
                    )
                    runner = LiveRunner.orchestrated_live(
                        orchestrator_config=orch_config,
                        target_timeframe=target_tf,
                        client=client,
                        futures_client=futures_client,
                        initial_capital=capital,
                        db_path=db_path,
                        discord_config=discord_config,
                        metrics_port=deploy_cfg.metrics_port,
                    )
                    await runner.run()
            else:
                runner = LiveRunner.orchestrated_paper(
                    orchestrator_config=orch_config,
                    target_timeframe=target_tf,
                    client=client,
                    initial_capital=capital,
                    db_path=db_path,
                    discord_config=discord_config,
                    metrics_port=deploy_cfg.metrics_port,
                )
                await runner.run()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def backtest(
    config_path: Annotated[str, typer.Argument(help="YAML config file path")],
    report: Annotated[
        bool, typer.Option("--report/--no-report", help="Generate QuantStats HTML report")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-V", help="Enable verbose output")] = False,
) -> None:
    """Run Orchestrator backtest from config file.

    멀티 전략 포트폴리오 백테스트를 실행합니다.
    각 Pod의 전략/심볼/자본비중은 YAML에서 설정합니다.
    """
    setup_logger(console_level="DEBUG" if verbose else "WARNING")

    run_cfg = load_orchestrator_config(config_path)
    orch_config = run_cfg.orchestrator
    settings = get_settings()

    tf = run_cfg.backtest.timeframe
    target_tf = tf.upper() if tf.lower() == "1d" else tf
    all_symbols = orch_config.all_symbols

    start_dt = datetime.strptime(run_cfg.backtest.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(run_cfg.backtest.end, "%Y-%m-%d").replace(tzinfo=UTC)

    service = MarketDataService(settings)
    data, loaded_symbols = _load_orchestrator_data(service, all_symbols, start_dt, end_dt)

    title = (
        f"Orchestrator Backtest: {orch_config.n_pods} pods / "
        f"{len(loaded_symbols)} symbols (1m → {target_tf})"
    )
    all_tfs = orch_config.all_timeframes
    if len(all_tfs) > 1:
        logger.info("Multi-TF mode: {}", ", ".join(all_tfs))

    logger.info(
        "Running Orchestrator backtest: {} pods, {} symbols, TF={}",
        orch_config.n_pods,
        len(loaded_symbols),
        ", ".join(all_tfs) if len(all_tfs) > 1 else target_tf,
    )

    from src.eda.orchestrated_runner import OrchestratedRunner

    runner = OrchestratedRunner.backtest(
        orchestrator_config=orch_config,
        data=data,
        target_timeframe=target_tf,
        initial_capital=run_cfg.backtest.capital,
    )
    result = asyncio.run(runner.run())

    _display_portfolio_metrics(title, result)
    _display_pod_summary(result)

    if report:
        _generate_orchestrated_report(result, run_cfg, loaded_symbols, target_tf)


@app.command()
def paper(
    config_path: Annotated[str, typer.Argument(help="YAML config file path")],
    db_path: Annotated[
        str | None,
        typer.Option("--db-path", help="SQLite DB path for persistence (None=disabled)"),
    ] = "data/trading.db",
    verbose: Annotated[bool, typer.Option("--verbose", "-V", help="Enable verbose output")] = False,
) -> None:
    """Run Orchestrator in paper mode (WebSocket + simulated execution)."""
    setup_logger(console_level="DEBUG" if verbose else "INFO")
    launch_orchestrated_live(config_path, mode="paper", db_path=db_path)


@app.command()
def live(
    config_path: Annotated[str, typer.Argument(help="YAML config file path")],
    db_path: Annotated[
        str | None,
        typer.Option("--db-path", help="SQLite DB path for persistence (None=disabled)"),
    ] = "data/trading.db",
    verbose: Annotated[bool, typer.Option("--verbose", "-V", help="Enable verbose output")] = False,
) -> None:
    """Run Orchestrator in LIVE mode (real funds on Binance Futures).

    WARNING: 실제 자금으로 거래합니다. 확인 프롬프트가 표시됩니다.
    """
    setup_logger(console_level="DEBUG" if verbose else "INFO")

    run_cfg = load_orchestrator_config(config_path)
    orch_config = run_cfg.orchestrator

    # Config 요약 테이블
    summary_table = Table(title="LIVE Mode Configuration", show_header=False)
    summary_table.add_column("Key", style="cyan")
    summary_table.add_column("Value", style="bold red")
    summary_table.add_row("Pods", str(orch_config.n_pods))
    summary_table.add_row("Symbols", ", ".join(orch_config.all_symbols))
    summary_table.add_row("Capital", f"${run_cfg.backtest.capital:,.0f}")
    summary_table.add_row("Max Leverage", f"{orch_config.max_gross_leverage:.1f}x")
    summary_table.add_row("Daily Loss Limit", f"{orch_config.daily_loss_limit:.0%}")
    summary_table.add_row("Allocation", orch_config.allocation_method.value)
    console.print(summary_table)

    # Pod config 테이블
    _display_pod_config_table(orch_config)

    confirm = typer.confirm(
        "WARNING: LIVE mode trades with real funds on Binance Futures. Continue?"
    )
    if not confirm:
        console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit(code=0)

    launch_orchestrated_live(config_path, mode="live", db_path=db_path)
