"""Typer CLI for backtesting strategies.

이 모듈은 VW-TSMOM 전략 백테스팅을 위한 CLI를 제공합니다.
Clean Architecture에 따라 MarketDataService, BacktestRequest를 사용합니다.

Commands:
    - run: 단일 백테스트 실행
    - optimize: 파라미터 최적화 실행
    - info: 전략 정보 출력

Rules Applied:
    - #15 Logging Standards: Loguru, structured logging
    - #18 Typer CLI: Annotated syntax, Rich UI, async handling
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.backtest.analyzer import PerformanceAnalyzer
from src.backtest.beta_attribution import (
    calculate_beta_attribution,
    summarize_suppression_impact,
)
from src.backtest.engine import BacktestEngine, run_parameter_sweep
from src.backtest.reporter import generate_quantstats_report, print_performance_summary
from src.backtest.request import BacktestRequest
from src.config.settings import get_settings
from src.core.exceptions import DataNotFoundError
from src.core.logger import setup_logger
from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.logging.context import get_strategy_logger
from src.portfolio import Portfolio
from src.strategy.tsmom import TSMOMConfig, TSMOMStrategy
from src.strategy.tsmom.signal import generate_signals_with_diagnostics

# Global Console Instance (Rich UI for user-facing output)
console = Console()

# Beta Attribution Thresholds
BETA_LOSS_THRESHOLD_LOW = 0.1
BETA_LOSS_THRESHOLD_HIGH = 0.2
BETA_RETENTION_GOOD = 0.7


def _print_startup_panel(
    symbol: str,
    years: list[int],
    capital: float,
    strategy: TSMOMStrategy,
    portfolio: Portfolio,
) -> None:
    """백테스트 시작 정보 패널 출력.

    전략과 포트폴리오의 핵심 설정값을 사용자에게 표시합니다.

    Args:
        symbol: 거래 심볼
        years: 백테스트 연도 목록
        capital: 초기 자본금
        strategy: 전략 인스턴스
        portfolio: 포트폴리오 인스턴스
    """
    cfg = strategy.config
    pm_cfg = portfolio.config

    # 전략 설정 요약
    strategy_info = (
        f"  lookback: {cfg.lookback}일, "
        f"vol_target: {cfg.vol_target:.0%}, "
        f"vol_window: {cfg.vol_window}일"
    )
    if cfg.use_trend_filter:
        strategy_info += f"\n  trend_filter: MA({cfg.trend_ma_period}), "
        strategy_info += f"deadband: {cfg.deadband_threshold}"

    # 포트폴리오 설정 요약
    stop_loss_str = (
        f"{pm_cfg.system_stop_loss:.0%}" if pm_cfg.system_stop_loss else "Disabled"
    )
    portfolio_info = (
        f"  max_leverage: {pm_cfg.max_leverage_cap}x, "
        f"stop_loss: {stop_loss_str}, "
        f"rebalance: {pm_cfg.rebalance_threshold:.0%}"
    )
    portfolio_info += f"\n  execution: {pm_cfg.execution_mode}, "
    portfolio_info += f"cost: {pm_cfg.cost_model.round_trip_cost:.2%} RT"

    panel_content = (
        f"[bold]VW-TSMOM Backtest[/bold]\n"
        f"Symbol: {symbol}\n"
        f"Years: {', '.join(map(str, years))}\n"
        f"Capital: ${capital:,.0f}\n\n"
        f"[bold cyan]Strategy (TSMOMConfig)[/bold cyan]\n"
        f"{strategy_info}\n\n"
        f"[bold cyan]Portfolio[/bold cyan]\n"
        f"{portfolio_info}"
    )

    console.print(Panel.fit(panel_content, border_style="blue"))


# Typer App
app = typer.Typer(
    name="backtest",
    help="VW-TSMOM Strategy Backtesting CLI",
    no_args_is_help=True,
)


@app.command()
def run(
    symbol: Annotated[
        str,
        typer.Argument(help="Trading symbol (e.g., BTC/USDT)"),
    ] = "BTC/USDT",
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s) to backtest"),
    ] = [2024, 2025],  # noqa: B006
    report: Annotated[
        bool,
        typer.Option("--report/--no-report", help="Generate QuantStats HTML report"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V", help="Enable verbose output"),
    ] = False,
    capital: Annotated[
        float,
        typer.Option("--capital", "-c", help="Initial capital (USD)"),
    ] = 10000.0,
) -> None:
    """Run VW-TSMOM backtest on historical data.

    Uses default configurations for both strategy and portfolio management:
        - TSMOMConfig: Default strategy parameters (use `info` command to see)
        - Portfolio: Default settings (use `info` command to see)

    For parameter optimization, use the `optimize` command instead.

    Example:
        uv run python -m src.cli.backtest run BTC/USDT --year 2024 --year 2025
        uv run python -m src.cli.backtest run BTC/USDT -y 2024 --report
        uv run python -m src.cli.backtest run ETH/USDT -y 2025 --verbose
        uv run python -m src.cli.backtest run BTC/USDT -c 50000 --report
    """
    # 로깅 설정: verbose 모드에서 DEBUG 레벨, 아니면 WARNING 레벨
    console_level = "DEBUG" if verbose else "WARNING"
    setup_logger(console_level=console_level)

    # 전략 컨텍스트가 바인딩된 로거 생성
    ctx_logger = get_strategy_logger(strategy="VW-TSMOM", symbol=symbol)

    if verbose:
        ctx_logger.info("Debug mode enabled - detailed logs will be shown")

    # 전략 및 포트폴리오 생성 (설정값 표시를 위해 먼저 생성)
    strategy = TSMOMStrategy()  # Uses default TSMOMConfig
    portfolio = Portfolio.create(initial_capital=Decimal(str(capital)))

    # 시작 정보 패널 (실제 설정값 표시)
    _print_startup_panel(
        symbol=symbol,
        years=year,
        capital=capital,
        strategy=strategy,
        portfolio=portfolio,
    )

    # Step 1: 데이터 로드 (MarketDataService 사용)
    logger.info("Step 1: Loading data...")
    try:
        settings = get_settings()
        data_service = MarketDataService(settings)

        # 연도 범위 계산
        start_date = datetime(min(year), 1, 1, tzinfo=UTC)
        end_date = datetime(max(year), 12, 31, 23, 59, 59, tzinfo=UTC)

        data_request = MarketDataRequest(
            symbol=symbol,
            timeframe="1D",
            start=start_date,
            end=end_date,
        )

        data = data_service.get(data_request)
        logger.success(
            f"Loaded {data.symbol}: {data.periods:,} daily candles ({data.start.date()} ~ {data.end.date()})"
        )
    except DataNotFoundError as e:
        logger.error(f"Data load failed: {e}")
        raise typer.Exit(code=1) from e

    # Step 2: 전략 설정 확인
    logger.info("Step 2: Configuring strategy...")
    logger.success(
        f"Using TSMOMConfig (lookback={strategy.config.lookback}, vol_target={strategy.config.vol_target:.0%})"
    )

    if verbose:
        config_table = Table(title="Strategy Configuration (TSMOMConfig)")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")
        for key, value in strategy.config.model_dump().items():
            config_table.add_row(key, str(value))
        console.print(config_table)

    # Step 3: 포트폴리오 설정 확인
    logger.info("Step 3: Configuring portfolio...")
    logger.success(f"Portfolio ready: {portfolio}")

    if verbose:
        pm_table = Table(title="Portfolio Configuration")
        pm_table.add_column("Parameter", style="cyan")
        pm_table.add_column("Value", style="green")
        for key, value in portfolio.summary().items():
            pm_table.add_row(key, str(value))
        console.print(pm_table)

    # Step 4: 백테스트 실행
    logger.info("Step 4: Running backtest...")
    ctx_logger.info("Starting backtest engine")

    try:
        engine = BacktestEngine()

        # 백테스트 요청 생성
        request = BacktestRequest(
            data=data,
            strategy=strategy,
            portfolio=portfolio,
            analyzer=PerformanceAnalyzer() if report else None,
        )

        # 백테스트 실행
        if report:
            ctx_logger.debug("Running with returns for report generation")
            result, strategy_returns, benchmark_returns = engine.run_with_returns(
                request
            )

            # 결과 출력
            print_performance_summary(result)
            ctx_logger.info(
                "Backtest completed",
                total_return=result.metrics.total_return,
                sharpe=result.metrics.sharpe_ratio,
            )

            # HTML 리포트 생성
            logger.info("Step 5: Generating report...")
            report_path = generate_quantstats_report(
                returns=strategy_returns,
                benchmark_returns=benchmark_returns,
                title=f"{strategy.name} Backtest - {symbol}",
            )
            logger.success(f"Report saved: {report_path}")
            ctx_logger.success(f"Report generated: {report_path}")
        else:
            result = engine.run(request)
            print_performance_summary(result)
            ctx_logger.info(
                "Backtest completed",
                total_return=result.metrics.total_return,
                sharpe=result.metrics.sharpe_ratio,
            )

    except ImportError as e:
        ctx_logger.exception("VectorBT import failed")
        logger.warning(f"VectorBT import failed: {e}")
        logger.info("Install VectorBT with: pip install vectorbt")
        raise typer.Exit(code=1) from e


@app.command()
def optimize(
    symbol: Annotated[
        str,
        typer.Argument(help="Trading symbol (e.g., BTC/USDT)"),
    ] = "BTC/USDT",
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s) to backtest"),
    ] = [2024, 2025],  # noqa: B006
    top_n: Annotated[
        int,
        typer.Option("--top", "-t", help="Number of top results to show"),
    ] = 10,
    capital: Annotated[
        float,
        typer.Option("--capital", "-c", help="Initial capital (USD)"),
    ] = 10000.0,
) -> None:
    """Run parameter optimization for VW-TSMOM strategy.

    Example:
        uv run python -m src.cli.backtest optimize BTC/USDT --year 2024 --year 2025
        uv run python -m src.cli.backtest optimize BTC/USDT -c 50000 --top 5
    """
    # 로깅 설정 (WARNING 레벨로 최소화)
    setup_logger(console_level="WARNING")
    opt_logger = get_strategy_logger(strategy="Optimizer", symbol=symbol)

    console.print(
        Panel.fit(
            (
                f"[bold]VW-TSMOM Parameter Optimization[/bold]\n"
                f"Symbol: {symbol}\n"
                f"Years: {', '.join(map(str, year))}\n"
                f"Capital: ${capital:,.0f}"
            ),
            border_style="magenta",
        )
    )

    # 데이터 로드
    logger.info("Loading data...")
    opt_logger.info("Starting parameter optimization")
    try:
        settings = get_settings()
        data_service = MarketDataService(settings)

        start_date = datetime(min(year), 1, 1, tzinfo=UTC)
        end_date = datetime(max(year), 12, 31, 23, 59, 59, tzinfo=UTC)

        data = data_service.get(
            MarketDataRequest(
                symbol=symbol,
                timeframe="1D",
                start=start_date,
                end=end_date,
            )
        )
        logger.success(f"Loaded {data.periods:,} daily candles")
    except DataNotFoundError as e:
        opt_logger.exception("Data load failed")
        logger.error(f"Data load failed: {e}")
        raise typer.Exit(code=1) from e

    # 파라미터 그리드 정의 (전략 파라미터만)
    param_grid = {
        "lookback": [12, 24, 36, 48, 72],
        "vol_target": [0.10, 0.15, 0.20, 0.25],
    }

    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    logger.info(f"Running {total_combinations} parameter combinations...")
    logger.debug("Portfolio settings (max_leverage_cap=2.0) applied uniformly")

    try:
        # 포트폴리오 설정
        portfolio = Portfolio.create(initial_capital=Decimal(str(capital)))

        results = run_parameter_sweep(
            strategy_class=TSMOMStrategy,
            data=data,
            param_grid=param_grid,
            portfolio=portfolio,
            top_n=top_n,
        )

        # 결과 테이블
        results_table = Table(title=f"Top {top_n} Parameter Combinations")
        results_table.add_column("#", style="dim", justify="right")
        results_table.add_column("Lookback", justify="right")
        results_table.add_column("Vol Target", justify="right")
        results_table.add_column("Sharpe", justify="right", style="cyan")
        results_table.add_column("Return", justify="right", style="green")
        results_table.add_column("MDD", justify="right", style="red")
        results_table.add_column("Win%", justify="right")

        for idx, (_, row) in enumerate(results.iterrows(), start=1):
            results_table.add_row(
                str(idx),
                str(int(row["lookback"])),
                f"{row['vol_target']:.0%}",
                f"{row['sharpe_ratio']:.2f}",
                f"{row['total_return']:+.1f}%",
                f"{row['max_drawdown']:.1f}%",
                f"{row['win_rate']:.1f}%",
            )

        console.print(results_table)

        # 최적 파라미터
        best = results.iloc[0]
        opt_logger.success(
            "Optimization completed",
            best_lookback=int(best["lookback"]),
            best_vol_target=best["vol_target"],
            best_sharpe=best["sharpe_ratio"],
        )
        console.print(
            Panel(
                (
                    f"[bold green]Best Strategy Parameters[/bold green]\n\n"
                    f"Lookback: {int(best['lookback'])} hours\n"
                    f"Vol Target: {best['vol_target']:.0%}\n\n"
                    f"[bold]Performance[/bold] (with max_leverage_cap=2.0x)\n"
                    f"Sharpe: {best['sharpe_ratio']:.2f} | "
                    f"Return: {best['total_return']:+.1f}% | "
                    f"MDD: {best['max_drawdown']:.1f}%"
                ),
                border_style="green",
            )
        )

    except ImportError as e:
        opt_logger.exception("VectorBT import failed")
        logger.warning(f"VectorBT import failed: {e}")
        raise typer.Exit(code=1) from e


@app.command()
def diagnose(
    symbol: Annotated[
        str,
        typer.Argument(help="Trading symbol (e.g., BTC/USDT)"),
    ] = "BTC/USDT",
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s) to backtest"),
    ] = [2024, 2025],  # noqa: B006
    window: Annotated[
        int,
        typer.Option("--window", "-w", help="Rolling window for Beta calculation"),
    ] = 60,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V", help="Enable verbose output"),
    ] = False,
    capital: Annotated[
        float,
        typer.Option("--capital", "-c", help="Initial capital (USD)"),
    ] = 10000.0,
) -> None:
    """Run Beta Attribution diagnosis for VW-TSMOM strategy.

    Analyzes why the strategy may not be capturing market upside by
    decomposing Beta losses across each filter stage.

    Filter Stages Analyzed:
        1. Trend Filter - Removes counter-trend signals
        2. Deadband - Filters weak signals
        3. Vol Scaling - Adjusts position size by volatility

    Example:
        uv run python -m src.cli.backtest diagnose BTC/USDT --year 2024 --year 2025
        uv run python -m src.cli.backtest diagnose BTC/USDT -y 2024 --window 30
        uv run python -m src.cli.backtest diagnose ETH/USDT -y 2025 --verbose
    """
    from src.strategy.tsmom.preprocessor import preprocess

    # 로깅 설정
    console_level = "DEBUG" if verbose else "WARNING"
    setup_logger(console_level=console_level)

    ctx_logger = get_strategy_logger(strategy="VW-TSMOM-Diagnosis", symbol=symbol)

    console.print(
        Panel.fit(
            (
                f"[bold]VW-TSMOM Beta Attribution Diagnosis[/bold]\n"
                f"Symbol: {symbol}\n"
                f"Years: {', '.join(map(str, year))}\n"
                f"Rolling Window: {window} days"
            ),
            border_style="yellow",
        )
    )

    # Step 1: 데이터 로드
    logger.info("Step 1: Loading data...")
    try:
        settings = get_settings()
        data_service = MarketDataService(settings)

        start_date = datetime(min(year), 1, 1, tzinfo=UTC)
        end_date = datetime(max(year), 12, 31, 23, 59, 59, tzinfo=UTC)

        data_request = MarketDataRequest(
            symbol=symbol,
            timeframe="1D",
            start=start_date,
            end=end_date,
        )

        data = data_service.get(data_request)
        logger.success(
            f"Loaded {data.symbol}: {data.periods:,} daily candles "
            + f"({data.start.date()} ~ {data.end.date()})"
        )
    except DataNotFoundError as e:
        logger.error(f"Data load failed: {e}")
        raise typer.Exit(code=1) from e

    # Step 2: 전략 설정 및 전처리
    logger.info("Step 2: Preprocessing data and generating signals...")
    config = TSMOMConfig()
    processed_df = preprocess(data.ohlcv, config)

    # Step 3: 진단 데이터 수집과 함께 시그널 생성
    ctx_logger.info("Generating signals with diagnostics")
    result = generate_signals_with_diagnostics(processed_df, config, symbol)
    diagnostics_df = result.diagnostics_df

    logger.success(f"Generated {len(diagnostics_df)} diagnostic records")

    # Step 4: 벤치마크 수익률 계산
    logger.info("Step 3: Calculating benchmark returns...")
    close_series = data.ohlcv["close"]
    benchmark_returns = close_series.pct_change().dropna()

    # Step 5: Beta Attribution 분석
    logger.info("Step 4: Running Beta Attribution analysis...")
    attribution = calculate_beta_attribution(
        diagnostics_df,
        benchmark_returns,  # type: ignore[arg-type]
        window=window,
    )

    # Step 6: 시그널 억제 통계
    suppression_stats = summarize_suppression_impact(diagnostics_df)

    # ========== 결과 출력 ==========

    # Beta Attribution Summary Panel
    beta_panel_content = (
        f"[bold cyan]Beta at Each Stage[/bold cyan]\n"
        f"  Potential Beta:         {attribution.potential_beta:>7.3f}\n"
        f"  After Trend Filter:     {attribution.beta_after_trend_filter:>7.3f}\n"
        f"  After Deadband:         {attribution.beta_after_deadband:>7.3f}\n"
        f"  [bold]Realized Beta:          {attribution.realized_beta:>7.3f}[/bold]\n\n"
        f"[bold yellow]Beta Losses by Stage[/bold yellow]\n"
        f"  Lost to Trend Filter:   {attribution.lost_to_trend_filter:>7.3f}"
    )

    if attribution.lost_to_trend_filter > 0:
        beta_panel_content += " [red](-)[/red]"

    beta_panel_content += (
        f"\n  Lost to Deadband:       {attribution.lost_to_deadband:>7.3f}"
    )
    if attribution.lost_to_deadband > 0:
        beta_panel_content += " [red](-)[/red]"

    beta_panel_content += (
        f"\n  Lost to Vol Scaling:    {attribution.lost_to_vol_scaling:>7.3f}"
    )
    if attribution.lost_to_vol_scaling > 0:
        beta_panel_content += " [red](-)[/red]"

    beta_panel_content += f"\n\n[bold green]Beta Retention: {attribution.beta_retention_ratio:.1%}[/bold green]"

    console.print(
        Panel(beta_panel_content, title="Beta Attribution", border_style="cyan")
    )

    # Signal Suppression Table
    suppression_table = Table(title="Signal Suppression Analysis")
    suppression_table.add_column("Reason", style="cyan")
    suppression_table.add_column("Count", justify="right")
    suppression_table.add_column("Percentage", justify="right")
    suppression_table.add_column("Avg Weight", justify="right")

    for reason, stats in suppression_stats.items():
        style = "green" if reason == "none" else "yellow"
        suppression_table.add_row(
            reason,
            f"{int(stats['count']):,}",
            f"{stats['percentage']:.1f}%",
            f"{stats['avg_potential_weight']:.3f}",
            style=style,
        )

    console.print(suppression_table)

    # 권장사항 패널
    recommendations: list[str] = []

    if attribution.lost_to_trend_filter > BETA_LOSS_THRESHOLD_LOW:
        recommendations.append(
            "[yellow]Trend Filter[/yellow]: Consider disabling or relaxing "
            + "(use_trend_filter=False or higher trend_ma_period)"
        )

    if attribution.lost_to_deadband > BETA_LOSS_THRESHOLD_LOW:
        recommendations.append(
            "[yellow]Deadband[/yellow]: Consider lowering deadband_threshold "
            + f"(current: {config.deadband_threshold})"
        )

    if attribution.lost_to_vol_scaling > BETA_LOSS_THRESHOLD_HIGH:
        recommendations.append(
            "[yellow]Vol Scaling[/yellow]: Consider raising vol_target "
            + f"(current: {config.vol_target:.0%})"
        )

    if attribution.beta_retention_ratio > BETA_RETENTION_GOOD:
        recommendations.append(
            "[green]Good![/green] Strategy retains most of market beta "
            + f"({attribution.beta_retention_ratio:.0%})"
        )

    if recommendations:
        rec_content = "\n".join(
            f"  {i + 1}. {rec}" for i, rec in enumerate(recommendations)
        )
        border = (
            "green"
            if attribution.beta_retention_ratio > BETA_RETENTION_GOOD
            else "yellow"
        )
        console.print(
            Panel(
                f"[bold]Recommendations[/bold]\n\n{rec_content}",
                border_style=border,
            )
        )

    ctx_logger.success(
        "Diagnosis completed",
        potential_beta=attribution.potential_beta,
        realized_beta=attribution.realized_beta,
        retention=f"{attribution.beta_retention_ratio:.1%}",
    )


@app.command()
def info() -> None:
    """Display VW-TSMOM strategy information."""
    console.print(
        Panel.fit(
            (
                "[bold]VW-TSMOM Strategy Information[/bold]\n"
                "Volume-Weighted Time Series Momentum"
            ),
            border_style="blue",
        )
    )

    # 전략 설명
    description = """
[bold]Algorithm Overview[/bold]

VW-TSMOM combines volume-weighted returns with volatility scaling:

1. [cyan]Volume-Weighted Momentum[/cyan]
   - Weight returns by trading volume
   - High volume = stronger signal

2. [cyan]Volatility Scaling[/cyan]
   - Target a specific annual volatility
   - Reduce position size in high volatility

[bold]Architecture (Clean Architecture)[/bold]

- [yellow]MarketDataService[/yellow]: Data access abstraction
- [yellow]Strategy (TSMOMConfig)[/yellow]: Signal generation (lookback, vol_target)
- [yellow]Portfolio[/yellow]: Capital + execution rules (max_leverage_cap, etc.)
- [yellow]BacktestEngine[/yellow]: Stateless executor
- [yellow]PerformanceAnalyzer[/yellow]: Metrics extraction
"""
    console.print(description)

    # Strategy Config
    console.print("[bold]Strategy Configuration (TSMOMConfig)[/bold]")
    default_config = TSMOMConfig()
    strategy_table = Table()
    strategy_table.add_column("Parameter", style="cyan")
    strategy_table.add_column("Value", style="green")
    strategy_table.add_column("Description")

    strategy_params = [
        ("lookback", default_config.lookback, "Momentum calculation window (hours)"),
        (
            "vol_window",
            default_config.vol_window,
            "Volatility calculation window (hours)",
        ),
        ("vol_target", f"{default_config.vol_target:.0%}", "Annual volatility target"),
        (
            "min_volatility",
            f"{default_config.min_volatility:.0%}",
            "Minimum volatility clamp",
        ),
    ]

    for name, value, desc in strategy_params:
        strategy_table.add_row(name, str(value), desc)

    console.print(strategy_table)

    # Portfolio Config
    console.print("\n[bold]Portfolio Configuration[/bold]")
    default_portfolio = Portfolio.create()
    pm_table = Table()
    pm_table.add_column("Parameter", style="cyan")
    pm_table.add_column("Value", style="green")
    pm_table.add_column("Description")

    pm_params = [
        (
            "initial_capital",
            f"${default_portfolio.initial_capital:,.0f}",
            "Starting capital",
        ),
        (
            "execution_mode",
            default_portfolio.config.execution_mode,
            "orders=continuous rebalancing, signals=event-based",
        ),
        (
            "max_leverage_cap",
            f"{default_portfolio.config.max_leverage_cap}x",
            "Maximum leverage cap (system limit)",
        ),
        (
            "rebalance_threshold",
            f"{default_portfolio.config.rebalance_threshold:.0%}",
            "Min change to trigger rebalancing",
        ),
    ]

    for name, value, desc in pm_params:
        pm_table.add_row(name, str(value), desc)

    console.print(pm_table)

    # 프리셋
    console.print("\n[bold]Available Presets[/bold]")

    presets_table = Table()
    presets_table.add_column("Type", style="dim")
    presets_table.add_column("Preset", style="cyan")
    presets_table.add_column("Description")

    presets_table.add_row(
        "Strategy", "TSMOMConfig.conservative()", "Longer lookback, lower vol target"
    )
    presets_table.add_row(
        "Strategy", "TSMOMConfig.aggressive()", "Shorter lookback, higher vol target"
    )
    presets_table.add_row(
        "Portfolio", "Portfolio.conservative()", "Lower leverage cap, tighter stop loss"
    )
    presets_table.add_row(
        "Portfolio", "Portfolio.aggressive()", "Higher leverage cap, faster rebalancing"
    )
    presets_table.add_row(
        "Portfolio", "Portfolio.paper_trading()", "Zero costs, for research only"
    )

    console.print(presets_table)


# Main entry point
if __name__ == "__main__":
    app()
