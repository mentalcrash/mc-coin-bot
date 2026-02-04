"""Typer CLI for backtesting strategies.

이 모듈은 다양한 트레이딩 전략의 백테스팅을 위한 CLI를 제공합니다.
Strategy Registry Pattern을 사용하여 전략 독립적으로 설계되었습니다.

Commands:
    - run: 단일 백테스트 실행
    - optimize: 파라미터 최적화 실행
    - strategies: 사용 가능한 전략 목록
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
from src.portfolio import Portfolio, PortfolioManagerConfig
from src.strategy import BaseStrategy, get_strategy, list_strategies

# NOTE: Legacy imports for backward compatibility (diagnose command)
# TODO: Migrate diagnose command to use Strategy-agnostic diagnostics
from src.strategy.tsmom import TSMOMConfig, TSMOMStrategy
from src.strategy.tsmom.signal import generate_signals_with_diagnostics

# Global Console Instance (Rich UI for user-facing output)
console = Console()

# Beta Attribution Thresholds
BETA_LOSS_THRESHOLD_LOW = 0.1
BETA_LOSS_THRESHOLD_HIGH = 0.2
BETA_RETENTION_GOOD = 0.7
BETA_RETENTION_WARNING = 0.5
BETA_RETENTION_CRITICAL = 0.3
VOL_SCALING_AMPLIFICATION_THRESHOLD = -0.1
NEUTRAL_RATIO_HIGH = 0.5
LONG_RATIO_LOW = 0.1
BENCHMARK_RETURN_SIGNIFICANT = 20

# Signal Quality Thresholds
HIT_RATE_GOOD = 55
HIT_RATE_AVERAGE = 50
LEVERAGE_CAP_LOW = 20
LEVERAGE_CAP_MEDIUM = 50
LOST_RETURN_THRESHOLD = 5

# Breakout Diagnosis Thresholds
BREAKOUT_LOW_EXPOSURE_THRESHOLD = 0.95


def _print_startup_panel(
    symbol: str,
    years: list[int],
    capital: float,
    strategy: BaseStrategy,
    portfolio: Portfolio,
) -> None:
    """백테스트 시작 정보 패널 출력.

    전략과 포트폴리오의 핵심 설정값을 사용자에게 표시합니다.
    전략 독립적으로 get_startup_info()를 사용합니다.

    Args:
        symbol: 거래 심볼
        years: 백테스트 연도 목록
        capital: 초기 자본금
        strategy: 전략 인스턴스 (BaseStrategy)
        portfolio: 포트폴리오 인스턴스
    """
    pm_cfg = portfolio.config

    # 전략 설정 요약 (전략이 제공하는 메타데이터 사용)
    strategy_info_dict = strategy.get_startup_info()
    strategy_info = "\n".join(f"  {k}: {v}" for k, v in strategy_info_dict.items())

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
        f"[bold]{strategy.name} Backtest[/bold]\n"
        f"Symbol: {symbol}\n"
        f"Years: {', '.join(map(str, years))}\n"
        f"Capital: ${capital:,.0f}\n\n"
        f"[bold cyan]Strategy ({strategy.name})[/bold cyan]\n"
        f"{strategy_info}\n\n"
        f"[bold cyan]Portfolio[/bold cyan]\n"
        f"{portfolio_info}"
    )

    console.print(Panel.fit(panel_content, border_style="blue"))


# Typer App
app = typer.Typer(
    name="backtest",
    help="Strategy Backtesting CLI (supports multiple strategies via Registry)",
    no_args_is_help=True,
)


@app.command()
def run(
    symbol: Annotated[
        str,
        typer.Argument(help="Trading symbol (e.g., BTC/USDT)"),
    ] = "BTC/USDT",
    strategy_name: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help="Strategy name (use 'strategies' command to list available)",
        ),
    ] = "tsmom",
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
    use_recommended: Annotated[
        bool,
        typer.Option(
            "--recommended/--custom",
            help="Use strategy's recommended portfolio settings",
        ),
    ] = True,
) -> None:
    """Run strategy backtest on historical data.

    Supports multiple strategies via the Registry pattern.
    Use --strategy to select a strategy (default: tsmom).
    Use --recommended to use the strategy's optimized portfolio settings.

    Example:
        uv run python -m src.cli.backtest run BTC/USDT --year 2024 --year 2025
        uv run python -m src.cli.backtest run BTC/USDT -s adaptive-breakout -y 2024
        uv run python -m src.cli.backtest run BTC/USDT -s tsmom --custom -c 50000
        uv run python -m src.cli.backtest run ETH/USDT -y 2025 --verbose --report
    """
    # 로깅 설정: verbose 모드에서 DEBUG 레벨, 아니면 WARNING 레벨
    console_level = "DEBUG" if verbose else "WARNING"
    setup_logger(console_level=console_level)

    # 전략 클래스 로드 (Registry에서)
    try:
        strategy_class = get_strategy(strategy_name)
    except KeyError as e:
        logger.error(f"Strategy not found: {e}")
        available = ", ".join(list_strategies())
        console.print(f"[red]Error:[/red] Strategy '{strategy_name}' not found.")
        console.print(f"Available strategies: {available}")
        raise typer.Exit(code=1) from e

    # 전략 컨텍스트가 바인딩된 로거 생성
    strategy_instance = strategy_class()
    ctx_logger = get_strategy_logger(strategy=strategy_instance.name, symbol=symbol)

    if verbose:
        ctx_logger.info("Debug mode enabled - detailed logs will be shown")

    # 포트폴리오 생성 (권장 설정 또는 기본 설정)
    if use_recommended:
        config_kwargs = strategy_class.recommended_config()
        portfolio = Portfolio.create(
            initial_capital=Decimal(str(capital)),
            config=PortfolioManagerConfig(**config_kwargs),
        )
        ctx_logger.debug(f"Using recommended portfolio for {strategy_instance.name}")
    else:
        portfolio = Portfolio.create(initial_capital=Decimal(str(capital)))
        ctx_logger.debug("Using default portfolio settings")

    # 시작 정보 패널 (실제 설정값 표시)
    _print_startup_panel(
        symbol=symbol,
        years=year,
        capital=capital,
        strategy=strategy_instance,
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
    strategy_info = strategy_instance.get_startup_info()
    logger.success(f"Using {strategy_instance.name} strategy")

    if verbose:
        config_table = Table(title=f"Strategy Configuration ({strategy_instance.name})")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")
        for key, value in strategy_info.items():
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
            strategy=strategy_instance,
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
                title=f"{strategy_instance.name} Backtest - {symbol}",
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
def strategies() -> None:
    """List all available strategies.

    Shows registered strategies with their descriptions and recommended settings.

    Example:
        uv run python -m src.cli.backtest strategies
    """
    available = list_strategies()

    if not available:
        console.print("[yellow]No strategies registered.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title="Available Strategies")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    table.add_column("Recommended Portfolio", style="yellow")

    for name in available:
        strategy_class = get_strategy(name)

        # 전략 설명 (docstring 첫 줄)
        doc = strategy_class.__doc__ or "No description"
        description = doc.split("\n")[0].strip()

        # 권장 포트폴리오 설정 요약
        try:
            config_kwargs = strategy_class.recommended_config()
            pm_cfg = PortfolioManagerConfig(**config_kwargs)
            if pm_cfg.system_stop_loss:
                portfolio_info = f"Lev: {pm_cfg.max_leverage_cap}x, SL: {pm_cfg.system_stop_loss:.0%}"
            else:
                portfolio_info = f"Lev: {pm_cfg.max_leverage_cap}x, SL: Disabled"
        except Exception:
            portfolio_info = "N/A"

        table.add_row(name, description, portfolio_info)

    console.print(table)
    console.print("\nUse [cyan]--strategy <name>[/cyan] with the run command.")


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


def _diagnose_breakout(
    symbol: str,
    year: list[int],
    verbose: bool,
) -> None:
    """Adaptive Breakout 전략 진단.

    ATR 기반 임계값 필터링과 돌파 감지 효율을 분석합니다.

    Args:
        symbol: 거래 심볼
        year: 분석 연도 목록
        verbose: 상세 로그 출력 여부
    """
    from src.strategy.breakout import AdaptiveBreakoutConfig, AdaptiveBreakoutStrategy

    # 로깅 설정
    console_level = "DEBUG" if verbose else "WARNING"
    setup_logger(console_level=console_level)
    ctx_logger = get_strategy_logger(
        strategy="AdaptiveBreakout-Diagnosis", symbol=symbol
    )

    config = AdaptiveBreakoutConfig()

    console.print(
        Panel.fit(
            (
                f"[bold]Adaptive Breakout Signal Diagnosis[/bold]\n"
                f"Symbol: {symbol}\n"
                f"Years: {', '.join(map(str, year))}\n"
                f"k_value: {config.k_value}x ATR"
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

        data = data_service.get(
            MarketDataRequest(
                symbol=symbol,
                timeframe="1D",
                start=start_date,
                end=end_date,
            )
        )
        logger.success(
            f"Loaded {data.symbol}: {data.periods:,} daily candles "
            + f"({data.start.date()} ~ {data.end.date()})"
        )
    except DataNotFoundError as e:
        logger.error(f"Data load failed: {e}")
        raise typer.Exit(code=1) from e

    # Step 2: 전략 실행 및 시그널 생성
    logger.info("Step 2: Preprocessing and generating signals...")
    strategy = AdaptiveBreakoutStrategy(config)
    processed_df, signals = strategy.run(data.ohlcv)

    logger.success(f"Generated signals for {len(processed_df)} candles")

    # Step 3: 벤치마크 수익률
    benchmark_returns = data.ohlcv["close"].pct_change().dropna()
    returns_aligned = benchmark_returns.reindex(processed_df.index).fillna(0)

    # ========== 분석 ==========
    close = processed_df["close"]
    upper = processed_df["upper_band"].shift(1)
    lower = processed_df["lower_band"].shift(1)
    threshold = processed_df["threshold"].shift(1)

    # 시그널 분포
    long_days = int((signals.direction == 1).sum())
    short_days = int((signals.direction == -1).sum())
    neutral_days = int((signals.direction == 0).sum())
    total_days = len(signals.direction)

    # 돌파 통계
    simple_long = int((close > upper).sum())
    simple_short = int((close < lower).sum())
    threshold_long = int((close > (upper + threshold)).sum())
    threshold_short = int((close < (lower - threshold)).sum())

    # 방향별 수익
    long_mask = signals.direction == 1
    short_mask = signals.direction == -1
    long_returns = returns_aligned[long_mask]
    short_returns = returns_aligned[short_mask]

    long_pnl = float((signals.strength[long_mask] * long_returns).sum()) * 100
    short_pnl = float((signals.strength[short_mask] * short_returns).sum()) * 100
    long_correct = int((long_returns > 0).sum()) if long_days > 0 else 0
    short_correct = int((short_returns < 0).sum()) if short_days > 0 else 0

    # 벤치마크
    total_benchmark = float(benchmark_returns.sum()) * 100
    strategy_return = float((signals.strength * returns_aligned).sum()) * 100
    exposure = (signals.direction != 0).sum() / total_days

    # ========== 결과 출력 ==========

    # 시그널 분포 테이블
    position_table = Table(title="Position Distribution")
    position_table.add_column("Position", style="cyan")
    position_table.add_column("Days", justify="right")
    position_table.add_column("Percentage", justify="right")

    position_table.add_row(
        "Long", f"{long_days:,}", f"{long_days / total_days * 100:.1f}%"
    )
    position_table.add_row(
        "Short", f"{short_days:,}", f"{short_days / total_days * 100:.1f}%"
    )
    position_table.add_row(
        "Neutral", f"{neutral_days:,}", f"{neutral_days / total_days * 100:.1f}%"
    )
    console.print(position_table)

    # 돌파 분석 테이블
    breakout_table = Table(title="Breakout Detection Analysis")
    breakout_table.add_column("Type", style="cyan")
    breakout_table.add_column("Simple", justify="right")
    breakout_table.add_column("With Threshold", justify="right")
    breakout_table.add_column("Filtered", justify="right", style="yellow")

    long_filtered = simple_long - threshold_long
    short_filtered = simple_short - threshold_short
    breakout_table.add_row(
        "Upper (Long)",
        f"{simple_long}",
        f"{threshold_long}",
        f"{long_filtered} ({long_filtered / max(simple_long, 1) * 100:.0f}%)",
    )
    breakout_table.add_row(
        "Lower (Short)",
        f"{simple_short}",
        f"{threshold_short}",
        f"{short_filtered} ({short_filtered / max(simple_short, 1) * 100:.0f}%)",
    )
    console.print(breakout_table)

    # 방향별 성과 테이블
    direction_table = Table(title="Long/Short Performance")
    direction_table.add_column("Metric", style="cyan")
    direction_table.add_column("Long", justify="right", style="green")
    direction_table.add_column("Short", justify="right", style="red")

    direction_table.add_row(
        "Days",
        f"{long_days:,} ({long_days / total_days * 100:.1f}%)",
        f"{short_days:,} ({short_days / total_days * 100:.1f}%)",
    )
    direction_table.add_row("Cumulative PnL", f"{long_pnl:+.1f}%", f"{short_pnl:+.1f}%")
    direction_table.add_row(
        "Hit Rate",
        f"{long_correct}/{long_days} ({long_correct / max(long_days, 1) * 100:.0f}%)",
        f"{short_correct}/{max(short_days, 1)} ({short_correct / max(short_days, 1) * 100:.0f}%)",
    )
    console.print(direction_table)

    # 벤치마크 비교 테이블
    benchmark_table = Table(title="Benchmark Comparison")
    benchmark_table.add_column("Metric", style="cyan")
    benchmark_table.add_column("Value", justify="right")

    benchmark_table.add_row("Buy & Hold Return", f"{total_benchmark:+.1f}%")
    benchmark_table.add_row("Strategy Return", f"{strategy_return:+.1f}%")
    benchmark_table.add_row(
        "Alpha (vs B&H)", f"{strategy_return - total_benchmark:+.1f}%"
    )
    benchmark_table.add_row("Market Exposure", f"{exposure * 100:.1f}%")
    console.print(benchmark_table)

    # 변동성 & 임계값 통계
    vol_table = Table(title="Volatility & Threshold Statistics")
    vol_table.add_column("Metric", style="cyan")
    vol_table.add_column("Value", justify="right")

    vol_table.add_row("ATR Mean", f"${float(processed_df['atr'].mean()):,.2f}")
    vol_table.add_row("Threshold Mean", f"${float(threshold.mean()):,.2f}")
    vol_table.add_row("Band Width Mean", f"${float((upper - lower).mean()):,.2f}")
    vol_table.add_row(
        "Threshold/Band Ratio",
        f"{float((threshold / (upper - lower)).mean()) * 100:.1f}%",
    )
    console.print(vol_table)

    # 권장사항
    recommendations: list[str] = []
    issues: list[str] = []

    if neutral_days / total_days > BREAKOUT_LOW_EXPOSURE_THRESHOLD:
        issues.append(
            f"[yellow]Low Exposure:[/yellow] {neutral_days / total_days * 100:.0f}% in cash. "
            + "Consider lowering k_value (e.g., 0.5~0.75)."
        )

    if threshold_short == 0:
        issues.append(
            "[yellow]No Short Signals:[/yellow] ATR threshold may be too high for downside breakouts."
        )

    if long_days > short_days * 5 and short_days > 0:
        issues.append(
            f"[yellow]Long Bias:[/yellow] {long_days}x Long vs {short_days}x Short. "
            + "Strategy may underperform in bear markets."
        )

    if strategy_return > total_benchmark:
        recommendations.append(
            f"[green]Outperforming:[/green] Strategy beats Buy & Hold by {strategy_return - total_benchmark:+.1f}%."
        )
    elif strategy_return > 0:
        recommendations.append(
            "[yellow]Underperforming:[/yellow] Positive return but below B&H. "
            + "Consider k_value=0.5 for more signals."
        )

    if long_correct == long_days and long_days > 0:
        recommendations.append(
            "[green]100% Long Hit Rate:[/green] ATR threshold is effective at filtering noise."
        )

    if issues:
        issues_content = "\n".join(f"  • {issue}" for issue in issues)
        console.print(
            Panel(
                f"[bold]Issues Detected[/bold]\n\n{issues_content}", border_style="red"
            )
        )

    if recommendations:
        rec_content = "\n".join(
            f"  {i + 1}. {rec}" for i, rec in enumerate(recommendations)
        )
        console.print(
            Panel(f"[bold]Assessment[/bold]\n\n{rec_content}", border_style="green")
        )

    ctx_logger.success(
        "Diagnosis completed",
        long_days=long_days,
        short_days=short_days,
        strategy_return=f"{strategy_return:.1f}%",
    )


@app.command()
def diagnose(  # noqa: PLR0912
    symbol: Annotated[
        str,
        typer.Argument(help="Trading symbol (e.g., BTC/USDT)"),
    ] = "BTC/USDT",
    strategy_name: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help="Strategy name (use 'strategies' command to list available)",
        ),
    ] = "tsmom",
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
    """Run signal pipeline diagnosis for trading strategies.

    Analyzes why the strategy may not be capturing market upside by
    examining signal generation pipeline and filter effects.

    Supported Strategies:
        - tsmom: Beta Attribution analysis (Trend Filter, Deadband, Vol Scaling)
        - adaptive-breakout: Breakout detection analysis (ATR Threshold, Channel)

    Example:
        uv run python -m src.cli.backtest diagnose BTC/USDT -s tsmom -y 2024 -y 2025
        uv run python -m src.cli.backtest diagnose BTC/USDT -s adaptive-breakout -y 2024
        uv run python -m src.cli.backtest diagnose ETH/USDT -s tsmom --verbose
    """
    # 전략 분기: adaptive-breakout은 별도 진단 함수 호출
    if strategy_name == "adaptive-breakout":
        _diagnose_breakout(symbol, year, verbose)
        return

    # TSMOM 전략 진단 (기존 로직)
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

    # ========== 추가 분석: Long/Short별, 시그널 효율성 ==========

    # Long/Short 방향별 분석
    long_mask = diagnostics_df["final_target_weight"] > 0
    short_mask = diagnostics_df["final_target_weight"] < 0
    neutral_mask = diagnostics_df["final_target_weight"] == 0

    long_count = int(long_mask.sum())
    short_count = int(short_mask.sum())
    neutral_count = int(neutral_mask.sum())
    total_days = len(diagnostics_df)

    # 시장 국면별 분석
    if "trend_regime" in diagnostics_df.columns:
        bull_mask = diagnostics_df["trend_regime"] == 1
        bear_mask = diagnostics_df["trend_regime"] == -1
        bull_days = int(bull_mask.sum())
        bear_days = int(bear_mask.sum())
    else:
        bull_days = bear_days = 0

    # 시그널 강도 분석
    avg_signal_strength = float(diagnostics_df["scaled_momentum"].abs().mean())
    avg_final_weight = float(diagnostics_df["final_target_weight"].abs().mean())

    # 벤치마크 수익률 통계
    total_benchmark_return = float(benchmark_returns.sum()) * 100
    benchmark_positive_days = int((benchmark_returns > 0).sum())

    # ========== 결과 출력 ==========

    # Beta Attribution Summary Panel (파이프라인 순서 반영)
    beta_panel_content = (
        f"[bold cyan]Beta Pipeline (Correct Order)[/bold cyan]\n"
        f"  1. Potential Beta (Raw):     {attribution.potential_beta:>7.3f}\n"
        f"  2. After Vol Scaling:        {attribution.potential_beta - attribution.lost_to_vol_scaling:>7.3f}"
    )
    if attribution.lost_to_vol_scaling > 0:
        beta_panel_content += f" [red](-{attribution.lost_to_vol_scaling:.3f})[/red]"
    elif attribution.lost_to_vol_scaling < 0:
        beta_panel_content += (
            f" [green](+{-attribution.lost_to_vol_scaling:.3f})[/green]"
        )

    beta_panel_content += (
        f"\n  3. After Deadband:           {attribution.beta_after_deadband:>7.3f}"
    )
    if attribution.lost_to_deadband > 0:
        beta_panel_content += f" [red](-{attribution.lost_to_deadband:.3f})[/red]"
    elif attribution.lost_to_deadband < 0:
        beta_panel_content += f" [green](+{-attribution.lost_to_deadband:.3f})[/green]"

    beta_panel_content += (
        f"\n  4. After Trend Filter:       {attribution.realized_beta:>7.3f}"
    )
    if attribution.lost_to_trend_filter > 0:
        beta_panel_content += f" [red](-{attribution.lost_to_trend_filter:.3f})[/red]"
    elif attribution.lost_to_trend_filter < 0:
        beta_panel_content += (
            f" [green](+{-attribution.lost_to_trend_filter:.3f})[/green]"
        )

    beta_panel_content += (
        f"\n\n  [bold]Realized Beta:             {attribution.realized_beta:>7.3f}[/bold]"
        f"\n  [bold green]Beta Retention:            {attribution.beta_retention_ratio:>6.1%}[/bold green]"
    )

    console.print(
        Panel(
            beta_panel_content, title="Beta Attribution Analysis", border_style="cyan"
        )
    )

    # 포지션 분포 테이블
    position_table = Table(title="Position Distribution")
    position_table.add_column("Position", style="cyan")
    position_table.add_column("Days", justify="right")
    position_table.add_column("Percentage", justify="right")

    position_table.add_row(
        "Long", f"{long_count:,}", f"{long_count / total_days * 100:.1f}%"
    )
    position_table.add_row(
        "Short", f"{short_count:,}", f"{short_count / total_days * 100:.1f}%"
    )
    position_table.add_row(
        "Neutral", f"{neutral_count:,}", f"{neutral_count / total_days * 100:.1f}%"
    )

    console.print(position_table)

    # 시장 분석 테이블
    market_table = Table(title="Market & Signal Analysis")
    market_table.add_column("Metric", style="cyan")
    market_table.add_column("Value", justify="right")

    market_table.add_row("Total Days", f"{total_days:,}")
    market_table.add_row(
        "Bull Market Days", f"{bull_days:,} ({bull_days / total_days * 100:.1f}%)"
    )
    market_table.add_row(
        "Bear Market Days", f"{bear_days:,} ({bear_days / total_days * 100:.1f}%)"
    )
    market_table.add_row("Benchmark Return", f"{total_benchmark_return:+.1f}%")
    market_table.add_row(
        "Benchmark Up Days",
        f"{benchmark_positive_days:,} ({benchmark_positive_days / total_days * 100:.1f}%)",
    )
    market_table.add_row("Avg Signal Strength", f"{avg_signal_strength:.4f}")
    market_table.add_row("Avg Final Weight", f"{avg_final_weight:.4f}")
    market_table.add_row(
        "Signal Efficiency",
        f"{avg_final_weight / avg_signal_strength * 100:.1f}%"
        if avg_signal_strength > 0
        else "N/A",
    )

    console.print(market_table)

    # ========== 심층 분석: Long/Short 별 성과, 시그널 정확도 ==========

    # Long/Short 별 수익률 분석
    long_returns = benchmark_returns[long_mask]
    short_returns = benchmark_returns[short_mask]

    long_pnl = (
        float(
            (
                diagnostics_df.loc[long_mask, "final_target_weight"]
                * benchmark_returns[long_mask]
            ).sum()
        )
        * 100
    )
    short_pnl = (
        float(
            (
                diagnostics_df.loc[short_mask, "final_target_weight"]
                * benchmark_returns[short_mask]
            ).sum()
        )
        * 100
    )

    # Long이 수익인 날 / Short이 수익인 날
    long_profitable_days = int((long_returns > 0).sum()) if len(long_returns) > 0 else 0
    short_profitable_days = (
        int((short_returns < 0).sum()) if len(short_returns) > 0 else 0
    )  # Short은 하락 시 수익

    # Bull/Bear 별 전략 성과
    if "trend_regime" in diagnostics_df.columns:
        bull_mask_diag = diagnostics_df["trend_regime"] == 1
        bear_mask_diag = diagnostics_df["trend_regime"] == -1

        bull_strategy_return = (
            float(
                (
                    diagnostics_df.loc[bull_mask_diag, "final_target_weight"]
                    * benchmark_returns[bull_mask_diag]
                ).sum()
            )
            * 100
        )
        bear_strategy_return = (
            float(
                (
                    diagnostics_df.loc[bear_mask_diag, "final_target_weight"]
                    * benchmark_returns[bear_mask_diag]
                ).sum()
            )
            * 100
        )

        bull_benchmark_return = float(benchmark_returns[bull_mask_diag].sum()) * 100
        bear_benchmark_return = float(benchmark_returns[bear_mask_diag].sum()) * 100
    else:
        bull_strategy_return = bear_strategy_return = 0.0
        bull_benchmark_return = bear_benchmark_return = 0.0

    # 시그널 방향 정확도 (Hit Rate)
    # Long 시그널이 맞았을 때: 다음 날 가격 상승
    # Short 시그널이 맞았을 때: 다음 날 가격 하락
    import numpy as np
    import pandas as pd

    final_weights: pd.Series = diagnostics_df["final_target_weight"]  # type: ignore[assignment]
    signal_direction = pd.Series(np.sign(final_weights), index=diagnostics_df.index)

    # 인덱스를 맞춤 (diagnostics_df 기준)
    next_day_return = (
        benchmark_returns.reindex(diagnostics_df.index).shift(-1).fillna(0)
    )
    next_day_direction = pd.Series(np.sign(next_day_return), index=diagnostics_df.index)

    correct_signals = (signal_direction == next_day_direction) & (signal_direction != 0)
    total_signals = int((signal_direction != 0).sum())
    hit_rate = (
        float(correct_signals.sum()) / total_signals * 100 if total_signals > 0 else 0.0
    )

    # 레버리지 캡 영향 분석
    leverage_capped_mask = suppression_stats.get("leverage_cap", {})
    leverage_cap_count = int(leverage_capped_mask.get("count", 0))
    leverage_cap_avg_weight = float(leverage_capped_mask.get("avg_potential_weight", 0))

    # 레버리지 캡으로 인한 손실 추정 (캡이 없었다면 얼마나 더 벌었을까)
    if leverage_cap_count > 0:
        # 캡된 날들의 원래 시그널 강도
        capped_days = diagnostics_df[
            diagnostics_df["signal_suppression_reason"] == "leverage_cap"
        ]
        scaled_mom: pd.Series = capped_days["scaled_momentum"]  # type: ignore[assignment]
        bench_at_capped: pd.Series = benchmark_returns.loc[capped_days.index]  # type: ignore[assignment]
        potential_extra_return = (
            float(((scaled_mom.abs() - 2.0) * bench_at_capped.abs()).sum()) * 100
        )
    else:
        potential_extra_return = 0.0

    # Long/Short 성과 테이블
    direction_table = Table(title="Long/Short Performance Analysis")
    direction_table.add_column("Metric", style="cyan")
    direction_table.add_column("Long", justify="right", style="green")
    direction_table.add_column("Short", justify="right", style="red")

    direction_table.add_row(
        "Days",
        f"{long_count:,} ({long_count / total_days * 100:.1f}%)",
        f"{short_count:,} ({short_count / total_days * 100:.1f}%)",
    )
    direction_table.add_row(
        "Cumulative PnL",
        f"{long_pnl:+.1f}%",
        f"{short_pnl:+.1f}%",
    )
    direction_table.add_row(
        "Profitable Days",
        f"{long_profitable_days:,} ({long_profitable_days / long_count * 100:.1f}%)"
        if long_count > 0
        else "N/A",
        f"{short_profitable_days:,} ({short_profitable_days / short_count * 100:.1f}%)"
        if short_count > 0
        else "N/A",
    )
    direction_table.add_row(
        "Avg Daily Return",
        f"{float(long_returns.mean()) * 100:+.2f}%" if len(long_returns) > 0 else "N/A",
        f"{float(short_returns.mean()) * 100:+.2f}%"
        if len(short_returns) > 0
        else "N/A",
    )

    console.print(direction_table)

    # Bull/Bear 시장 별 성과 테이블
    regime_table = Table(title="Bull/Bear Market Performance")
    regime_table.add_column("Metric", style="cyan")
    regime_table.add_column("Bull (Price > MA)", justify="right", style="green")
    regime_table.add_column("Bear (Price < MA)", justify="right", style="red")

    regime_table.add_row(
        "Days",
        f"{bull_days:,} ({bull_days / total_days * 100:.1f}%)",
        f"{bear_days:,} ({bear_days / total_days * 100:.1f}%)",
    )
    regime_table.add_row(
        "Benchmark Return",
        f"{bull_benchmark_return:+.1f}%",
        f"{bear_benchmark_return:+.1f}%",
    )
    regime_table.add_row(
        "Strategy Return",
        f"{bull_strategy_return:+.1f}%",
        f"{bear_strategy_return:+.1f}%",
    )
    regime_table.add_row(
        "Capture Rate",
        f"{bull_strategy_return / bull_benchmark_return * 100:.1f}%"
        if bull_benchmark_return != 0
        else "N/A",
        f"{bear_strategy_return / bear_benchmark_return * 100:.1f}%"
        if bear_benchmark_return != 0
        else "N/A",
    )

    console.print(regime_table)

    # 시그널 품질 테이블
    quality_table = Table(title="Signal Quality & Leverage Analysis")
    quality_table.add_column("Metric", style="cyan")
    quality_table.add_column("Value", justify="right")
    quality_table.add_column("Assessment", justify="right")

    # Hit Rate 평가
    if hit_rate > HIT_RATE_GOOD:
        hit_assessment = "[green]Good[/green]"
    elif hit_rate > HIT_RATE_AVERAGE:
        hit_assessment = "[yellow]Average[/yellow]"
    else:
        hit_assessment = "[red]Poor[/red]"
    quality_table.add_row("Signal Hit Rate", f"{hit_rate:.1f}%", hit_assessment)

    # 레버리지 캡 영향
    if leverage_cap_count < LEVERAGE_CAP_LOW:
        leverage_assessment = "[green]Low[/green]"
    elif leverage_cap_count < LEVERAGE_CAP_MEDIUM:
        leverage_assessment = "[yellow]Medium[/yellow]"
    else:
        leverage_assessment = "[red]High[/red]"
    quality_table.add_row(
        "Leverage Capped Days",
        f"{leverage_cap_count:,} ({leverage_cap_count / total_days * 100:.1f}%)",
        leverage_assessment,
    )
    quality_table.add_row(
        "Avg Capped Signal",
        f"{leverage_cap_avg_weight:.2f}x" if leverage_cap_count > 0 else "N/A",
        "",
    )
    lost_return_assessment = (
        "[yellow]Consider higher cap[/yellow]"
        if potential_extra_return > LOST_RETURN_THRESHOLD
        else ""
    )
    quality_table.add_row(
        "Est. Lost Return (Capping)",
        f"{potential_extra_return:+.1f}%",
        lost_return_assessment,
    )

    console.print(quality_table)

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

    # 권장사항 패널 (더 구체적인 분석)
    recommendations: list[str] = []
    issues: list[str] = []

    # 1. Beta 보존율 분석
    if attribution.beta_retention_ratio < BETA_RETENTION_CRITICAL:
        issues.append(
            f"[bold red]Critical:[/bold red] Beta retention is very low ({attribution.beta_retention_ratio:.0%}). "
            + "Strategy is missing most market moves."
        )
    elif attribution.beta_retention_ratio < BETA_RETENTION_WARNING:
        issues.append(
            f"[yellow]Warning:[/yellow] Beta retention is low ({attribution.beta_retention_ratio:.0%}). "
            + "Consider relaxing filters."
        )

    # 2. Vol Scaling 영향 분석
    if attribution.lost_to_vol_scaling > BETA_LOSS_THRESHOLD_HIGH:
        recommendations.append(
            f"[yellow]Vol Scaling[/yellow]: Lost {attribution.lost_to_vol_scaling:.3f} Beta. "
            + f"Current vol_target={config.vol_target:.0%}. "
            + "Consider raising to 50-60% to increase exposure."
        )
    elif attribution.lost_to_vol_scaling < VOL_SCALING_AMPLIFICATION_THRESHOLD:
        recommendations.append(
            f"[cyan]Vol Scaling[/cyan]: Added {-attribution.lost_to_vol_scaling:.3f} Beta (leverage). "
            + "Vol scaling is amplifying positions."
        )

    # 3. Deadband 영향 분석
    if attribution.lost_to_deadband > BETA_LOSS_THRESHOLD_LOW:
        recommendations.append(
            f"[yellow]Deadband[/yellow]: Lost {attribution.lost_to_deadband:.3f} Beta. "
            + f"Current threshold={config.deadband_threshold:.2f}. "
            + "Consider lowering to 0.1 or disabling."
        )

    # 4. Trend Filter 영향 분석
    if attribution.lost_to_trend_filter > BETA_LOSS_THRESHOLD_LOW:
        recommendations.append(
            f"[yellow]Trend Filter[/yellow]: Lost {attribution.lost_to_trend_filter:.3f} Beta. "
            + f"MA period={config.trend_ma_period}. "
            + "Consider disabling (use_trend_filter=False) or longer MA (100+)."
        )

    # 5. 포지션 불균형 분석
    long_ratio = long_count / total_days if total_days > 0 else 0
    neutral_ratio = neutral_count / total_days if total_days > 0 else 0

    if neutral_ratio > NEUTRAL_RATIO_HIGH:
        issues.append(
            f"[yellow]High Neutral:[/yellow] {neutral_ratio:.0%} of days with no position. "
            + "Filters may be too aggressive."
        )

    if (
        long_ratio < LONG_RATIO_LOW
        and total_benchmark_return > BENCHMARK_RETURN_SIGNIFICANT
    ):
        issues.append(
            f"[yellow]Missing Bull:[/yellow] Only {long_ratio:.0%} long days in +{total_benchmark_return:.0f}% market. "
            + "Strategy is missing upside."
        )

    # 6. 성공 케이스
    if attribution.beta_retention_ratio > BETA_RETENTION_GOOD:
        recommendations.append(
            f"[green]Good![/green] Strategy retains {attribution.beta_retention_ratio:.0%} of market beta. "
            + "Filters are well-calibrated."
        )

    # 출력
    if issues:
        issues_content = "\n".join(f"  • {issue}" for issue in issues)
        console.print(
            Panel(
                f"[bold]Issues Detected[/bold]\n\n{issues_content}",
                border_style="red",
            )
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
