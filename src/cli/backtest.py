"""Typer CLI for backtesting strategies.

이 모듈은 VW-TSMOM 전략 백테스팅을 위한 CLI를 제공합니다.
Typer + Rich UI 통합으로 직관적인 사용 경험을 제공합니다.

Commands:
    - run: 단일 백테스트 실행
    - optimize: 파라미터 최적화 실행
    - report: QuantStats 리포트 생성

Rules Applied:
    - #18 Typer CLI: Annotated syntax, Rich UI, async handling
"""

from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.backtest.cost_model import CostModel
from src.backtest.engine import BacktestEngine, run_parameter_sweep
from src.backtest.reporter import generate_quantstats_report, print_performance_summary
from src.config.settings import get_settings
from src.data.silver import SilverProcessor
from src.strategy.tsmom import TSMOMConfig, TSMOMStrategy

# Global Console Instance
console = Console()

# Typer App
app = typer.Typer(
    name="backtest",
    help="VW-TSMOM Strategy Backtesting CLI",
    no_args_is_help=True,
)


def _load_silver_data(symbol: str, years: list[int]) -> pd.DataFrame:
    """Silver 데이터 로드.

    Args:
        symbol: 심볼 (예: BTC/USDT)
        years: 연도 목록

    Returns:
        병합된 OHLCV DataFrame
    """
    settings = get_settings()
    processor = SilverProcessor(settings)

    dfs = []
    for year in years:
        try:
            df = processor.load(symbol, year)
            dfs.append(df)
            console.print(f"  [green]✓[/green] Loaded {symbol} {year}: {len(df):,} candles")
        except FileNotFoundError:
            console.print(f"  [yellow]![/yellow] {symbol} {year} not found, skipping")

    if not dfs:
        msg = f"No data found for {symbol}"
        raise FileNotFoundError(msg)

    # 병합 및 정렬
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    if not isinstance(combined, pd.DataFrame):
        msg = "Expected DataFrame from concat"
        raise TypeError(msg)

    return combined


def _resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """1분봉을 1시간봉으로 리샘플링.

    Args:
        df: 1분봉 DataFrame

    Returns:
        1시간봉 DataFrame
    """
    resampled: pd.DataFrame = df.resample("1h").agg({  # type: ignore[assignment]
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    # Decimal 타입을 float64로 변환 (Parquet에서 로드된 경우)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in resampled.columns:
            resampled[col] = pd.to_numeric(resampled[col], errors="coerce")

    return resampled


@app.command()
def run(  # noqa: PLR0913
    symbol: Annotated[
        str,
        typer.Argument(help="Trading symbol (e.g., BTC/USDT)"),
    ] = "BTC/USDT",
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s) to backtest"),
    ] = [2024, 2025],  # noqa: B006
    lookback: Annotated[
        int,
        typer.Option("--lookback", "-l", help="Momentum lookback period (hours)"),
    ] = 24,
    vol_target: Annotated[
        float,
        typer.Option("--vol-target", "-v", help="Annual volatility target (0.0-1.0)"),
    ] = 0.15,
    max_leverage: Annotated[
        float,
        typer.Option("--max-leverage", "-m", help="Maximum leverage"),
    ] = 2.0,
    cost_model: Annotated[
        str,
        typer.Option("--cost", "-c", help="Cost model (binance_futures, conservative, zero)"),
    ] = "binance_futures",
    report: Annotated[
        bool,
        typer.Option("--report/--no-report", help="Generate QuantStats HTML report"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V", help="Enable verbose output"),
    ] = False,
) -> None:
    """Run VW-TSMOM backtest on historical data.

    Example:
        uv run python -m src.cli.backtest run BTC/USDT --year 2024 --year 2025
        uv run python -m src.cli.backtest run ETH/USDT -y 2024 --lookback 48 --vol-target 0.20
        uv run python -m src.cli.backtest run BTC/USDT -y 2025 --report
    """
    console.print(Panel.fit(
        (
            f"[bold]VW-TSMOM Backtest[/bold]\n"
            f"Symbol: {symbol}\n"
            f"Years: {', '.join(map(str, year))}\n"
            f"Lookback: {lookback}h | Vol Target: {vol_target:.0%} | Max Leverage: {max_leverage}x"
        ),
        border_style="blue",
    ))

    # Step 1: 데이터 로드
    console.print("\n[bold cyan]Step 1: Loading data...[/bold cyan]")
    try:
        df = _load_silver_data(symbol, year)
        console.print(f"  Total: {len(df):,} 1-minute candles")
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    # Step 2: 리샘플링
    console.print("\n[bold cyan]Step 2: Resampling to 1h...[/bold cyan]")
    hourly_df = _resample_to_hourly(df)
    console.print(f"  Resampled: {len(hourly_df):,} hourly candles")

    # Step 3: 전략 설정
    console.print("\n[bold cyan]Step 3: Configuring strategy...[/bold cyan]")
    config = TSMOMConfig(
        lookback=lookback,
        vol_window=lookback,
        vol_target=vol_target,
        max_leverage=max_leverage,
    )
    strategy = TSMOMStrategy(config)

    if verbose:
        config_table = Table(title="Strategy Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")
        for key, value in config.model_dump().items():
            config_table.add_row(key, str(value))
        console.print(config_table)

    # Step 4: 비용 모델
    cost_models = {
        "binance_futures": CostModel.binance_futures,
        "binance_spot": CostModel.binance_spot,
        "conservative": CostModel.conservative,
        "optimistic": CostModel.optimistic,
        "zero": CostModel.zero,
    }

    if cost_model not in cost_models:
        console.print(f"[bold red]Unknown cost model:[/bold red] {cost_model}")
        console.print(f"Available: {', '.join(cost_models.keys())}")
        raise typer.Exit(code=1)

    selected_cost_model = cost_models[cost_model]()
    console.print(f"  Cost model: {cost_model} (round-trip: {selected_cost_model.round_trip_cost:.2%})")

    # Step 5: 백테스트 실행
    console.print("\n[bold cyan]Step 4: Running backtest...[/bold cyan]")

    try:
        engine = BacktestEngine(
            cost_model=selected_cost_model,
            initial_capital=10000.0,
            freq="1h",
        )

        # 백테스트 실행 (리포트 생성 시 returns도 함께 반환)
        if report:
            result, strategy_returns, benchmark_returns = engine.run_with_returns(
                strategy, hourly_df, symbol
            )

            # 결과 출력
            print_performance_summary(result)

            # HTML 리포트 생성
            console.print("\n[bold cyan]Step 5: Generating report...[/bold cyan]")
            report_path = generate_quantstats_report(
                returns=strategy_returns,
                benchmark_returns=benchmark_returns,
                title=f"{strategy.name} Backtest - {symbol}",
            )
            console.print(f"  [green]✓[/green] Report saved: {report_path}")
        else:
            result = engine.run(strategy, hourly_df, symbol)
            print_performance_summary(result)

    except ImportError as e:
        console.print(f"[bold yellow]Warning:[/bold yellow] {e}")
        console.print("Install VectorBT with: pip install vectorbt")
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
) -> None:
    """Run parameter optimization for VW-TSMOM strategy.

    Example:
        uv run python -m src.cli.backtest optimize BTC/USDT --year 2024 --year 2025
    """
    console.print(Panel.fit(
        (
            f"[bold]VW-TSMOM Parameter Optimization[/bold]\n"
            f"Symbol: {symbol}\n"
            f"Years: {', '.join(map(str, year))}"
        ),
        border_style="magenta",
    ))

    # 데이터 로드
    console.print("\n[bold cyan]Loading data...[/bold cyan]")
    try:
        df = _load_silver_data(symbol, year)
        hourly_df = _resample_to_hourly(df)
        console.print(f"  Total: {len(hourly_df):,} hourly candles")
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    # 파라미터 그리드 정의
    param_grid = {
        "lookback": [12, 24, 36, 48, 72],
        "vol_target": [0.10, 0.15, 0.20, 0.25],
        "max_leverage": [1.5, 2.0, 2.5, 3.0],
    }

    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    console.print(f"\n[bold cyan]Running {total_combinations} parameter combinations...[/bold cyan]")

    try:
        results = run_parameter_sweep(
            strategy_class=TSMOMStrategy,
            data=hourly_df,
            param_grid=param_grid,
            cost_model=CostModel.binance_futures(),
            symbol=symbol,
            top_n=top_n,
        )

        # 결과 테이블
        results_table = Table(title=f"Top {top_n} Parameter Combinations")
        results_table.add_column("#", style="dim", justify="right")
        results_table.add_column("Lookback", justify="right")
        results_table.add_column("Vol Target", justify="right")
        results_table.add_column("Leverage", justify="right")
        results_table.add_column("Sharpe", justify="right", style="cyan")
        results_table.add_column("Return", justify="right", style="green")
        results_table.add_column("MDD", justify="right", style="red")
        results_table.add_column("Win%", justify="right")

        for idx, (_, row) in enumerate(results.iterrows(), start=1):
            results_table.add_row(
                str(idx),
                str(int(row["lookback"])),
                f"{row['vol_target']:.0%}",
                f"{row['max_leverage']:.1f}x",
                f"{row['sharpe_ratio']:.2f}",
                f"{row['total_return']:+.1f}%",
                f"{row['max_drawdown']:.1f}%",
                f"{row['win_rate']:.1f}%",
            )

        console.print(results_table)

        # 최적 파라미터
        best = results.iloc[0]
        console.print(Panel(
            (
                f"[bold green]Best Parameters[/bold green]\n\n"
                f"Lookback: {int(best['lookback'])} hours\n"
                f"Vol Target: {best['vol_target']:.0%}\n"
                f"Max Leverage: {best['max_leverage']:.1f}x\n\n"
                f"[bold]Performance[/bold]\n"
                f"Sharpe: {best['sharpe_ratio']:.2f} | Return: {best['total_return']:+.1f}% | MDD: {best['max_drawdown']:.1f}%"
            ),
            border_style="green",
        ))

    except ImportError as e:
        console.print(f"[bold yellow]Warning:[/bold yellow] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def info() -> None:
    """Display VW-TSMOM strategy information."""
    console.print(Panel.fit(
        (
            "[bold]VW-TSMOM Strategy Information[/bold]\n"
            "Volume-Weighted Time Series Momentum"
        ),
        border_style="blue",
    ))

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

3. [cyan]Leverage Control[/cyan]
   - Cap maximum leverage
   - Prevent excessive risk

[bold]Default Parameters[/bold]
"""
    console.print(description)

    default_config = TSMOMConfig()
    param_table = Table(title="Default Configuration")
    param_table.add_column("Parameter", style="cyan")
    param_table.add_column("Value", style="green")
    param_table.add_column("Description")

    params = [
        ("lookback", default_config.lookback, "Momentum calculation window (hours)"),
        ("vol_window", default_config.vol_window, "Volatility calculation window (hours)"),
        ("vol_target", f"{default_config.vol_target:.0%}", "Annual volatility target"),
        ("max_leverage", f"{default_config.max_leverage}x", "Maximum position leverage"),
        ("min_volatility", f"{default_config.min_volatility:.0%}", "Minimum volatility clamp"),
    ]

    for name, value, desc in params:
        param_table.add_row(name, str(value), desc)

    console.print(param_table)

    # 프리셋
    console.print("\n[bold]Available Presets[/bold]")

    presets_table = Table()
    presets_table.add_column("Preset", style="cyan")
    presets_table.add_column("Description")
    presets_table.add_column("Risk Level")

    presets_table.add_row("conservative()", "Lower leverage, higher vol target", "[green]Low[/green]")
    presets_table.add_row("aggressive()", "Higher leverage, faster response", "[red]High[/red]")
    presets_table.add_row("for_timeframe('1h')", "Optimized for hourly data", "[yellow]Medium[/yellow]")

    console.print(presets_table)


# Main entry point
if __name__ == "__main__":
    app()
