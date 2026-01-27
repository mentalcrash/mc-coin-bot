"""Typer CLI for data ingestion pipeline.

This module provides a command-line interface for the data ingestion
pipeline using Typer with Rich UI integration.

Commands:
    - bronze: Fetch raw data from API and save to Bronze layer
    - silver: Process Bronze data with gap-filling to Silver layer
    - pipeline: Run full Bronze → Silver pipeline
    - validate: Validate Parquet file integrity

Rules Applied:
    - #18 Typer CLI: Annotated syntax, Rich UI, async handling
"""

import asyncio
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.config.settings import get_settings
from src.core.logger import setup_logger
from src.data.bronze import BronzeStorage
from src.data.fetcher import DataFetcher
from src.data.silver import SilverProcessor

# Global Console Instance
console = Console()

# Typer App
app = typer.Typer(
    name="ingest",
    help="CCXT Binance Data Ingestion Pipeline (Medallion Architecture)",
    no_args_is_help=True,
)


def _display_settings() -> None:
    """현재 설정을 테이블로 표시."""
    settings = get_settings()
    table = Table(title="Current Settings", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Bronze Directory", str(settings.bronze_dir))
    table.add_row("Silver Directory", str(settings.silver_dir))
    table.add_row("Log Directory", str(settings.log_dir))
    table.add_row("Batch Size", str(settings.batch_size))
    table.add_row("Max Retries", str(settings.max_retries))
    table.add_row("API Credentials", "Set" if settings.has_api_credentials() else "Not Set")

    console.print(table)


async def _fetch_bronze(symbol: str, years: list[int], show_progress: bool) -> list[Path]:
    """Bronze 데이터 수집 비동기 로직."""
    settings = get_settings()
    settings.ensure_directories()

    fetcher = DataFetcher(settings)
    bronze = BronzeStorage(settings)

    saved_paths: list[Path] = []

    for year in years:
        console.print(f"\n[bold blue]Fetching {symbol} for year {year}...[/bold blue]")

        try:
            batch = await fetcher.fetch_year(symbol, year, show_progress=show_progress)
            path = bronze.save(batch, year)
            saved_paths.append(path)

            console.print(
                Panel(
                    f"[green]✓[/green] Saved {batch.candle_count:,} candles to:\n{path}",
                    title=f"Bronze Complete: {symbol} {year}",
                    border_style="green",
                )
            )

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(code=1) from e

    return saved_paths


@app.command()
def bronze(
    symbol: Annotated[str, typer.Argument(help="Trading symbol (e.g., BTC/USDT)")],
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s) to fetch (can specify multiple)"),
    ],
    no_progress: Annotated[
        bool,
        typer.Option("--no-progress", help="Disable progress bar"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Fetch raw OHLCV data from Binance and save to Bronze layer.

    Example:
        python -m src.cli.ingest bronze BTC/USDT --year 2025
        python -m src.cli.ingest bronze ETH/USDT -y 2024 -y 2025
    """
    # 로거 설정
    settings = get_settings()
    setup_logger(
        log_dir=settings.log_dir,
        console_level="DEBUG" if verbose else "INFO",
    )

    console.print(Panel.fit(
        f"[bold]Bronze Data Ingestion[/bold]\nSymbol: {symbol}\nYears: {', '.join(map(str, year))}",
        border_style="blue",
    ))

    if verbose:
        _display_settings()

    try:
        asyncio.run(_fetch_bronze(symbol, year, show_progress=not no_progress))
        console.print("\n[bold green]✓ Bronze ingestion completed successfully![/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]✗ Bronze ingestion failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def silver(
    symbol: Annotated[str, typer.Argument(help="Trading symbol (e.g., BTC/USDT)")],
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s) to process (can specify multiple)"),
    ],
    skip_validation: Annotated[
        bool,
        typer.Option("--skip-validation", help="Skip data validation"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Process Bronze data with gap-filling and save to Silver layer.

    Example:
        python -m src.cli.ingest silver BTC/USDT --year 2025
        python -m src.cli.ingest silver BTC/USDT -y 2024 -y 2025 --skip-validation
    """
    settings = get_settings()
    setup_logger(
        log_dir=settings.log_dir,
        console_level="DEBUG" if verbose else "INFO",
    )

    validation_status = "Disabled" if skip_validation else "Enabled"
    console.print(Panel.fit(
        f"[bold]Silver Data Processing[/bold]\nSymbol: {symbol}\nYears: {', '.join(map(str, year))}\nValidation: {validation_status}",
        border_style="yellow",
    ))

    processor = SilverProcessor(settings)

    for y in year:
        console.print(f"\n[bold yellow]Processing {symbol} for year {y}...[/bold yellow]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Bronze → Silver: {symbol} {y}", total=None)

            try:
                # 갭 분석
                gap_report = processor.analyze_gaps(symbol, y)
                progress.update(task, description=f"Gaps found: {gap_report.gap_count:,}")

                # Silver 처리
                path = processor.process(symbol, y, validate=not skip_validation)
                progress.update(task, completed=True)

                # 결과 테이블
                result_table = Table(title=f"Silver Complete: {symbol} {y}")
                result_table.add_column("Metric", style="cyan")
                result_table.add_column("Value", style="green")
                result_table.add_row("Expected Candles", f"{gap_report.expected_count:,}")
                result_table.add_row("Original Candles", f"{gap_report.actual_count:,}")
                result_table.add_row("Gaps Filled", f"{gap_report.gap_count:,}")
                result_table.add_row("Gap Percentage", f"{gap_report.gap_percentage:.2f}%")
                result_table.add_row("Output Path", str(path))

                console.print(result_table)

            except Exception as e:
                console.print(f"[bold red]Error processing {symbol} {y}:[/bold red] {e}")
                raise typer.Exit(code=1) from e

    console.print("\n[bold green]✓ Silver processing completed successfully![/bold green]")


@app.command()
def pipeline(
    symbol: Annotated[str, typer.Argument(help="Trading symbol (e.g., BTC/USDT)")],
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s) to process"),
    ],
    no_progress: Annotated[
        bool,
        typer.Option("--no-progress", help="Disable progress bar"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Run full pipeline: Bronze (fetch) → Silver (gap-fill).

    Example:
        python -m src.cli.ingest pipeline BTC/USDT --year 2025
    """
    settings = get_settings()
    setup_logger(
        log_dir=settings.log_dir,
        console_level="DEBUG" if verbose else "INFO",
    )

    console.print(Panel.fit(
        f"[bold]Full Ingestion Pipeline[/bold]\nSymbol: {symbol}\nYears: {', '.join(map(str, year))}\nSteps: Bronze → Silver",
        border_style="magenta",
    ))

    # Step 1: Bronze
    console.print("\n[bold blue]Step 1/2: Bronze (Data Fetching)[/bold blue]")
    try:
        asyncio.run(_fetch_bronze(symbol, year, show_progress=not no_progress))
    except Exception as e:
        console.print(f"[bold red]Pipeline failed at Bronze step:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    # Step 2: Silver
    console.print("\n[bold yellow]Step 2/2: Silver (Gap Filling)[/bold yellow]")
    processor = SilverProcessor(settings)

    for y in year:
        try:
            path = processor.process(symbol, y)
            console.print(f"[green]✓[/green] Silver saved: {path}")
        except Exception as e:
            console.print(f"[bold red]Pipeline failed at Silver step:[/bold red] {e}")
            raise typer.Exit(code=1) from e

    console.print(Panel(
        "[bold green]✓ Full pipeline completed successfully![/bold green]",
        border_style="green",
    ))


@app.command()
def validate(
    path: Annotated[Path, typer.Argument(help="Path to Parquet file to validate")],
) -> None:
    """Validate a Parquet file for data integrity.

    Example:
        python -m src.cli.ingest validate data/silver/BTC_USDT/2025.parquet
    """
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {path}")
        raise typer.Exit(code=1)

    console.print(f"Validating: {path}")

    try:
        df = pd.read_parquet(path)

        # 기본 정보
        info_table = Table(title="File Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")

        info_table.add_row("Rows", f"{len(df):,}")
        info_table.add_row("Columns", ", ".join(df.columns.tolist()))
        info_table.add_row("Index Type", str(type(df.index).__name__))
        info_table.add_row("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        info_table.add_row("File Size", f"{path.stat().st_size / 1024 / 1024:.2f} MB")

        console.print(info_table)

        # 검증 결과
        validation_table = Table(title="Validation Results")
        validation_table.add_column("Check", style="cyan")
        validation_table.add_column("Status", style="green")
        validation_table.add_column("Details")

        # NaN 검사
        nan_count: int = df.isna().sum().sum()  # type: ignore[assignment]
        validation_table.add_row(
            "NaN Values",
            "✓ Pass" if nan_count == 0 else "✗ Fail",
            f"{nan_count:,} NaN values found" if nan_count > 0 else "No NaN values",
        )

        # 가격 검사
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        for col in price_cols:
            invalid: int = (df[col] <= 0).sum()  # type: ignore[assignment]
            validation_table.add_row(
                f"{col.capitalize()} > 0",
                "✓ Pass" if invalid == 0 else "⚠ Warning",
                f"{invalid:,} invalid values" if invalid > 0 else "All values positive",
            )

        # 인덱스 연속성
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            gaps = df.index.to_series().diff().dropna()
            expected_diff = pd.Timedelta(minutes=1)
            irregular = (gaps != expected_diff).sum()
            validation_table.add_row(
                "Index Continuity",
                "✓ Pass" if irregular == 0 else "⚠ Warning",
                f"{irregular:,} irregular intervals" if irregular > 0 else "Continuous 1-min intervals",
            )

        console.print(validation_table)

        # 샘플 데이터
        console.print("\n[bold]Sample Data (first 5 rows):[/bold]")
        console.print(df.head().to_string())

        console.print("\n[bold green]✓ Validation complete![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Validation failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def info() -> None:
    """Display current configuration and status."""
    settings = get_settings()

    console.print(Panel.fit(
        "[bold]MC Coin Bot - Data Ingestion Pipeline[/bold]\nMedallion Architecture: Bronze → Silver",
        border_style="blue",
    ))

    _display_settings()

    # 디렉토리 상태
    status_table = Table(title="Directory Status")
    status_table.add_column("Directory", style="cyan")
    status_table.add_column("Exists", style="green")
    status_table.add_column("Files")

    for name, path in [
        ("Bronze", settings.bronze_dir),
        ("Silver", settings.silver_dir),
        ("Logs", settings.log_dir),
    ]:
        exists = path.exists()
        file_count = len(list(path.rglob("*.parquet"))) if exists else 0
        status_table.add_row(
            f"{name} ({path})",
            "✓" if exists else "✗",
            f"{file_count} parquet files" if exists else "-",
        )

    console.print(status_table)


# Main entry point
if __name__ == "__main__":
    app()
