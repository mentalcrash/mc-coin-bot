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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from src.cli.ingest_deriv_ext import app as deriv_ext_app
from src.cli.ingest_derivatives import app as derivatives_app
from src.cli.ingest_macro import app as macro_app
from src.cli.ingest_onchain import app as onchain_app
from src.cli.ingest_options import app as options_app
from src.config.settings import get_settings
from src.core.logger import setup_logger
from src.data.bronze import BronzeStorage
from src.data.fetcher import DataFetcher
from src.data.silver import SilverProcessor
from src.exchange.binance_client import BinanceClient

# Global Console Instance
console = Console()

# Typer App
app = typer.Typer(
    name="ingest",
    help="CCXT Binance Data Ingestion Pipeline (Medallion Architecture)",
    no_args_is_help=True,
)

# Sub-command 등록
app.add_typer(derivatives_app, name="derivatives")
app.add_typer(onchain_app, name="onchain")
app.add_typer(macro_app, name="macro")
app.add_typer(options_app, name="options")
app.add_typer(deriv_ext_app, name="deriv-ext")


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
        uv run mcbot ingest bronze BTC/USDT --year 2025
        uv run mcbot ingest bronze ETH/USDT -y 2024 -y 2025
    """
    # 로거 설정
    settings = get_settings()
    setup_logger(
        log_dir=settings.log_dir,
        console_level="DEBUG" if verbose else "INFO",
    )

    console.print(
        Panel.fit(
            f"[bold]Bronze Data Ingestion[/bold]\nSymbol: {symbol}\nYears: {', '.join(map(str, year))}",
            border_style="blue",
        )
    )

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
        uv run mcbot ingest silver BTC/USDT --year 2025
        uv run mcbot ingest silver BTC/USDT -y 2024 -y 2025 --skip-validation
    """
    settings = get_settings()
    setup_logger(
        log_dir=settings.log_dir,
        console_level="DEBUG" if verbose else "INFO",
    )

    validation_status = "Disabled" if skip_validation else "Enabled"
    console.print(
        Panel.fit(
            f"[bold]Silver Data Processing[/bold]\nSymbol: {symbol}\nYears: {', '.join(map(str, year))}\nValidation: {validation_status}",
            border_style="yellow",
        )
    )

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
        uv run mcbot ingest pipeline BTC/USDT --year 2025
    """
    settings = get_settings()
    setup_logger(
        log_dir=settings.log_dir,
        console_level="DEBUG" if verbose else "INFO",
    )

    console.print(
        Panel.fit(
            f"[bold]Full Ingestion Pipeline[/bold]\nSymbol: {symbol}\nYears: {', '.join(map(str, year))}\nSteps: Bronze → Silver",
            border_style="magenta",
        )
    )

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

    console.print(
        Panel(
            "[bold green]✓ Full pipeline completed successfully![/bold green]",
            border_style="green",
        )
    )


@app.command()
def validate(
    path: Annotated[Path, typer.Argument(help="Path to Parquet file to validate")],
) -> None:
    """Validate a Parquet file for data integrity.

    Example:
        uv run mcbot ingest validate data/silver/BTC_USDT/2025.parquet
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
        info_table.add_row(
            "Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        )
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
                f"{irregular:,} irregular intervals"
                if irregular > 0
                else "Continuous 1-min intervals",
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

    console.print(
        Panel.fit(
            "[bold]MC Coin Bot - Data Ingestion Pipeline[/bold]\nMedallion Architecture: Bronze → Silver",
            border_style="blue",
        )
    )

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


# =============================================================================
# Bulk Download Command
# =============================================================================


# 테이블 표시 최대 행 수
_MAX_DISPLAY_ROWS = 20


@dataclass
class BulkDownloadResult:
    """벌크 다운로드 결과 추적."""

    total: int = 0
    success: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (symbol, error)
    skipped: list[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return len(self.success)

    @property
    def failed_count(self) -> int:
        return len(self.failed)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


@dataclass
class PipelineContext:
    """파이프라인 실행 컨텍스트."""

    symbol: str
    years: list[int]
    skip_existing: bool
    result: BulkDownloadResult
    progress: Progress
    task_id: TaskID
    client: "BinanceClient"  # 재사용할 BinanceClient


async def _get_top_symbols(quote: str, limit: int) -> list[str]:
    """상위 N개 심볼 조회 (24시간 거래대금 기준)."""
    settings = get_settings()
    async with BinanceClient(settings) as client:
        return await client.fetch_top_symbols(quote=quote, limit=limit)


async def _run_pipeline_for_symbol(ctx: PipelineContext) -> None:
    """단일 심볼에 대해 파이프라인 실행.

    전달받은 BinanceClient를 재사용합니다.
    """
    settings = get_settings()
    bronze_storage = BronzeStorage(settings)
    processor = SilverProcessor(settings)
    fetcher = DataFetcher(settings, client=ctx.client)

    for year in ctx.years:
        task_key = f"{ctx.symbol}_{year}"

        # 기존 파일 확인 (skip_existing)
        if ctx.skip_existing and bronze_storage.exists(ctx.symbol, year):
            ctx.result.skipped.append(task_key)
            ctx.progress.update(
                ctx.task_id, advance=1, description=f"[yellow]Skipped[/yellow] {ctx.symbol} {year}"
            )
            continue

        try:
            # Bronze: 데이터 수집
            ctx.progress.update(
                ctx.task_id, description=f"[blue]Fetching[/blue] {ctx.symbol} {year}"
            )
            batch = await fetcher.fetch_year(ctx.symbol, year, show_progress=False)
            bronze_storage.save(batch, year)

            # Silver: 갭 필링
            ctx.progress.update(
                ctx.task_id, description=f"[yellow]Processing[/yellow] {ctx.symbol} {year}"
            )
            processor.process(ctx.symbol, year, validate=True)

            ctx.result.success.append(task_key)
            ctx.progress.update(
                ctx.task_id, advance=1, description=f"[green]Done[/green] {ctx.symbol} {year}"
            )

            # Rate Limit 안전장치: 연도별 작업 완료 후 0.5초 대기
            await asyncio.sleep(0.5)

        except Exception as e:
            ctx.result.failed.append((task_key, str(e)))
            ctx.progress.update(
                ctx.task_id, advance=1, description=f"[red]Failed[/red] {ctx.symbol} {year}: {e}"
            )
            # 에러 후에도 대기 (Rate Limit 회복 시간)
            await asyncio.sleep(2.0)


async def _bulk_download(
    symbols: list[str],
    years: list[int],
    skip_existing: bool,
    progress: Progress,
) -> BulkDownloadResult:
    """벌크 다운로드 메인 로직.

    전체 다운로드에서 BinanceClient를 단 1번만 초기화하여 재사용합니다.
    """
    settings = get_settings()
    result = BulkDownloadResult(total=len(symbols) * len(years))

    # 전체 진행률 Task
    task_id = progress.add_task(
        "[cyan]Bulk Download Progress[/cyan]",
        total=result.total,
    )

    # BinanceClient를 전체 다운로드에서 단 1번만 생성
    async with BinanceClient(settings) as client:
        # 순차 처리 (Rate Limit 고려)
        for idx, symbol in enumerate(symbols):
            ctx = PipelineContext(
                symbol=symbol,
                years=years,
                skip_existing=skip_existing,
                result=result,
                progress=progress,
                task_id=task_id,
                client=client,  # 재사용할 client 전달
            )
            await _run_pipeline_for_symbol(ctx)

            # 종목 간 추가 대기 (첫 번째 종목 후부터)
            if idx < len(symbols) - 1:  # 마지막 종목 제외
                await asyncio.sleep(1.0)

    return result


def _display_bulk_result(result: BulkDownloadResult) -> None:
    """벌크 다운로드 결과 리포트 출력."""
    # 요약 테이블
    summary_table = Table(title="Bulk Download Summary", show_header=True)
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Count", justify="right")
    summary_table.add_column("Percentage", justify="right")

    total = result.total
    summary_table.add_row(
        "[green]Success[/green]",
        str(result.success_count),
        f"{result.success_count / total * 100:.1f}%" if total > 0 else "0%",
    )
    summary_table.add_row(
        "[red]Failed[/red]",
        str(result.failed_count),
        f"{result.failed_count / total * 100:.1f}%" if total > 0 else "0%",
    )
    summary_table.add_row(
        "[yellow]Skipped[/yellow]",
        str(result.skipped_count),
        f"{result.skipped_count / total * 100:.1f}%" if total > 0 else "0%",
    )
    summary_table.add_row(
        "[bold]Total[/bold]",
        str(total),
        "100%",
        style="bold",
    )

    console.print(summary_table)

    # 실패 목록 (있는 경우)
    if result.failed:
        failed_table = Table(title="Failed Tasks", show_header=True)
        failed_table.add_column("Task", style="cyan")
        failed_table.add_column("Error", style="red")

        for task_key, error in result.failed[:_MAX_DISPLAY_ROWS]:
            failed_table.add_row(task_key, error[:80])

        if len(result.failed) > _MAX_DISPLAY_ROWS:
            failed_table.add_row("...", f"and {len(result.failed) - _MAX_DISPLAY_ROWS} more")

        console.print(failed_table)


@app.command("bulk-download")
def bulk_download(
    top: Annotated[
        int,
        typer.Option("--top", "-t", help="Number of top symbols to download"),
    ] = 100,
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s) to fetch (can specify multiple)"),
    ] = [2023, 2024, 2025],  # noqa: B006
    quote: Annotated[
        str,
        typer.Option("--quote", "-q", help="Quote currency for filtering symbols"),
    ] = "USDT",
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing/--no-skip-existing", help="Skip if Bronze file already exists"
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Only show target symbols without downloading"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Bulk download top N symbols for multiple years.

    Fetches the top N symbols by 24h quote volume from Binance,
    then runs the full Bronze -> Silver pipeline for each.

    Example:
        # Download top 100 symbols for 2023-2025 (default)
        uv run mcbot ingest bulk-download

        # Download top 10 symbols for specific years
        uv run mcbot ingest bulk-download --top 10 -y 2024 -y 2025

        # Preview target symbols without downloading
        uv run mcbot ingest bulk-download --top 100 --dry-run

        # Re-download (don't skip existing files)
        uv run mcbot ingest bulk-download --no-skip-existing
    """
    # 로거 설정
    settings = get_settings()
    setup_logger(
        log_dir=settings.log_dir,
        console_level="DEBUG" if verbose else "INFO",
    )
    settings.ensure_directories()

    # 헤더 표시
    console.print(
        Panel.fit(
            f"[bold]Bulk Download[/bold]\nTop {top} symbols by {quote} volume (24h)\nYears: {', '.join(map(str, year))}\nSkip existing: {skip_existing}",
            border_style="magenta",
        )
    )

    # Step 1: 상위 심볼 조회
    console.print("\n[bold cyan]Step 1: Fetching top symbols by quote volume...[/bold cyan]")

    try:
        symbols = asyncio.run(_get_top_symbols(quote=quote, limit=top))
    except Exception as e:
        console.print(f"[bold red]Failed to fetch top symbols:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    if not symbols:
        console.print("[bold red]No symbols found![/bold red]")
        raise typer.Exit(code=1)

    # 심볼 목록 표시
    symbols_table = Table(title=f"Top {len(symbols)} Symbols by {quote} Volume")
    symbols_table.add_column("#", style="dim", justify="right")
    symbols_table.add_column("Symbol", style="cyan")

    for i, sym in enumerate(symbols[:_MAX_DISPLAY_ROWS], 1):
        symbols_table.add_row(str(i), sym)

    if len(symbols) > _MAX_DISPLAY_ROWS:
        symbols_table.add_row("...", f"and {len(symbols) - _MAX_DISPLAY_ROWS} more")

    console.print(symbols_table)

    # Dry-run 모드
    if dry_run:
        total_tasks = len(symbols) * len(year)
        console.print(
            Panel(
                f"[yellow]Dry-run mode[/yellow]\n\nTotal tasks: {total_tasks} ({len(symbols)} symbols x {len(year)} years)\n\nRun without --dry-run to start downloading.",
                border_style="yellow",
            )
        )
        return

    # Step 2: 벌크 다운로드 실행
    console.print(
        f"\n[bold cyan]Step 2: Downloading {len(symbols)} symbols x {len(year)} years...[/bold cyan]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=1,
    ) as progress:
        result = asyncio.run(
            _bulk_download(
                symbols=symbols,
                years=year,
                skip_existing=skip_existing,
                progress=progress,
            )
        )

    # Step 3: 결과 리포트
    console.print("\n[bold cyan]Step 3: Results[/bold cyan]")
    _display_bulk_result(result)

    # 최종 상태
    if result.failed_count == 0:
        console.print(
            Panel(
                "[bold green]✓ Bulk download completed successfully![/bold green]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold yellow]⚠ Bulk download completed with {result.failed_count} failures[/bold yellow]",
                border_style="yellow",
            )
        )


# Main entry point
if __name__ == "__main__":
    app()
