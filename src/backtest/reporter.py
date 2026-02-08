"""QuantStats Report Generator.

이 모듈은 백테스트 결과를 QuantStats 기반 HTML 리포트로 생성합니다.
암호화폐 특화 설정(365일 거래, BTC 벤치마크)을 적용합니다.

Rules Applied:
    - #25 QuantStats Standards: periods=365, BTC benchmark
"""

import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.backtest import BacktestResult


def generate_quantstats_report(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    output_path: str | Path | None = None,
    title: str = "Strategy Performance Report",
    auto_open: bool = False,
) -> str:
    """QuantStats HTML 리포트 생성.

    Args:
        returns: 전략 수익률 시리즈 (일별/시간별)
        benchmark_returns: 벤치마크 수익률 시리즈 (선택적)
        output_path: 출력 파일 경로 (None이면 자동 생성)
        title: 리포트 제목
        auto_open: 생성 후 브라우저에서 자동 열기

    Returns:
        생성된 리포트 파일 경로

    Example:
        >>> report_path = generate_quantstats_report(
        ...     returns=strategy_returns,
        ...     benchmark_returns=btc_returns,
        ...     title="VW-TSMOM Backtest",
        ... )
    """
    qs = _import_quantstats()

    # pandas FutureWarning 억제 (QuantStats 내부 fillna downcasting)
    pd.set_option("future.no_silent_downcasting", True)

    # QuantStats 확장 활성화
    qs.extend_pandas()  # type: ignore[no-untyped-call]

    # 수익률 데이터 정제
    clean_returns = _prepare_returns_for_quantstats(returns)

    # 출력 경로 생성
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/backtest_{timestamp}.html"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 벤치마크 처리
    if benchmark_returns is not None:
        benchmark = _prepare_returns_for_quantstats(benchmark_returns)
    else:
        benchmark = None  # type: ignore[assignment]

    # HTML 리포트 생성
    qs.reports.html(  # type: ignore[no-untyped-call]
        clean_returns,
        benchmark=benchmark,
        output=str(output_path),
        title=title,
        periods_per_year=365,  # 암호화폐는 연중무휴
        compounded=True,
        rf=0.0,  # 내부 메트릭과 동일한 무위험 수익률
    )

    if auto_open:
        webbrowser.open(f"file://{output_path.absolute()}")

    return str(output_path)


def generate_report_from_backtest_result(
    result: BacktestResult,
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    output_dir: str | Path = "reports",
) -> str:
    """BacktestResult에서 종합 리포트 생성.

    Args:
        result: BacktestResult 모델
        returns: 전략 수익률 시리즈
        benchmark_returns: 벤치마크 수익률
        output_dir: 출력 디렉토리

    Returns:
        리포트 파일 경로
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_name = result.config.strategy_name.replace(" ", "_").replace("-", "_")
    symbol = result.config.symbol.replace("/", "_")
    filename = f"{strategy_name}_{symbol}_{timestamp}.html"

    output_path = output_dir / filename

    # 리포트 생성
    title = (
        f"{result.config.strategy_name} - {result.config.symbol} "
        f"({result.config.start_date.date()} ~ {result.config.end_date.date()})"
    )

    return generate_quantstats_report(
        returns=returns,
        benchmark_returns=benchmark_returns,
        output_path=output_path,
        title=title,
    )


def _import_quantstats() -> Any:
    """QuantStats 모듈 임포트 (지연 로딩).

    Returns:
        quantstats 모듈

    Raises:
        ImportError: QuantStats가 설치되지 않은 경우
    """
    try:
        import quantstats as qs  # type: ignore[import-not-found]
    except ImportError as e:
        msg = "QuantStats is required. Install with: pip install quantstats"
        raise ImportError(msg) from e
    return qs


def _prepare_returns_for_quantstats(returns: pd.Series) -> pd.Series:
    """QuantStats용 수익률 데이터 정제.

    Args:
        returns: 원본 수익률 시리즈

    Returns:
        정제된 수익률 시리즈
    """
    # Decimal → float 변환 (QuantStats는 float만 지원)
    if returns.dtype == object:
        returns = returns.astype(float)

    # NaN 제거
    clean = returns.dropna()

    # 무한대 값 제거
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()

    # 인덱스가 DatetimeIndex인지 확인
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean.index)

    # timezone 제거 (QuantStats는 tz-naive 인덱스 필요)
    if clean.index.tz is not None:
        clean.index = clean.index.tz_localize(None)

    # 정렬
    clean = clean.sort_index()

    return clean


def print_performance_summary(
    result: BacktestResult,
    show_trades: bool = False,
) -> None:
    """성과 요약 콘솔 출력.

    Args:
        result: BacktestResult 모델
        show_trades: 개별 거래 표시 여부
    """
    metrics = result.metrics
    config = result.config
    benchmark = result.benchmark

    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULT: {config.strategy_name}")
    print("=" * 60)
    print(f"  Symbol: {config.symbol} | Timeframe: {config.timeframe}")
    print(f"  Period: {config.start_date.date()} ~ {config.end_date.date()}")
    print(f"  Initial Capital: ${config.initial_capital:,.2f}")
    print("-" * 60)
    print("  PERFORMANCE METRICS")
    print("-" * 60)
    print(f"  Total Return:    {metrics.total_return:+.2f}%")
    print(f"  CAGR:            {metrics.cagr:+.2f}%")
    print(f"  Sharpe Ratio:    {metrics.sharpe_ratio:.2f}")
    if metrics.sortino_ratio:
        print(f"  Sortino Ratio:   {metrics.sortino_ratio:.2f}")
    print(f"  Max Drawdown:    {metrics.max_drawdown:.2f}%")
    print("-" * 60)
    print("  TRADE STATISTICS")
    print("-" * 60)
    print(f"  Total Trades:    {metrics.total_trades}")
    print(f"  Win Rate:        {metrics.win_rate:.1f}%")
    print(f"  Winning Trades:  {metrics.winning_trades}")
    print(f"  Losing Trades:   {metrics.losing_trades}")
    if metrics.profit_factor:
        print(f"  Profit Factor:   {metrics.profit_factor:.2f}")

    if benchmark:
        print("-" * 60)
        print("  BENCHMARK COMPARISON")
        print("-" * 60)
        print(f"  {benchmark.benchmark_name}: {benchmark.benchmark_return:+.2f}%")
        print(f"  Alpha:           {benchmark.alpha:+.2f}%")
        if benchmark.beta:
            print(f"  Beta:            {benchmark.beta:.2f}")

    print("-" * 60)
    passed = result.passed_minimum_criteria()
    status = "PASSED" if passed else "FAILED"
    print(f"  Minimum Criteria: {status}")
    print("=" * 60 + "\n")

    if show_trades and result.trades:
        print("\n  TRADE HISTORY (Last 10)")
        print("-" * 60)
        for trade in result.trades[-10:]:
            direction_symbol = "↗" if trade.direction == "LONG" else "↘"
            pnl_str = f"{trade.pnl_pct:+.2f}%" if trade.pnl_pct else "Open"
            print(f"  {direction_symbol} {trade.entry_time.date()} | {pnl_str}")
