"""Performance Analyzer for backtest results.

이 모듈은 백테스트 성과 분석 책임을 전담합니다.
VectorBT 포트폴리오에서 성과 지표를 추출하고,
벤치마크와 비교하며, 거래 기록을 추출합니다.

Design Principles:
    - Separation of Concerns: Engine에서 분석 로직 분리
    - Stateless: 분석기 자체는 상태를 가지지 않음
    - Reusable: 백테스트, Dry Run, Live 모두에서 사용 가능

Rules Applied:
    - #10 Python Standards: Modern typing
    - #12 Data Engineering: Vectorized operations
    - #15 Logging Standards: Loguru, structured logging
    - #25 QuantStats Standards: Crypto-specific metrics
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
from loguru import logger

from src.data.market_data import MarketDataSet
from src.models.backtest import BenchmarkComparison, PerformanceMetrics, TradeRecord


class PerformanceAnalyzer:
    """성과 분석기.

    VectorBT 포트폴리오에서 성과 지표를 추출하고,
    벤치마크와 비교하며, 거래 기록을 추출합니다.

    Attributes:
        benchmark: 벤치마크 데이터셋 (선택)
        risk_free_rate: 무위험 수익률 (연간, 기본값 5%)

    Example:
        >>> analyzer = PerformanceAnalyzer()
        >>> metrics = analyzer.analyze(vbt_portfolio)
        >>> benchmark = analyzer.compare_benchmark(vbt_portfolio, data, "BTC/USDT")
        >>> trades = analyzer.extract_trades(vbt_portfolio, "BTC/USDT")
    """

    def __init__(
        self,
        benchmark: MarketDataSet | None = None,
        risk_free_rate: float = 0.05,
    ) -> None:
        """PerformanceAnalyzer 초기화.

        Args:
            benchmark: 벤치마크 데이터셋 (None이면 데이터 자체를 벤치마크로 사용)
            risk_free_rate: 무위험 수익률 (연간, 기본값 5%)
        """
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate

    def analyze(
        self,
        vbt_portfolio: Any,
    ) -> PerformanceMetrics:
        """VectorBT 포트폴리오에서 성과 지표 추출.

        Args:
            vbt_portfolio: VectorBT Portfolio 객체

        Returns:
            PerformanceMetrics 모델
        """
        logger.debug("PerformanceAnalyzer.analyze() 시작")

        stats = vbt_portfolio.stats()
        logger.debug(f"  VBT stats 추출 완료 ({len(stats)} metrics)")

        # 기본 지표 추출
        total_return = float(stats.get("Total Return [%]", 0))
        sharpe = float(stats.get("Sharpe Ratio", 0))
        max_dd = float(stats.get("Max Drawdown [%]", 0))

        logger.debug(
            f"  기본 지표: Return={total_return:.2f}%, Sharpe={sharpe:.2f}, MDD={max_dd:.2f}%"
        )

        # 거래 통계
        total_trades = int(stats.get("Total Trades", 0))
        win_rate_raw = stats.get("Win Rate [%]", 0)
        # NaN 처리: Win Rate이 NaN이면 0으로 대체
        win_rate = 0.0 if pd.isna(win_rate_raw) else float(win_rate_raw)

        # 승/패 거래 수 계산
        winning_trades = int(total_trades * win_rate / 100) if total_trades > 0 else 0
        losing_trades = total_trades - winning_trades

        logger.debug(
            f"  거래 통계: Total={total_trades}, Win={winning_trades}, Lose={losing_trades}, WinRate={win_rate:.1f}%"
        )

        # 추가 지표 (안전하게 추출)
        sortino = self._safe_get(stats, "Sortino Ratio")
        calmar = self._safe_get(stats, "Calmar Ratio")
        avg_drawdown = self._safe_get(stats, "Avg Drawdown [%]")
        profit_factor = self._safe_get(stats, "Profit Factor")
        avg_win = self._safe_get(stats, "Avg Winning Trade [%]")
        avg_loss = self._safe_get(stats, "Avg Losing Trade [%]")

        logger.debug(
            f"  추가 지표: Sortino={sortino}, Calmar={calmar}, ProfitFactor={profit_factor}"
        )

        # CAGR: VBT Annualized Return 사용 (없으면 Total Return 사용)
        cagr = self._safe_get(stats, "Annualized Return [%]") or total_return
        logger.debug(f"  CAGR: {cagr:.2f}%")

        logger.debug("PerformanceAnalyzer.analyze() 완료")

        return PerformanceMetrics(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            avg_drawdown=avg_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
        )

    def compare_benchmark(
        self,
        vbt_portfolio: Any,
        data: pd.DataFrame,
        symbol: str,
    ) -> BenchmarkComparison:
        """벤치마크 (Buy & Hold) 비교.

        Args:
            vbt_portfolio: 전략 포트폴리오
            data: OHLCV 데이터 (벤치마크로 사용)
            symbol: 심볼 이름

        Returns:
            BenchmarkComparison 모델
        """
        try:
            # Buy & Hold 수익률
            close = data["close"]
            bh_return = ((close.iloc[-1] / close.iloc[0]) - 1) * 100

            # 전략 수익률
            strategy_return = vbt_portfolio.total_return() * 100

            # Alpha (초과 수익률)
            alpha = strategy_return - bh_return

            # Beta 및 상관계수 계산
            strategy_returns = vbt_portfolio.returns()
            bh_returns = close.pct_change().dropna()

            # 인덱스 맞추기
            common_idx = strategy_returns.index.intersection(bh_returns.index)
            min_data_points = 10
            if len(common_idx) > min_data_points:
                sr = strategy_returns.loc[common_idx]
                br = bh_returns.loc[common_idx]

                correlation = float(sr.corr(br))
                beta = float(sr.cov(br) / br.var()) if br.var() != 0 else None
            else:
                correlation = None
                beta = None

            return BenchmarkComparison(
                benchmark_name=f"{symbol} Buy & Hold",
                benchmark_return=float(bh_return),
                alpha=float(alpha),
                beta=beta,
                correlation=correlation,
            )
        except Exception:
            return BenchmarkComparison(
                benchmark_name=f"{symbol} Buy & Hold",
                benchmark_return=0.0,
                alpha=0.0,
            )

    def extract_trades(
        self,
        vbt_portfolio: Any,
        symbol: str,
    ) -> tuple[TradeRecord, ...]:
        """거래 기록 추출.

        Args:
            vbt_portfolio: VectorBT Portfolio
            symbol: 심볼

        Returns:
            TradeRecord 튜플
        """
        try:
            trades_df = vbt_portfolio.trades.records_readable
            if trades_df.empty:
                return ()

            records: list[TradeRecord] = []
            for _, row in trades_df.iterrows():
                # Entry time 처리
                entry_ts = pd.Timestamp(row["Entry Timestamp"])
                if pd.isna(entry_ts):  # type: ignore[arg-type]
                    continue  # Skip invalid entries
                entry_dt_raw = entry_ts.to_pydatetime()
                entry_dt: datetime = (  # type: ignore[assignment]
                    entry_dt_raw.replace(tzinfo=UTC)
                    if entry_dt_raw.tzinfo is None
                    else entry_dt_raw
                )

                # Exit time 처리 (optional)
                exit_dt: datetime | None = None
                if pd.notna(row.get("Exit Timestamp")):  # type: ignore[arg-type]
                    exit_ts = pd.Timestamp(row["Exit Timestamp"])
                    if pd.notna(exit_ts):  # type: ignore[arg-type]
                        exit_dt_raw = exit_ts.to_pydatetime()
                        exit_dt = (  # type: ignore[assignment]
                            exit_dt_raw.replace(tzinfo=UTC)
                            if exit_dt_raw.tzinfo is None
                            else exit_dt_raw
                        )

                record = TradeRecord(
                    entry_time=entry_dt,
                    exit_time=exit_dt,
                    symbol=symbol,
                    direction="LONG"
                    if row.get("Direction", "Long") == "Long"
                    else "SHORT",
                    entry_price=Decimal(str(row["Avg Entry Price"])),
                    exit_price=Decimal(str(row["Avg Exit Price"]))
                    if pd.notna(row.get("Avg Exit Price"))
                    else None,
                    size=Decimal(str(row["Size"])),
                    pnl=Decimal(str(row["PnL"])) if pd.notna(row.get("PnL")) else None,
                    pnl_pct=float(row["Return"]) * 100
                    if pd.notna(row.get("Return"))
                    else None,
                )
                records.append(record)

            # 디버그 로깅
            long_trades = [r for r in records if r.direction == "LONG"]
            short_trades = [r for r in records if r.direction == "SHORT"]

            logger.info(
                f"Trade Summary | Total: {len(records)}, Long: {len(long_trades)}, Short: {len(short_trades)}"
            )

            return tuple(records)
        except Exception:
            return ()

    def get_returns_series(
        self,
        vbt_portfolio: Any,
        benchmark_data: pd.DataFrame,
        symbol: str,
    ) -> tuple[pd.Series, pd.Series]:  # type: ignore[type-arg]
        """QuantStats 리포트용 수익률 시리즈 생성.

        Args:
            vbt_portfolio: VectorBT Portfolio
            benchmark_data: 벤치마크 OHLCV 데이터
            symbol: 심볼

        Returns:
            (strategy_returns, benchmark_returns) 튜플
        """
        # 전략 수익률: VBT Portfolio에서 직접 추출
        strategy_returns: pd.Series = vbt_portfolio.returns()  # type: ignore[type-arg]
        strategy_returns.name = "Strategy"

        # 벤치마크 수익률: Buy & Hold
        close_series: pd.Series = benchmark_data["close"]  # type: ignore[assignment]
        benchmark_returns: pd.Series = close_series.pct_change().dropna()  # type: ignore[type-arg, assignment]
        benchmark_returns.name = f"{symbol} Buy & Hold"

        # QuantStats는 timezone-naive 인덱스를 요구함
        strat_idx = strategy_returns.index
        if isinstance(strat_idx, pd.DatetimeIndex) and strat_idx.tz is not None:
            strategy_returns = strategy_returns.copy()
            strategy_returns.index = strat_idx.tz_localize(None)

        bench_idx = benchmark_returns.index
        if isinstance(bench_idx, pd.DatetimeIndex) and bench_idx.tz is not None:
            benchmark_returns = benchmark_returns.copy()
            benchmark_returns.index = bench_idx.tz_localize(None)

        # 인덱스 정렬: 공통 기간만 사용 (QuantStats 비교 정확도 향상)
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]

        return strategy_returns, benchmark_returns

    @staticmethod
    def _safe_get(
        stats: pd.Series,  # type: ignore[type-arg]
        key: str,
        default: float | None = None,
    ) -> float | None:
        """안전하게 통계값 추출.

        Args:
            stats: VectorBT stats Series
            key: 키 이름
            default: 기본값

        Returns:
            값 또는 None
        """
        try:
            value = stats.get(key)
            if value is None or pd.isna(value):
                return default
            return float(value)
        except (KeyError, TypeError, ValueError):
            return default
