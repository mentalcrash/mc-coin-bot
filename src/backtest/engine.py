"""VectorBT Backtest Engine.

이 모듈은 VectorBT를 사용하여 전략을 백테스트하는 엔진을 제공합니다.
Broadcasting 패턴을 지원하여 대규모 파라미터 최적화가 가능합니다.

Rules Applied:
    - #26 VectorBT Standards: Broadcasting, fees, freq
    - #25 QuantStats Standards: Benchmark comparison
    - #12 Data Engineering: Vectorization
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.cost_model import CostModel
from src.models.backtest import (
    BacktestConfig,
    BacktestResult,
    BenchmarkComparison,
    PerformanceMetrics,
    TradeRecord,
)
from src.portfolio.config import PortfolioManagerConfig
from src.strategy.base import BaseStrategy

# 전략 생성 (파라미터 주입)
from src.strategy.tsmom import TSMOMConfig, TSMOMStrategy

# VectorBT is an optional dependency
# TYPE_CHECKING import removed to avoid unused import warning


class BacktestEngine:
    """VectorBT 기반 백테스트 엔진.

    전략과 데이터를 받아 VectorBT로 시뮬레이션을 수행합니다.
    PortfolioManagerConfig를 통해 집행 규칙, 리스크 가드레일, 비용을 설정합니다.

    Attributes:
        portfolio_config: 포트폴리오 매니저 설정
        cost_model: 거래 비용 모델 (portfolio_config에서 추출)
        initial_capital: 초기 자본
        freq: 데이터 주기 (연환산 계산용)

    Example:
        >>> from src.strategy.tsmom import TSMOMStrategy
        >>> from src.portfolio import PortfolioManagerConfig
        >>> # 기본 설정 사용
        >>> engine = BacktestEngine(initial_capital=10000)
        >>> result = engine.run(TSMOMStrategy(), ohlcv_df)
        >>> # 보수적 설정 사용
        >>> engine = BacktestEngine(
        ...     portfolio_config=PortfolioManagerConfig.conservative(),
        ...     initial_capital=10000,
        ... )
        >>> result = engine.run(TSMOMStrategy(), ohlcv_df)
        >>> print(result.metrics.sharpe_ratio)
    """

    def __init__(
        self,
        portfolio_config: PortfolioManagerConfig | None = None,
        initial_capital: float = 10000.0,
        freq: str = "1h",
        *,
        cost_model: CostModel | None = None,  # DEPRECATED: use portfolio_config
    ) -> None:
        """BacktestEngine 초기화.

        Args:
            portfolio_config: 포트폴리오 매니저 설정 (None이면 기본값)
            initial_capital: 초기 자본 (USD)
            freq: 데이터 주기 (VectorBT용, 예: "1h", "15m")
            cost_model: [DEPRECATED] 거래 비용 모델 - portfolio_config 사용 권장

        Note:
            cost_model 파라미터는 하위 호환성을 위해 유지됩니다.
            portfolio_config와 cost_model 모두 제공되면 portfolio_config가 우선합니다.
        """
        # portfolio_config 설정 (기본값 또는 제공된 값)
        if portfolio_config is not None:
            self.portfolio_config = portfolio_config
        elif cost_model is not None:
            # 하위 호환성: cost_model만 제공된 경우
            self.portfolio_config = PortfolioManagerConfig(cost_model=cost_model)
        else:
            self.portfolio_config = PortfolioManagerConfig()

        # cost_model은 portfolio_config에서 추출
        self.cost_model = self.portfolio_config.cost_model
        self.initial_capital = initial_capital
        self.freq = freq

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str = "BTC/USDT",
        benchmark_data: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """백테스트 실행.

        전략을 적용하고 VectorBT로 시뮬레이션합니다.

        Args:
            strategy: 전략 인스턴스 (BaseStrategy 상속)
            data: OHLCV DataFrame (DatetimeIndex 필수)
            symbol: 거래 심볼
            benchmark_data: 벤치마크 데이터 (None이면 data 사용)

        Returns:
            BacktestResult with metrics, trades, etc.

        Raises:
            ImportError: VectorBT가 설치되지 않은 경우
            ValueError: 데이터 검증 실패 시
        """
        try:
            import vectorbt as vbt  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as e:
            msg = "VectorBT is required for backtesting. Install with: pip install vectorbt"
            raise ImportError(msg) from e

        # 전략 실행 (전처리 + 시그널 생성)
        processed_df, signals = strategy.run(data)

        # VectorBT Portfolio 생성
        portfolio = self._create_portfolio(vbt, processed_df, signals)

        # 성과 지표 계산
        metrics = self._calculate_metrics(portfolio)

        # 벤치마크 비교
        benchmark = self._compare_benchmark(portfolio, benchmark_data or data, symbol)

        # 거래 기록 추출
        trades = self._extract_trades(portfolio, symbol)

        # 설정 기록
        config = BacktestConfig(
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=self.freq,
            start_date=data.index[0],  # type: ignore[arg-type]
            end_date=data.index[-1],  # type: ignore[arg-type]
            initial_capital=Decimal(str(self.initial_capital)),
            maker_fee=self.cost_model.maker_fee,
            taker_fee=self.cost_model.taker_fee,
            slippage=self.cost_model.slippage,
            strategy_params=strategy.params,
        )

        return BacktestResult(
            config=config,
            metrics=metrics,
            benchmark=benchmark,
            trades=trades,
        )

    def run_with_returns(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str = "BTC/USDT",
        benchmark_data: pd.DataFrame | None = None,
    ) -> tuple[BacktestResult, pd.Series, pd.Series]:
        """백테스트 실행 + 수익률 시리즈 반환.

        run()과 동일하지만 QuantStats 리포트 생성을 위한
        수익률 시리즈도 함께 반환합니다.

        Args:
            strategy: 전략 인스턴스 (BaseStrategy 상속)
            data: OHLCV DataFrame (DatetimeIndex 필수)
            symbol: 거래 심볼
            benchmark_data: 벤치마크 데이터 (None이면 data 사용)

        Returns:
            (BacktestResult, strategy_returns, benchmark_returns) 튜플

        Example:
            >>> result, strat_ret, bench_ret = engine.run_with_returns(strategy, data)
            >>> generate_quantstats_report(strat_ret, bench_ret)
        """
        try:
            import vectorbt as vbt  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as e:
            msg = "VectorBT is required for backtesting. Install with: pip install vectorbt"
            raise ImportError(msg) from e

        # 전략 실행 (전처리 + 시그널 생성)
        processed_df, signals = strategy.run(data)

        # VectorBT Portfolio 생성
        portfolio = self._create_portfolio(vbt, processed_df, signals)

        # 성과 지표 계산
        metrics = self._calculate_metrics(portfolio)

        # 벤치마크 비교
        benchmark_df = benchmark_data or data
        benchmark = self._compare_benchmark(portfolio, benchmark_df, symbol)

        # 거래 기록 추출
        trades = self._extract_trades(portfolio, symbol)

        # 설정 기록
        config = BacktestConfig(
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=self.freq,
            start_date=data.index[0],  # type: ignore[arg-type]
            end_date=data.index[-1],  # type: ignore[arg-type]
            initial_capital=Decimal(str(self.initial_capital)),
            maker_fee=self.cost_model.maker_fee,
            taker_fee=self.cost_model.taker_fee,
            slippage=self.cost_model.slippage,
            strategy_params=strategy.params,
        )

        result = BacktestResult(
            config=config,
            metrics=metrics,
            benchmark=benchmark,
            trades=trades,
        )

        # 수익률 시리즈 생성 (QuantStats용)
        # 전략 수익률: VBT Portfolio에서 직접 추출
        strategy_returns: pd.Series = portfolio.returns()
        strategy_returns.name = "Strategy"

        # 벤치마크 수익률: Buy & Hold
        close_series: pd.Series = benchmark_df["close"]  # type: ignore[assignment]
        benchmark_returns: pd.Series = close_series.pct_change().dropna()  # type: ignore[assignment]
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

        return result, strategy_returns, benchmark_returns

    def _create_portfolio(
        self,
        vbt: Any,  # VectorBT module
        df: pd.DataFrame,
        signals: Any,  # StrategySignals
    ) -> Any:  # vbt.Portfolio
        """VectorBT Portfolio 생성 (execution_mode에 따라 라우팅).

        PortfolioManagerConfig의 execution_mode 설정에 따라
        적절한 포트폴리오 생성 메서드를 호출합니다.

        Args:
            vbt: VectorBT 모듈
            df: 전처리된 DataFrame
            signals: 전략 시그널 (entries, exits, direction, strength)

        Returns:
            vbt.Portfolio 인스턴스
        """
        if self.portfolio_config.execution_mode == "orders":
            return self._create_portfolio_from_orders(vbt, df, signals)
        return self._create_portfolio_from_signals(vbt, df, signals)

    def _create_portfolio_from_orders(
        self,
        vbt: Any,  # VectorBT module
        df: pd.DataFrame,
        signals: Any,  # StrategySignals
    ) -> Any:  # vbt.Portfolio
        """VectorBT Portfolio 생성 (from_orders - 연속 리밸런싱).

        VW-TSMOM과 같이 매 봉마다 목표 비중(target_weights)에 맞춰
        리밸런싱이 필요한 전략에 사용합니다.

        signals.strength를 target_weights로 사용하며:
        - Look-Ahead Bias 방지를 위해 shift(1) 적용
        - max_leverage_cap으로 레버리지 클램핑
        - rebalance_threshold로 거래 비용 최적화

        Args:
            vbt: VectorBT 모듈
            df: 전처리된 DataFrame
            signals: 전략 시그널 (strength가 target_weights로 사용됨)

        Returns:
            vbt.Portfolio 인스턴스
        """
        pm = self.portfolio_config

        # 1. target_weights 계산 (strength가 이미 direction * vol_scalar)
        target_weights: pd.Series = signals.strength.copy()

        # 2. Look-Ahead Bias 방지: 현재 봉의 시그널은 다음 봉에 적용
        target_weights = pd.Series(
            target_weights.shift(1).fillna(0),
            index=target_weights.index,
        )

        # 3. max_leverage_cap 적용 (전략 요청과 무관한 시스템 상한)
        target_weights = target_weights.clip(
            lower=-pm.max_leverage_cap,
            upper=pm.max_leverage_cap,
        )

        # 4. rebalance_threshold 적용 (거래 비용 최적화)
        target_weights = self._apply_rebalance_threshold(
            target_weights,
            pm.rebalance_threshold,
        )

        # 5. price 결정 (next_open 또는 close)
        # next_open: 시그널 발생 다음 봉 시가에 체결 (Look-Ahead Bias 방지)
        price = df["open"] if pm.price_type == "next_open" else df["close"]

        # 6. Portfolio 생성
        portfolio = vbt.Portfolio.from_orders(
            close=df["close"],
            size=target_weights,
            size_type="targetpercent",
            price=price,
            fees=pm.cost_model.effective_fee,
            slippage=pm.cost_model.slippage,
            init_cash=self.initial_capital,
            freq=self.freq,
        )

        return portfolio

    def _create_portfolio_from_signals(
        self,
        vbt: Any,  # VectorBT module
        df: pd.DataFrame,
        signals: Any,  # StrategySignals
    ) -> Any:  # vbt.Portfolio
        """VectorBT Portfolio 생성 (from_signals - 이벤트 기반).

        단순한 entry/exit 시그널에 반응하는 전략에 사용합니다.
        Long과 Short 시그널을 분리하여 전달하고,
        upon_opposite_entry="reverse"로 포지션 반전을 처리합니다.

        Note: SizeType.Percent는 반전을 지원하지 않으므로,
        고정 비율 (100%)로 진입하고 반전 시 자동으로 처리합니다.

        Args:
            vbt: VectorBT 모듈
            df: 전처리된 DataFrame
            signals: 전략 시그널 (entries, exits, direction)

        Returns:
            vbt.Portfolio 인스턴스
        """
        pm_config = self.portfolio_config
        vbt_params = pm_config.to_vbt_params()

        # Long/Short 진입 시그널 분리 (entries를 direction으로 분리)
        long_entries = signals.entries & (signals.direction == 1)
        short_entries = signals.entries & (signals.direction == -1)

        # Long/Short 청산 시그널: 중립으로 전환될 때
        # direction이 0이 되는 순간 (양쪽 다 청산)
        prev_direction = signals.direction.shift(1).fillna(0)
        long_exits = (signals.direction != 1) & (prev_direction == 1)
        short_exits = (signals.direction != -1) & (prev_direction == -1)

        # Portfolio 생성 (from_signals with separate long/short)
        # size=np.inf → 가용 현금 전액 사용 (반전 지원)
        portfolio = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            size=np.inf,  # 가용 현금 전액 사용
            upon_opposite_entry=pm_config.upon_opposite_entry,
            accumulate=pm_config.accumulate,
            init_cash=self.initial_capital,
            freq=self.freq,
            **vbt_params,
        )

        return portfolio

    def _apply_rebalance_threshold(
        self,
        target_weights: pd.Series,
        threshold: float,
    ) -> pd.Series:
        """리밸런싱 임계값 적용 (거래 비용 최적화).

        목표 비중의 변화량이 임계값 미만이면 이전 값을 유지하여
        불필요한 거래를 줄입니다.

        Args:
            target_weights: 목표 비중 시리즈
            threshold: 리밸런싱 임계값 (예: 0.05 = 5%)

        Returns:
            임계값이 적용된 목표 비중 시리즈
        """
        if threshold <= 0:
            return target_weights

        result = target_weights.copy()
        prev_weight = 0.0

        # 벡터화가 어려운 로직이므로 iterative 처리
        # (성능이 중요하면 Numba로 최적화 가능)
        for i in range(len(result)):
            current_target = result.iloc[i]
            change = abs(current_target - prev_weight)

            if change < threshold:
                # 변화량이 임계값 미만이면 이전 비중 유지
                result.iloc[i] = prev_weight
            else:
                # 변화량이 임계값 이상이면 새 비중 적용
                prev_weight = current_target

        return result

    def _calculate_metrics(
        self,
        portfolio: Any,  # vbt.Portfolio
    ) -> PerformanceMetrics:
        """성과 지표 계산.

        Args:
            portfolio: VectorBT Portfolio

        Returns:
            PerformanceMetrics 모델
        """
        stats = portfolio.stats()

        # 기본 지표 추출
        total_return = float(stats.get("Total Return [%]", 0))
        sharpe = float(stats.get("Sharpe Ratio", 0))
        max_dd = float(stats.get("Max Drawdown [%]", 0))

        # 거래 통계
        total_trades = int(stats.get("Total Trades", 0))
        win_rate_raw = stats.get("Win Rate [%]", 0)
        # NaN 처리: Win Rate이 NaN이면 0으로 대체
        win_rate = 0.0 if pd.isna(win_rate_raw) else float(win_rate_raw)

        # 승/패 거래 수 계산
        winning_trades = int(total_trades * win_rate / 100) if total_trades > 0 else 0
        losing_trades = total_trades - winning_trades

        # 추가 지표 (안전하게 추출)
        sortino = self._safe_get(stats, "Sortino Ratio")
        calmar = self._safe_get(stats, "Calmar Ratio")
        avg_drawdown = self._safe_get(stats, "Avg Drawdown [%]")
        profit_factor = self._safe_get(stats, "Profit Factor")
        avg_win = self._safe_get(stats, "Avg Winning Trade [%]")
        avg_loss = self._safe_get(stats, "Avg Losing Trade [%]")

        # CAGR: VBT Annualized Return 사용 (없으면 Total Return 사용)
        cagr = self._safe_get(stats, "Annualized Return [%]") or total_return

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

    def _compare_benchmark(
        self,
        portfolio: Any,  # vbt.Portfolio
        data: pd.DataFrame,
        symbol: str,
    ) -> BenchmarkComparison:
        """벤치마크 (Buy & Hold) 비교.

        Args:
            portfolio: 전략 포트폴리오
            data: OHLCV 데이터
            symbol: 심볼 이름

        Returns:
            BenchmarkComparison 모델
        """
        try:
            # Buy & Hold 수익률
            close = data["close"]
            bh_return = ((close.iloc[-1] / close.iloc[0]) - 1) * 100

            # 전략 수익률
            strategy_return = portfolio.total_return() * 100

            # Alpha (초과 수익률)
            alpha = strategy_return - bh_return

            # Beta 및 상관계수 계산
            strategy_returns = portfolio.returns()
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

    def _extract_trades(
        self,
        portfolio: Any,  # vbt.Portfolio
        symbol: str,
    ) -> tuple[TradeRecord, ...]:
        """거래 기록 추출.

        Args:
            portfolio: VectorBT Portfolio
            symbol: 심볼

        Returns:
            TradeRecord 튜플
        """
        try:
            trades_df = portfolio.trades.records_readable
            if trades_df.empty:
                return ()

            records = []
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
                    entry_price=Decimal(str(row["Entry Price"])),
                    exit_price=Decimal(str(row["Exit Price"]))
                    if pd.notna(row.get("Exit Price"))
                    else None,
                    size=Decimal(str(row["Size"])),
                    pnl=Decimal(str(row["PnL"])) if pd.notna(row.get("PnL")) else None,
                    pnl_pct=float(row["Return [%]"])
                    if pd.notna(row.get("Return [%]"))
                    else None,
                )
                records.append(record)

            return tuple(records)
        except Exception:
            return ()

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


def run_parameter_sweep(  # noqa: PLR0913
    strategy_class: type[BaseStrategy],
    data: pd.DataFrame,
    param_grid: dict[str, list[Any]],
    portfolio_config: PortfolioManagerConfig | None = None,
    initial_capital: float = 10000.0,
    freq: str = "1h",
    symbol: str = "BTC/USDT",
    top_n: int = 10,
    *,
    cost_model: CostModel | None = None,  # DEPRECATED: use portfolio_config
) -> pd.DataFrame:
    """파라미터 스윕 실행.

    여러 파라미터 조합으로 백테스트를 실행하고 결과를 비교합니다.
    VectorBT의 Broadcasting을 활용하여 효율적으로 처리합니다.

    Args:
        strategy_class: 전략 클래스 (BaseStrategy 상속)
        data: OHLCV DataFrame
        param_grid: 파라미터 그리드 (예: {"lookback": [12, 24, 48]})
        portfolio_config: 포트폴리오 매니저 설정
        initial_capital: 초기 자본
        freq: 데이터 주기
        symbol: 심볼
        top_n: 상위 N개 결과만 반환
        cost_model: [DEPRECATED] 거래 비용 모델 - portfolio_config 사용 권장

    Returns:
        파라미터별 성과 DataFrame (Sharpe 기준 정렬)

    Example:
        >>> results = run_parameter_sweep(
        ...     TSMOMStrategy,
        ...     data,
        ...     param_grid={"lookback": [12, 24, 48], "vol_target": [0.10, 0.15, 0.20]},
        ...     portfolio_config=PortfolioManagerConfig.conservative(),
        ... )
        >>> print(results.head())
    """
    from itertools import product  # noqa: PLC0415

    # 하위 호환성: cost_model이 제공되면 portfolio_config로 변환
    if portfolio_config is None and cost_model is not None:
        portfolio_config = PortfolioManagerConfig(cost_model=cost_model)

    engine = BacktestEngine(
        portfolio_config=portfolio_config,
        initial_capital=initial_capital,
        freq=freq,
    )

    results = []
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for combination in product(*param_values):
        params = dict(zip(param_names, combination, strict=True))

        try:
            if strategy_class == TSMOMStrategy:
                config = TSMOMConfig(**params)
                strategy = TSMOMStrategy(config)
            else:
                strategy = strategy_class(**params)  # type: ignore[call-arg]

            result = engine.run(strategy, data, symbol)

            results.append(
                {
                    **params,
                    "sharpe_ratio": result.metrics.sharpe_ratio,
                    "total_return": result.metrics.total_return,
                    "max_drawdown": result.metrics.max_drawdown,
                    "win_rate": result.metrics.win_rate,
                    "total_trades": result.metrics.total_trades,
                    "cagr": result.metrics.cagr,
                }
            )
        except Exception as e:
            results.append(
                {
                    **params,
                    "sharpe_ratio": np.nan,
                    "total_return": np.nan,
                    "max_drawdown": np.nan,
                    "win_rate": np.nan,
                    "total_trades": 0,
                    "cagr": np.nan,
                    "error": str(e),
                }
            )

    # DataFrame으로 변환 및 정렬
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("sharpe_ratio", ascending=False)

    return results_df.head(top_n)
