"""VectorBT Backtest Engine.

이 모듈은 VectorBT를 사용하여 전략을 백테스트하는 엔진을 제공합니다.
Clean Architecture 원칙에 따라 Stateless 실행자로 설계되었습니다.

Design Principles:
    - Stateless: Engine은 상태를 가지지 않음
    - Single Responsibility: 실행만 담당, 분석은 PerformanceAnalyzer로 위임
    - Dependency Injection: 모든 의존성은 BacktestRequest로 주입

Rules Applied:
    - #26 VectorBT Standards: Broadcasting, fees, freq
    - #12 Data Engineering: Vectorization
    - #15 Logging Standards: Loguru, structured logging
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.analyzer import PerformanceAnalyzer
from src.backtest.request import BacktestRequest
from src.data.market_data import MarketDataSet
from src.models.backtest import BacktestConfig, BacktestResult
from src.portfolio.portfolio import Portfolio
from src.strategy.base import BaseStrategy

# 전략 생성 (파라미터 주입) - run_parameter_sweep용
from src.strategy.tsmom import TSMOMConfig, TSMOMStrategy


class BacktestEngine:
    """VectorBT 기반 백테스트 엔진 (Stateless).

    BacktestRequest를 받아 VectorBT로 시뮬레이션을 수행합니다.
    성과 분석은 PerformanceAnalyzer에 위임합니다.

    Example:
        >>> from src.backtest import BacktestEngine, BacktestRequest
        >>> from src.data import MarketDataService, MarketDataRequest
        >>> from src.portfolio import Portfolio
        >>> from src.strategy.tsmom import TSMOMStrategy
        >>>
        >>> # 데이터 로드
        >>> data = MarketDataService().get(MarketDataRequest(...))
        >>>
        >>> # 백테스트 요청 생성
        >>> request = BacktestRequest(
        ...     data=data,
        ...     strategy=TSMOMStrategy(),
        ...     portfolio=Portfolio.create(initial_capital=10000),
        ... )
        >>>
        >>> # 실행
        >>> result = BacktestEngine().run(request)
        >>> print(result.metrics.sharpe_ratio)
    """

    def run(self, request: BacktestRequest) -> BacktestResult:
        """백테스트 실행.

        Args:
            request: 백테스트 요청 (데이터, 전략, 포트폴리오, 분석기)

        Returns:
            BacktestResult with metrics, trades, etc.

        Raises:
            ImportError: VectorBT가 설치되지 않은 경우
            ValueError: 데이터 검증 실패 시
        """
        logger.debug("=" * 60)
        logger.debug("BacktestEngine.run() 시작")
        logger.debug(f"  Request: {request}")

        try:
            import vectorbt as vbt  # type: ignore[import-not-found]  # noqa: PLC0415

            logger.debug(f"  VectorBT version: {vbt.__version__}")
        except ImportError as e:
            msg = "VectorBT is required for backtesting. Install with: pip install vectorbt"
            raise ImportError(msg) from e

        # 요청에서 컴포넌트 추출
        data = request.data
        strategy = request.strategy
        portfolio = request.portfolio
        analyzer = request.analyzer or PerformanceAnalyzer()

        logger.debug("[1/5] 컴포넌트 추출 완료")
        logger.debug(
            f"  - Data: {data.symbol} {data.timeframe} ({data.periods} periods)"
        )
        logger.debug(f"  - Strategy: {strategy.name}")
        logger.debug(
            f"  - Portfolio: capital=${portfolio.initial_capital:,.0f}, leverage_cap={portfolio.config.max_leverage_cap}x"
        )

        # 전략 실행 (전처리 + 시그널 생성)
        logger.debug("[2/5] 전략 실행 (전처리 + 시그널 생성)...")

        processed_df, signals = strategy.run(data.ohlcv)

        logger.debug(
            f"  - 전처리된 데이터: {len(processed_df)} rows, columns: {list(processed_df.columns)}"
        )
        logger.debug(
            f"  - 시그널: entries={signals.entries.sum()}, direction range=[{signals.direction.min()}, {signals.direction.max()}]"
        )

        # VectorBT Portfolio 생성
        logger.debug(
            f"[3/5] VectorBT Portfolio 생성 (mode={portfolio.config.execution_mode})..."
        )
        vbt_portfolio = self._create_vbt_portfolio(
            vbt=vbt,
            df=processed_df,
            signals=signals,
            portfolio=portfolio,
            freq=data.freq,
        )
        logger.debug(f"  - VBT Portfolio 생성 완료: {type(vbt_portfolio).__name__}")

        # 성과 분석 (PerformanceAnalyzer에 위임)
        logger.debug("[4/5] 성과 분석 (PerformanceAnalyzer에 위임)...")
        metrics = analyzer.analyze(vbt_portfolio)
        benchmark = analyzer.compare_benchmark(vbt_portfolio, data.ohlcv, data.symbol)
        trades = analyzer.extract_trades(vbt_portfolio, data.symbol)
        logger.debug(
            f"  - Metrics: Sharpe={metrics.sharpe_ratio:.2f}, Return={metrics.total_return:.2f}%"
        )
        logger.debug(f"  - Benchmark Alpha: {benchmark.alpha:.2f}%")
        logger.debug(f"  - Trades: {len(trades)} closed trades")

        # 설정 기록
        logger.debug("[5/5] 결과 조립...")
        config = BacktestConfig(
            strategy_name=strategy.name,
            symbol=data.symbol,
            timeframe=data.timeframe,
            start_date=data.start,
            end_date=data.end,
            initial_capital=portfolio.initial_capital,
            maker_fee=portfolio.config.cost_model.maker_fee,
            taker_fee=portfolio.config.cost_model.taker_fee,
            slippage=portfolio.config.cost_model.slippage,
            strategy_params=strategy.params,
        )

        logger.debug("BacktestEngine.run() 완료")
        logger.debug("=" * 60)

        # #region agent log
        import json as _json

        _f = open("/Users/user/Project/mc-coin-bot/.cursor/debug.log", "a")
        _f.write(
            _json.dumps(
                {
                    "location": "engine.py:run:final_result",
                    "message": "Backtest final results",
                    "data": {
                        "total_return": metrics.total_return,
                        "sharpe_ratio": metrics.sharpe_ratio,
                        "max_drawdown": metrics.max_drawdown,
                        "total_trades": metrics.total_trades,
                        "win_rate": metrics.win_rate,
                        "benchmark_alpha": benchmark.alpha,
                        "total_entries": int(signals.entries.sum()),
                        "symbol": data.symbol,
                        "periods": data.periods,
                    },
                    "timestamp": __import__("time").time(),
                    "sessionId": "debug-session",
                    "hypothesisId": "ALL",
                }
            )
            + "\n"
        )
        _f.close()
        # #endregion
        return BacktestResult(
            config=config,
            metrics=metrics,
            benchmark=benchmark,
            trades=trades,
        )

    def run_with_returns(
        self,
        request: BacktestRequest,
    ) -> tuple[BacktestResult, pd.Series, pd.Series]:  # type: ignore[type-arg]
        """백테스트 실행 + 수익률 시리즈 반환.

        run()과 동일하지만 QuantStats 리포트 생성을 위한
        수익률 시리즈도 함께 반환합니다.

        Args:
            request: 백테스트 요청

        Returns:
            (BacktestResult, strategy_returns, benchmark_returns) 튜플

        Example:
            >>> result, strat_ret, bench_ret = engine.run_with_returns(request)
            >>> generate_quantstats_report(strat_ret, bench_ret)
        """
        try:
            import vectorbt as vbt  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as e:
            msg = "VectorBT is required for backtesting. Install with: pip install vectorbt"
            raise ImportError(msg) from e

        # 요청에서 컴포넌트 추출
        data = request.data
        strategy = request.strategy
        portfolio = request.portfolio
        analyzer = request.analyzer or PerformanceAnalyzer()

        # 전략 실행 (전처리 + 시그널 생성)
        processed_df, signals = strategy.run(data.ohlcv)

        # VectorBT Portfolio 생성
        vbt_portfolio = self._create_vbt_portfolio(
            vbt=vbt,
            df=processed_df,
            signals=signals,
            portfolio=portfolio,
            freq=data.freq,
        )

        # 성과 분석
        metrics = analyzer.analyze(vbt_portfolio)
        benchmark = analyzer.compare_benchmark(vbt_portfolio, data.ohlcv, data.symbol)
        trades = analyzer.extract_trades(vbt_portfolio, data.symbol)

        # 설정 기록
        config = BacktestConfig(
            strategy_name=strategy.name,
            symbol=data.symbol,
            timeframe=data.timeframe,
            start_date=data.start,
            end_date=data.end,
            initial_capital=portfolio.initial_capital,
            maker_fee=portfolio.config.cost_model.maker_fee,
            taker_fee=portfolio.config.cost_model.taker_fee,
            slippage=portfolio.config.cost_model.slippage,
            strategy_params=strategy.params,
        )

        result = BacktestResult(
            config=config,
            metrics=metrics,
            benchmark=benchmark,
            trades=trades,
        )

        # 수익률 시리즈 생성 (QuantStats용)
        strategy_returns, benchmark_returns = analyzer.get_returns_series(
            vbt_portfolio, data.ohlcv, data.symbol
        )

        return result, strategy_returns, benchmark_returns

    def _create_vbt_portfolio(
        self,
        vbt: Any,
        df: pd.DataFrame,
        signals: Any,
        portfolio: Portfolio,
        freq: str,
    ) -> Any:
        """VectorBT Portfolio 생성 (execution_mode에 따라 라우팅).

        Args:
            vbt: VectorBT 모듈
            df: 전처리된 DataFrame
            signals: 전략 시그널 (entries, exits, direction, strength)
            portfolio: 포트폴리오 객체
            freq: 데이터 주기 (VectorBT용)

        Returns:
            vbt.Portfolio 인스턴스
        """
        if portfolio.config.execution_mode == "orders":
            return self._create_portfolio_from_orders(vbt, df, signals, portfolio, freq)
        return self._create_portfolio_from_signals(vbt, df, signals, portfolio, freq)

    def _create_portfolio_from_orders(
        self,
        vbt: Any,
        df: pd.DataFrame,
        signals: Any,
        portfolio: Portfolio,
        freq: str,
    ) -> Any:
        """VectorBT Portfolio 생성 (from_orders - 연속 리밸런싱).

        VW-TSMOM과 같이 매 봉마다 목표 비중(target_weights)에 맞춰
        리밸런싱이 필요한 전략에 사용합니다.

        Args:
            vbt: VectorBT 모듈
            df: 전처리된 DataFrame
            signals: 전략 시그널
            portfolio: 포트폴리오 객체
            freq: 데이터 주기

        Returns:
            vbt.Portfolio 인스턴스
        """
        pm = portfolio.config

        # 1. target_weights 계산 (strength가 이미 direction * vol_scalar)
        target_weights: pd.Series = signals.strength.copy()

        # 디버그: Raw target weights (레버리지 클램핑 전)
        valid_weights = target_weights.dropna()
        if len(valid_weights) > 0:
            logger.info(
                f"Raw Target Weights | Range: [{valid_weights.min():.2f}, {valid_weights.max():.2f}], Mean: {valid_weights.mean():.2f}, Std: {valid_weights.std():.2f}",
            )

        # 2. max_leverage_cap 적용
        weights_before_cap = target_weights.copy()
        target_weights = target_weights.clip(
            lower=-pm.max_leverage_cap,
            upper=pm.max_leverage_cap,
        )

        # 디버그: 레버리지 클램핑 효과
        capped_count = (weights_before_cap.abs() > pm.max_leverage_cap).sum()
        if capped_count > 0:
            logger.warning(
                f"Leverage Capping | {capped_count} signals exceeded {pm.max_leverage_cap}x limit and were capped",
            )
        # #region agent log
        import json as _json

        _f = open("/Users/user/Project/mc-coin-bot/.cursor/debug.log", "a")
        _f.write(
            _json.dumps(
                {
                    "location": "engine.py:_create_portfolio_from_orders:leverage_cap",
                    "message": "Leverage capping stats",
                    "data": {
                        "max_leverage_cap": pm.max_leverage_cap,
                        "capped_count": int(capped_count),
                        "weights_before_cap_max": float(weights_before_cap.abs().max())
                        if len(weights_before_cap.dropna()) > 0
                        else 0.0,
                        "weights_after_cap_max": float(target_weights.abs().max())
                        if len(target_weights.dropna()) > 0
                        else 0.0,
                    },
                    "timestamp": __import__("time").time(),
                    "sessionId": "debug-session",
                    "hypothesisId": "H5",
                }
            )
            + "\n"
        )
        _f.close()
        # #endregion
        # 3. rebalance_threshold 적용 (거래 비용 최적화)
        weights_before_threshold = target_weights.copy()
        target_weights = self._apply_rebalance_threshold(
            target_weights,
            pm.rebalance_threshold,
        )

        # 디버그: Rebalance threshold 효과
        num_before = weights_before_threshold.notna().sum()
        num_after = target_weights.notna().sum()
        filtered_pct = (1 - num_after / num_before) * 100 if num_before > 0 else 0
        logger.info(
            f"Rebalance Threshold Effect | Before: {num_before} signals, After: {num_after} orders (Filtered: {num_before - num_after}, {filtered_pct:.1f}%)",
        )
        # #region agent log
        import json as _json

        _valid_weights = target_weights.dropna()
        _f = open("/Users/user/Project/mc-coin-bot/.cursor/debug.log", "a")
        _f.write(
            _json.dumps(
                {
                    "location": "engine.py:_create_portfolio_from_orders:rebalance",
                    "message": "Rebalance threshold effect",
                    "data": {
                        "rebalance_threshold": pm.rebalance_threshold,
                        "signals_before": int(num_before),
                        "orders_after": int(num_after),
                        "filtered_by_threshold": int(num_before - num_after),
                        "filtered_pct": float(filtered_pct),
                        "final_weights_mean": float(_valid_weights.mean())
                        if len(_valid_weights) > 0
                        else 0.0,
                        "final_weights_abs_mean": float(_valid_weights.abs().mean())
                        if len(_valid_weights) > 0
                        else 0.0,
                        "max_leverage_cap": pm.max_leverage_cap,
                    },
                    "timestamp": __import__("time").time(),
                    "sessionId": "debug-session",
                    "hypothesisId": "H4",
                }
            )
            + "\n"
        )
        _f.close()
        # #endregion
        # 4. price 결정 (next_open 또는 close)
        price = df["open"] if pm.price_type == "next_open" else df["close"]

        # 5. Portfolio 생성
        vbt_portfolio = vbt.Portfolio.from_orders(
            close=df["close"],
            size=target_weights,
            size_type="targetpercent",
            direction="both",
            price=price,
            fees=pm.cost_model.effective_fee,
            slippage=pm.cost_model.slippage,
            init_cash=portfolio.initial_capital_float,
            freq=freq,
        )

        return vbt_portfolio

    def _create_portfolio_from_signals(
        self,
        vbt: Any,
        df: pd.DataFrame,
        signals: Any,
        portfolio: Portfolio,
        freq: str,
    ) -> Any:
        """VectorBT Portfolio 생성 (from_signals - 이벤트 기반).

        단순한 entry/exit 시그널에 반응하는 전략에 사용합니다.

        Args:
            vbt: VectorBT 모듈
            df: 전처리된 DataFrame
            signals: 전략 시그널
            portfolio: 포트폴리오 객체
            freq: 데이터 주기

        Returns:
            vbt.Portfolio 인스턴스
        """
        pm_config = portfolio.config
        vbt_params = pm_config.to_vbt_params()

        # Long/Short 진입 시그널 분리
        long_entries = signals.entries & (signals.direction == 1)
        short_entries = signals.entries & (signals.direction == -1)

        # Long/Short 청산 시그널
        prev_direction = signals.direction.shift(1).fillna(0)
        long_exits = (signals.direction != 1) & (prev_direction == 1)
        short_exits = (signals.direction != -1) & (prev_direction == -1)

        # Portfolio 생성
        vbt_portfolio = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            size=np.inf,
            upon_opposite_entry=pm_config.upon_opposite_entry,
            accumulate=pm_config.accumulate,
            init_cash=portfolio.initial_capital_float,
            freq=freq,
            **vbt_params,
        )

        return vbt_portfolio

    def _apply_rebalance_threshold(
        self,
        target_weights: pd.Series,  # type: ignore[type-arg]
        threshold: float,
    ) -> pd.Series:  # type: ignore[type-arg]
        """리밸런싱 임계값 적용 (거래 비용 최적화).

        목표 비중의 변화량이 임계값 미만이면 np.nan으로 설정하여
        VectorBT가 해당 캔들에서 주문을 생성하지 않도록 합니다.

        Args:
            target_weights: 목표 비중 시리즈
            threshold: 리밸런싱 임계값 (예: 0.05 = 5%)

        Returns:
            임계값이 적용된 목표 비중 시리즈
        """
        if threshold <= 0:
            return target_weights

        result = pd.Series(np.nan, index=target_weights.index)
        last_executed_weight = 0.0

        for i in range(len(target_weights)):
            current_target = target_weights.iloc[i]
            change = abs(current_target - last_executed_weight)

            if change >= threshold or (
                last_executed_weight == 0 and current_target != 0
            ):
                result.iloc[i] = current_target
                last_executed_weight = current_target

        return result


def run_parameter_sweep(
    strategy_class: type[BaseStrategy],
    data: MarketDataSet,
    param_grid: dict[str, list[Any]],
    portfolio: Portfolio,
    top_n: int = 10,
) -> pd.DataFrame:
    """파라미터 스윕 실행.

    여러 파라미터 조합으로 백테스트를 실행하고 결과를 비교합니다.

    Args:
        strategy_class: 전략 클래스 (BaseStrategy 상속)
        data: MarketDataSet 객체
        param_grid: 파라미터 그리드 (예: {"lookback": [12, 24, 48]})
        portfolio: 포트폴리오 객체
        top_n: 상위 N개 결과만 반환

    Returns:
        파라미터별 성과 DataFrame (Sharpe 기준 정렬)

    Example:
        >>> from src.backtest import run_parameter_sweep
        >>> from src.portfolio import Portfolio
        >>>
        >>> results = run_parameter_sweep(
        ...     strategy_class=TSMOMStrategy,
        ...     data=market_data,
        ...     param_grid={"lookback": [12, 24, 48], "vol_target": [0.10, 0.15]},
        ...     portfolio=Portfolio.create(initial_capital=10000),
        ... )
        >>> print(results.head())
    """
    from itertools import product  # noqa: PLC0415

    from src.backtest.request import BacktestRequest  # noqa: PLC0415

    engine = BacktestEngine()
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

            request = BacktestRequest(
                data=data,
                strategy=strategy,
                portfolio=portfolio,
            )
            result = engine.run(request)

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

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("sharpe_ratio", ascending=False)

    return results_df.head(top_n)
