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

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.backtest.validation import ValidationResult
import numpy.typing as npt
import pandas as pd
from loguru import logger
from numba import njit

from src.backtest.analyzer import PerformanceAnalyzer
from src.backtest.request import BacktestRequest, MultiAssetBacktestRequest
from src.data.market_data import MarketDataSet
from src.models.backtest import (
    BacktestConfig,
    BacktestResult,
    MultiAssetBacktestResult,
    MultiAssetConfig,
    PerformanceMetrics,
)
from src.orchestrator.asset_allocator import AssetAllocationConfig, IntraPodAllocator
from src.portfolio.portfolio import Portfolio
from src.strategy.base import BaseStrategy

# =============================================================================
# Numba 최적화 함수들 (모듈 레벨에 정의 - JIT 컴파일 캐싱)
# =============================================================================


@njit(cache=True)  # type: ignore[misc]
def apply_stop_loss_to_weights(
    weights: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    stop_loss_pct: float,
    use_intrabar: bool = False,
    execution_prices: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64),
) -> npt.NDArray[np.float64]:
    """목표 비중 배열에 손절매 로직을 적용합니다 (Long/Short 모두 지원).

    진입가 대비 특정 비율 이상 손실 시 비중을 0으로 강제합니다.

    Args:
        weights: 목표 비중 배열 (양수=롱, 음수=숏)
        close: 종가 배열 (손절 체크용)
        high: 고가 배열 (숏 손절 체크용, use_intrabar=True)
        low: 저가 배열 (롱 손절 체크용, use_intrabar=True)
        stop_loss_pct: 손절 비율 (예: 0.10 = 10%)
        use_intrabar: True면 Low/High로 손절 체크, False면 Close 기준
        execution_prices: 실제 체결 가격 배열 (진입가 기록용, 비어있으면 close 사용)

    Returns:
        손절이 적용된 비중 배열
    """
    out_weights = weights.copy()
    entry_price = 0.0
    position_direction = 0  # 1=롱, -1=숏, 0=중립
    use_exec = len(execution_prices) == len(weights)

    for i in range(len(weights)):
        current_weight = weights[i]

        # 1. 포지션이 없다가 새로 생기는 경우 (진입)
        if position_direction == 0 and current_weight != 0:
            entry_price = execution_prices[i] if use_exec else close[i]
            position_direction = 1 if current_weight > 0 else -1

        # 2. 이미 포지션이 있는 경우
        elif position_direction != 0:
            # 전략에 의해 청산된 경우 (신호가 0이 됨)
            if current_weight == 0:
                position_direction = 0
                entry_price = 0.0
                continue

            # 방향 전환된 경우 (롱→숏 또는 숏→롱)
            new_direction = 1 if current_weight > 0 else -1
            if new_direction != position_direction:
                # 새 포지션으로 진입가 갱신
                entry_price = execution_prices[i] if use_exec else close[i]
                position_direction = new_direction
                continue

            # 손절 조건 체크 (use_intrabar: Low/High, 기본: Close)
            stop_triggered = False
            long_check = low[i] if use_intrabar else close[i]
            short_check = high[i] if use_intrabar else close[i]

            if position_direction == 1:  # 롱 포지션
                if long_check < entry_price * (1 - stop_loss_pct):
                    stop_triggered = True
            # 숏 포지션
            elif short_check > entry_price * (1 + stop_loss_pct):
                stop_triggered = True

            if stop_triggered:
                out_weights[i] = 0.0
                position_direction = 0
                entry_price = 0.0

    return out_weights


@njit(cache=True)  # type: ignore[misc]
def apply_trailing_stop_to_weights(  # noqa: PLR0912 - Numba @njit state machine; branching cannot be simplified
    weights: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    atr: npt.NDArray[np.float64],
    atr_multiplier: float,
    execution_prices: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64),
) -> npt.NDArray[np.float64]:
    """목표 비중 배열에 Trailing Stop 로직을 적용합니다.

    진입 후 최고/최저가 대비 ATR * multiplier만큼 역행하면 청산합니다.

    동작 방식:
        - Long: 진입 후 최고가(highest_since_entry) 추적
                현재 가격이 최고가 - (ATR * multiplier) 아래로 하락 시 청산
        - Short: 진입 후 최저가(lowest_since_entry) 추적
                현재 가격이 최저가 + (ATR * multiplier) 위로 상승 시 청산

    Args:
        weights: 목표 비중 배열 (양수=롱, 음수=숏)
        close: 종가 배열
        high: 고가 배열
        low: 저가 배열
        atr: ATR 배열 (Average True Range)
        atr_multiplier: ATR 배수 (예: 2.0 = 2 ATR)
        execution_prices: 실제 체결 가격 배열 (비어있으면 high/low 사용)

    Returns:
        Trailing Stop이 적용된 비중 배열
    """
    out_weights = weights.copy()
    position_direction = 0  # 1=롱, -1=숏, 0=중립
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    use_exec = len(execution_prices) == len(weights)

    for i in range(len(weights)):
        current_weight = weights[i]
        current_atr = atr[i]

        # ATR이 NaN이면 스킵 (초기 워밍업 기간)
        if np.isnan(current_atr):
            continue

        # 1. 포지션이 없다가 새로 생기는 경우 (진입)
        if position_direction == 0 and current_weight != 0:
            position_direction = 1 if current_weight > 0 else -1
            if position_direction == 1:
                exec_p = execution_prices[i] if use_exec else high[i]
                highest_since_entry = max(high[i], exec_p)
            else:
                exec_p = execution_prices[i] if use_exec else low[i]
                lowest_since_entry = min(low[i], exec_p)

        # 2. 이미 포지션이 있는 경우
        elif position_direction != 0:
            # 전략에 의해 청산된 경우 (신호가 0이 됨)
            if current_weight == 0:
                position_direction = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue

            # 방향 전환된 경우 (롱→숏 또는 숏→롱)
            new_direction = 1 if current_weight > 0 else -1
            if new_direction != position_direction:
                position_direction = new_direction
                if new_direction == 1:
                    exec_p = execution_prices[i] if use_exec else high[i]
                    highest_since_entry = max(high[i], exec_p)
                    lowest_since_entry = 0.0
                else:
                    exec_p = execution_prices[i] if use_exec else low[i]
                    lowest_since_entry = min(low[i], exec_p)
                    highest_since_entry = 0.0
                continue

            # 최고/최저가 갱신
            if position_direction == 1:
                highest_since_entry = max(highest_since_entry, high[i])
            elif low[i] < lowest_since_entry:
                lowest_since_entry = low[i]

            # Trailing Stop 조건 체크
            stop_triggered = False
            trailing_distance = current_atr * atr_multiplier

            if position_direction == 1:  # 롱 포지션
                # 장중 최저가가 최고가 - trailing_distance 아래로 하락
                stop_triggered = low[i] < highest_since_entry - trailing_distance
            elif high[i] > lowest_since_entry + trailing_distance:  # 숏 포지션
                # 장중 최고가가 최저가 + trailing_distance 위로 상승
                stop_triggered = True

            if stop_triggered:
                out_weights[i] = 0.0
                position_direction = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0

    return out_weights


@njit(cache=True)  # type: ignore[misc]
def _numba_single_bar_iv(  # noqa: PLR0912 - Numba @njit bar-by-bar IV; branching cannot be simplified
    returns_matrix: npt.NDArray[np.float64],
    bar_idx: int,
    vol_lookback: int,
    n_symbols: int,
) -> npt.NDArray[np.float64]:
    """단일 bar에서 IV weight 계산.

    Args:
        returns_matrix: (n_bars, n_symbols) 수익률 행렬 (NaN 포함)
        bar_idx: 현재 bar 인덱스
        vol_lookback: 변동성 계산 윈도우
        n_symbols: 심볼 수

    Returns:
        (n_symbols,) IV weights (합계 1.0). 데이터 부족 시 EW.
    """
    weights = np.empty(n_symbols, dtype=np.float64)
    ew = 1.0 / n_symbols

    # Python slow path에서는 bar i의 returns_df[i]를 history에 추가한 뒤 on_bar() 호출.
    # returns_matrix는 pct_change().shift(1) 결과이므로 row i = bar i-1→i의 return.
    # bar i 시점 history = rows 0..i (inclusive) → end = bar_idx + 1
    end = bar_idx + 1  # inclusive → Python slice exclusive
    start = max(0, end - vol_lookback)

    min_samples = 2  # Numba에서 모듈 상수 사용 불가 → 로컬 변수
    if end - start < min_samples:
        for j in range(n_symbols):
            weights[j] = ew
        return weights

    inv_vols = np.empty(n_symbols, dtype=np.float64)
    min_vol = 1e-8

    for j in range(n_symbols):
        # 유효 데이터 수 계산
        vals = np.empty(end - start, dtype=np.float64)
        count = 0
        for k in range(start, end):
            v = returns_matrix[k, j]
            if not np.isnan(v):
                vals[count] = v
                count += 1

        if count < min_samples:
            # 데이터 부족 → EW fallback
            for jj in range(n_symbols):
                weights[jj] = ew
            return weights

        # std (ddof=1)
        mean = 0.0
        for m in range(count):
            mean += vals[m]
        mean /= count

        var_sum = 0.0
        for m in range(count):
            diff = vals[m] - mean
            var_sum += diff * diff
        std = (var_sum / (count - 1)) ** 0.5
        inv_vols[j] = 1.0 / max(std, min_vol)

    total = 0.0
    for j in range(n_symbols):
        total += inv_vols[j]

    clamp_tol = 1e-10  # Numba에서 모듈 상수 사용 불가
    if total < clamp_tol:
        for j in range(n_symbols):
            weights[j] = ew
    else:
        for j in range(n_symbols):
            weights[j] = inv_vols[j] / total

    return weights


@njit(cache=True)  # type: ignore[misc]
def _numba_clamp_normalize(  # noqa: PLR0912 - Numba @njit water-filling clamp; branching cannot be simplified
    weights: npt.NDArray[np.float64],
    min_weight: float,
    max_weight: float,
) -> npt.NDArray[np.float64]:
    """Water-filling clamp + sum=1.0 정규화.

    Args:
        weights: (n_symbols,) raw weights
        min_weight: 에셋당 최소 비중
        max_weight: 에셋당 최대 비중

    Returns:
        (n_symbols,) clamped & normalized weights (합계 1.0)
    """
    n = len(weights)
    result = weights.copy()
    fixed = np.zeros(n, dtype=np.int64)
    tolerance = 1e-10

    for _ in range(5):  # max iterations
        changed = False
        fixed_sum = 0.0
        free_count = 0
        free_total = 0.0

        for j in range(n):
            if fixed[j] == 1:
                fixed_sum += result[j]
            elif result[j] <= min_weight:
                result[j] = min_weight
                fixed[j] = 1
                fixed_sum += min_weight
                changed = True
            elif result[j] >= max_weight:
                result[j] = max_weight
                fixed[j] = 1
                fixed_sum += max_weight
                changed = True
            else:
                free_count += 1
                free_total += result[j]

        if not changed:
            break

        remaining = 1.0 - fixed_sum
        if free_count == 0 or remaining < tolerance:
            break

        if free_total < tolerance:
            per_free = remaining / free_count
            for j in range(n):
                if fixed[j] == 0:
                    result[j] = per_free
        else:
            scale = remaining / free_total
            for j in range(n):
                if fixed[j] == 0:
                    result[j] *= scale

    # 최종 정규화
    final_total = 0.0
    for j in range(n):
        final_total += result[j]
    if abs(final_total - 1.0) > tolerance:
        for j in range(n):
            result[j] /= final_total

    return result


@njit(cache=True)  # type: ignore[misc]
def _numba_inverse_vol_weights(
    returns_matrix: npt.NDArray[np.float64],
    vol_lookback: int,
    min_weight: float,
    max_weight: float,
    rebalance_bars: int,
    n_symbols: int,
) -> npt.NDArray[np.float64]:
    """Bar-by-bar IV weight 계산 (EDA Pod bar_count parity 유지).

    Args:
        returns_matrix: (n_bars, n_symbols) 수익률 행렬 (pct_change().shift(1))
        vol_lookback: 변동성 계산 윈도우
        min_weight: 에셋당 최소 비중
        max_weight: 에셋당 최대 비중
        rebalance_bars: 비중 재계산 주기 (bar_count 기준)
        n_symbols: 심볼 수

    Returns:
        (n_bars, n_symbols) rolling asset weights
    """
    n_bars = returns_matrix.shape[0]
    result = np.empty((n_bars, n_symbols), dtype=np.float64)
    ew = 1.0 / n_symbols

    # 초기 weights
    current_weights = np.empty(n_symbols, dtype=np.float64)
    for j in range(n_symbols):
        current_weights[j] = ew

    bar_count = 0

    for i in range(n_bars):
        # EDA Pod parity: 심볼당 1회 on_bar() → bar_count += n_symbols
        for _s in range(n_symbols):
            bar_count += 1

            if bar_count % rebalance_bars == 0:
                # IV 재계산
                raw_weights = _numba_single_bar_iv(returns_matrix, i, vol_lookback, n_symbols)
                current_weights = _numba_clamp_normalize(raw_weights, min_weight, max_weight)

        # 현재 bar weights 기록
        for j in range(n_symbols):
            result[i, j] = current_weights[j]

    return result


@njit(cache=True)  # type: ignore[misc]
def apply_rebalance_threshold_numba(
    weights: npt.NDArray[np.float64],
    threshold: float,
) -> npt.NDArray[np.float64]:
    """리밸런싱 임계값 적용 (Numba 최적화 버전).

    목표 비중의 변화량이 임계값 미만이면 NaN으로 설정합니다.

    Args:
        weights: 목표 비중 배열
        threshold: 리밸런싱 임계값 (예: 0.05 = 5%)

    Returns:
        임계값이 적용된 비중 배열 (NaN 포함)
    """
    result = np.empty(len(weights))
    result[:] = np.nan
    last_executed_weight = 0.0

    for i in range(len(weights)):
        current_target = weights[i]

        # NaN 체크 (Numba에서는 np.isnan 사용 권장)
        if np.isnan(current_target):
            continue

        change = abs(current_target - last_executed_weight)

        if change >= threshold or (last_executed_weight == 0 and current_target != 0):
            result[i] = current_target
            last_executed_weight = current_target

    return result


def freq_to_hours(freq: str) -> float:
    """freq 문자열을 시간 단위로 변환.

    Args:
        freq: 데이터 주기 문자열 (예: "1d", "4h", "15T")

    Returns:
        시간 수
    """
    freq_upper = freq.strip().upper()
    num_str = ""
    unit = ""
    for ch in freq_upper:
        if ch.isdigit() or ch == ".":
            num_str += ch
        else:
            unit += ch
    num = float(num_str) if num_str else 1.0
    if unit in ("D", "DAY", "DAYS"):
        return num * 24.0
    if unit in ("H", "HOUR", "HOURS"):
        return num
    if unit in ("T", "MIN", "MINUTE", "MINUTES"):
        return num / 60.0
    return 24.0  # 기본값: 일봉


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

    def _execute(self, request: BacktestRequest) -> tuple[BacktestResult, Any]:
        """Core backtest logic.

        Args:
            request: 백테스트 요청

        Returns:
            (BacktestResult, vbt_portfolio) 튜플

        Raises:
            ImportError: VectorBT가 설치되지 않은 경우
        """
        try:
            import vectorbt as vbt  # type: ignore[import-not-found]
        except ImportError as e:
            msg = "VectorBT is required for backtesting. Install with: pip install vectorbt"
            raise ImportError(msg) from e

        data = request.data
        strategy = request.strategy
        portfolio = request.portfolio
        analyzer = request.analyzer or PerformanceAnalyzer()

        logger.debug(
            "BacktestEngine._execute() | {symbol} {tf} ({n} periods) | strategy={name}",
            symbol=data.symbol,
            tf=data.timeframe,
            n=data.periods,
            name=strategy.name,
        )

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

        # 성과 분석 (PerformanceAnalyzer에 위임)
        metrics = analyzer.analyze(vbt_portfolio)
        # H-001: 펀딩비 보정
        metrics = self._adjust_metrics_for_funding(
            vbt_portfolio, metrics, portfolio.config.cost_model, data.freq
        )
        benchmark = analyzer.compare_benchmark(vbt_portfolio, data.ohlcv, data.symbol)
        trades = analyzer.extract_trades(vbt_portfolio, data.symbol)

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

        return result, vbt_portfolio

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
        result, _ = self._execute(request)
        return result

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
        result, vbt_portfolio = self._execute(request)

        analyzer = request.analyzer or PerformanceAnalyzer()
        strategy_returns, benchmark_returns = analyzer.get_returns_series(
            vbt_portfolio,
            request.data.ohlcv,
            request.data.symbol,
            cost_model=request.portfolio.config.cost_model,
            freq=request.data.freq,
        )

        return result, strategy_returns, benchmark_returns

    def run_validated(
        self,
        request: BacktestRequest,
        level: str = "quick",
    ) -> tuple["BacktestResult", "ValidationResult"]:
        """백테스트 + Tiered Validation 실행.

        백테스트를 수행하고 지정된 레벨의 검증을 함께 수행합니다.

        Args:
            request: 백테스트 요청
            level: 검증 레벨 ("quick", "milestone", "final")

        Returns:
            (BacktestResult, ValidationResult) 튜플

        Example:
            >>> result, validation = engine.run_validated(request, level="quick")
            >>> print(f"Sharpe: {result.metrics.sharpe_ratio}")
            >>> print(f"Validation: {validation.verdict}")
        """
        from src.backtest.validation import TieredValidator, ValidationLevel

        # 기본 백테스트 실행 (전체 데이터)
        result = self.run(request)

        # 검증 실행
        validation_level = ValidationLevel(level)
        validator = TieredValidator(engine=self)
        validation = validator.validate(
            level=validation_level,
            data=request.data,
            strategy=request.strategy,
            portfolio=request.portfolio,
        )

        return result, validation

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

        # 2.5. execution_prices 준비 (SL/TS에서 체결가 기준 진입가 설정)
        if pm.price_type == "next_open" and "open" in df.columns:
            exec_prices = np.asarray(
                df["open"].shift(-1).fillna(df["close"]).values, dtype=np.float64
            )
        else:
            exec_prices = np.asarray(df["close"].values, dtype=np.float64)

        # 3. 시스템 손절매 적용 (System Stop Loss)
        stop_loss_count = 0
        if pm.system_stop_loss is not None and pm.system_stop_loss > 0:
            weights_before_sl = target_weights.copy()
            target_weights_sl = apply_stop_loss_to_weights(
                np.asarray(target_weights.fillna(0).values, dtype=np.float64),
                np.asarray(df["close"].values, dtype=np.float64),
                np.asarray(df["high"].values, dtype=np.float64),
                np.asarray(df["low"].values, dtype=np.float64),
                pm.system_stop_loss,
                pm.use_intrabar_stop,
                exec_prices,
            )
            target_weights = pd.Series(target_weights_sl, index=target_weights.index)
            # NaN 복원 (원래 NaN이었던 곳은 유지)
            target_weights = target_weights.where(
                weights_before_sl.notna() | (target_weights != 0), np.nan
            )
            stop_loss_count = int(((weights_before_sl != 0) & (target_weights == 0)).sum())
            if stop_loss_count > 0:
                logger.warning(
                    f"Stop Loss Triggered | {stop_loss_count} positions closed at {pm.system_stop_loss:.0%} loss limit",
                )

        # 3.5. Trailing Stop 적용 (선택적)
        trailing_stop_count = 0
        if pm.use_trailing_stop and "atr" in df.columns:
            weights_before_ts = target_weights.copy()
            target_weights_ts = apply_trailing_stop_to_weights(
                np.asarray(target_weights.fillna(0).values, dtype=np.float64),
                np.asarray(df["close"].values, dtype=np.float64),
                np.asarray(df["high"].values, dtype=np.float64),
                np.asarray(df["low"].values, dtype=np.float64),
                np.asarray(df["atr"].fillna(0).values, dtype=np.float64),
                pm.trailing_stop_atr_multiplier,
                exec_prices,
            )
            target_weights = pd.Series(target_weights_ts, index=target_weights.index)
            # NaN 복원 (원래 NaN이었던 곳은 유지)
            target_weights = target_weights.where(
                weights_before_ts.notna() | (target_weights != 0), np.nan
            )
            trailing_stop_count = int(((weights_before_ts != 0) & (target_weights == 0)).sum())
            if trailing_stop_count > 0:
                logger.info(
                    f"Trailing Stop Triggered | {trailing_stop_count} positions closed at {pm.trailing_stop_atr_multiplier}x ATR",
                )

        # 4. rebalance_threshold 적용 (거래 비용 최적화) - Numba 최적화
        weights_before_threshold = target_weights.copy()
        target_weights_arr = apply_rebalance_threshold_numba(
            np.asarray(target_weights.fillna(0).values, dtype=np.float64),
            pm.rebalance_threshold,
        )
        target_weights = pd.Series(target_weights_arr, index=target_weights.index)

        # 디버그: Rebalance threshold 효과
        num_before = weights_before_threshold.notna().sum()
        num_after = target_weights.notna().sum()
        filtered_pct = (1 - num_after / num_before) * 100 if num_before > 0 else 0
        logger.info(
            f"Rebalance Threshold Effect | Before: {num_before} signals, After: {num_after} orders (Filtered: {num_before - num_after}, {filtered_pct:.1f}%)",
        )

        # 5. price 결정 (next_open 또는 close)
        # CRITICAL: next_open 모드에서는 다음 봉의 시가에 체결해야 함
        # shift(-1): 다음 행의 값을 현재 행으로 가져옴 (Look-ahead Bias 방지)
        # 시그널이 현재 봉 종가 기준이므로, 체결은 다음 봉 시가에서 이루어져야 함
        # NOTE: 마지막 행은 NaN이 되므로 해당 시그널은 체결 불가 (현실적)
        price = df["open"].shift(-1) if pm.price_type == "next_open" else df["close"]

        # 6. Portfolio 생성
        vbt_portfolio = vbt.Portfolio.from_orders(
            close=df["close"],
            size=target_weights,
            size_type="targetpercent",
            direction="both",
            price=price,
            fees=pm.cost_model.effective_fee,
            slippage=pm.cost_model.slippage + pm.cost_model.market_impact,
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

        .. deprecated::
            Numba 최적화 버전 `apply_rebalance_threshold_numba()`를 직접 사용하세요.
            이 메서드는 하위 호환성을 위해 유지됩니다.

        목표 비중의 변화량이 임계값 미만이면 np.nan으로 설정하여
        VectorBT가 해당 캔들에서 주문을 생성하지 않도록 합니다.

        Args:
            target_weights: 목표 비중 시리즈
            threshold: 리밸런싱 임계값 (예: 0.05 = 5%)

        Returns:
            임계값이 적용된 목표 비중 시리즈
        """
        # Numba 최적화 버전으로 위임
        result_arr = apply_rebalance_threshold_numba(
            np.asarray(target_weights.fillna(0).values, dtype=np.float64),
            threshold,
        )
        return pd.Series(result_arr, index=target_weights.index)

    # =========================================================================
    # Multi-Asset Backtest Methods
    # =========================================================================

    def run_multi(self, request: MultiAssetBacktestRequest) -> MultiAssetBacktestResult:
        """멀티에셋 포트폴리오 백테스트 실행.

        Args:
            request: 멀티에셋 백테스트 요청

        Returns:
            MultiAssetBacktestResult
        """
        result, _ = self._execute_multi(request)
        return result

    def run_multi_with_returns(
        self,
        request: MultiAssetBacktestRequest,
    ) -> tuple[MultiAssetBacktestResult, pd.Series, pd.Series]:  # type: ignore[type-arg]
        """멀티에셋 백테스트 + 수익률 시리즈 반환.

        Args:
            request: 멀티에셋 백테스트 요청

        Returns:
            (MultiAssetBacktestResult, portfolio_returns, benchmark_returns) 튜플
        """
        result, vbt_portfolio = self._execute_multi(request)

        # 포트폴리오 수익률 (펀딩비 보정 적용)
        cost_model = request.portfolio.config.cost_model
        if cost_model.funding_rate_8h != 0:
            portfolio_returns = PerformanceAnalyzer.get_funding_adjusted_returns(
                vbt_portfolio, cost_model, request.data.freq
            )
        else:
            portfolio_returns = vbt_portfolio.returns()
        if isinstance(portfolio_returns, pd.DataFrame):
            portfolio_returns = portfolio_returns.iloc[:, 0]

        # 벤치마크: 첫 번째 심볼의 Buy & Hold
        first_symbol = request.data.symbols[0]
        close = request.data.ohlcv[first_symbol]["close"]
        benchmark_returns: pd.Series = close.pct_change().fillna(0)  # type: ignore[assignment]

        return result, portfolio_returns, benchmark_returns

    def run_multi_validated(
        self,
        request: MultiAssetBacktestRequest,
        level: str = "quick",
    ) -> tuple["MultiAssetBacktestResult", "ValidationResult"]:
        """멀티에셋 백테스트 + Tiered Validation 실행.

        Args:
            request: 멀티에셋 백테스트 요청
            level: 검증 레벨 ("quick", "milestone", "final")

        Returns:
            (MultiAssetBacktestResult, ValidationResult) 튜플
        """
        from src.backtest.validation import TieredValidator, ValidationLevel

        result = self.run_multi(request)

        validation_level = ValidationLevel(level)
        validator = TieredValidator(engine=self)
        validation = validator.validate_multi(
            level=validation_level,
            data=request.data,
            strategy=request.strategy,
            portfolio=request.portfolio,
            weights=request.weights,
        )

        return result, validation

    @staticmethod
    def _compute_rolling_asset_weights(
        close_df: pd.DataFrame,
        config: AssetAllocationConfig,
        symbols: tuple[str, ...],
        strengths_dict: dict[str, pd.Series] | None,  # type: ignore[type-arg]
    ) -> dict[str, pd.Series]:  # type: ignore[type-arg]
        """Bar-by-bar rolling asset weight 계산 (EDA Pod parity).

        EDA Pod에서는 심볼당 1회 on_bar() 호출.
        4개 심볼이면 TF bar당 4번 호출 → bar_count 4씩 증가.
        이를 backtest에서 동일하게 복제.

        Args:
            close_df: 심볼별 종가 DataFrame (index=date, columns=symbols)
            config: 에셋 배분 설정
            symbols: 심볼 순서
            strengths_dict: 심볼별 raw strength Series (signal_weighted용)

        Returns:
            심볼별 rolling weight Series (index 동일)
        """
        allocator = IntraPodAllocator(config=config, symbols=symbols)
        n_bars = len(close_df)
        n_symbols = len(symbols)

        # 수익률 계산: pct_change().shift(1) → look-ahead bias 없음
        returns_df = close_df.pct_change().shift(1)

        # 결과 저장
        weight_arrays: dict[str, list[float]] = {s: [] for s in symbols}

        # 누적 수익률 히스토리 (allocator.on_bar()에 전달)
        returns_history: dict[str, list[float]] = {s: [] for s in symbols}

        for i in range(n_bars):
            # 수익률 히스토리 업데이트
            for s in symbols:
                ret_val = returns_df[s].iloc[i]
                if pd.notna(ret_val):
                    returns_history[s].append(float(ret_val))

            # EDA Pod parity: 심볼당 1회 on_bar() 호출
            for _s in symbols:
                strength_val: dict[str, float] | None = None
                if strengths_dict is not None:
                    strength_val = {
                        sym: float(strengths_dict[sym].iloc[i])
                        if pd.notna(strengths_dict[sym].iloc[i])
                        else 0.0
                        for sym in symbols
                    }
                allocator.on_bar(returns=returns_history, strengths=strength_val)

            # 현재 weight 기록
            current_weights = allocator.weights
            for s in symbols:
                weight_arrays[s].append(current_weights.get(s, 1.0 / n_symbols))

        return {s: pd.Series(weight_arrays[s], index=close_df.index) for s in symbols}

    @staticmethod
    def _compute_rolling_asset_weights_fast(
        close_df: pd.DataFrame,
        config: AssetAllocationConfig,
        symbols: tuple[str, ...],
        strengths_dict: dict[str, pd.Series] | None,  # type: ignore[type-arg]
    ) -> dict[str, pd.Series]:  # type: ignore[type-arg]
        """Numba 최적화 rolling asset weight 계산.

        IV/EW → Numba fast path, RP/SW → Python fallback.

        Args:
            close_df: 심볼별 종가 DataFrame (index=date, columns=symbols)
            config: 에셋 배분 설정
            symbols: 심볼 순서
            strengths_dict: 심볼별 raw strength Series (signal_weighted용)

        Returns:
            심볼별 rolling weight Series (index 동일)
        """
        from src.orchestrator.models import AssetAllocationMethod

        method = config.method
        n_symbols = len(symbols)

        # EW: trivial (전체 1/N Series)
        if method == AssetAllocationMethod.EQUAL_WEIGHT:
            ew = 1.0 / n_symbols
            return {s: pd.Series(ew, index=close_df.index) for s in symbols}

        # IV: Numba fast path
        if method == AssetAllocationMethod.INVERSE_VOLATILITY:
            returns_df = close_df.pct_change().shift(1)
            returns_matrix = np.asarray(returns_df[list(symbols)].values, dtype=np.float64)

            weight_matrix = _numba_inverse_vol_weights(
                returns_matrix=returns_matrix,
                vol_lookback=config.vol_lookback,
                min_weight=config.min_weight,
                max_weight=config.max_weight,
                rebalance_bars=config.rebalance_bars,
                n_symbols=n_symbols,
            )

            return {
                s: pd.Series(weight_matrix[:, j], index=close_df.index)
                for j, s in enumerate(symbols)
            }

        # RP/SW: Python fallback
        return BacktestEngine._compute_rolling_asset_weights(
            close_df=close_df,
            config=config,
            symbols=symbols,
            strengths_dict=strengths_dict,
        )

    @staticmethod
    def _scale_with_dynamic_weights(
        raw_strengths: dict[str, pd.Series],  # type: ignore[type-arg]
        rolling_weights: dict[str, pd.Series],  # type: ignore[type-arg]
        processed_dict: dict[str, pd.DataFrame],
        pm: Any,
        n_symbols: int,
    ) -> dict[str, pd.Series]:  # type: ignore[type-arg]
        """Raw strength에 동적 asset weight를 적용하고 PM 규칙 적용.

        scaled = raw_strength * rolling_weight * n_symbols + PM rules

        Args:
            raw_strengths: 심볼별 raw strength Series
            rolling_weights: 심볼별 rolling asset weight Series
            processed_dict: 심볼별 전처리된 DataFrame
            pm: PortfolioManagerConfig
            n_symbols: 심볼 수

        Returns:
            심볼별 최종 target weight Series
        """
        target_weights_dict: dict[str, pd.Series] = {}  # type: ignore[type-arg]
        for symbol, raw_strength in raw_strengths.items():
            scaled = raw_strength * rolling_weights[symbol] * n_symbols
            scaled = _apply_pm_rules_to_weights(scaled, processed_dict[symbol], pm)
            target_weights_dict[symbol] = scaled
        return target_weights_dict

    def _execute_multi(
        self,
        request: MultiAssetBacktestRequest,
    ) -> tuple[MultiAssetBacktestResult, Any]:
        """멀티에셋 백테스트 핵심 로직.

        1. 심볼별 전략 독립 실행
        2. 자산 배분 비중 적용 + PM 규칙 적용
        3. VectorBT cash_sharing 포트폴리오 생성
        4. 포트폴리오/심볼별 분석

        Args:
            request: 멀티에셋 백테스트 요청

        Returns:
            (MultiAssetBacktestResult, vbt_portfolio) 튜플
        """
        try:
            import vectorbt as vbt  # type: ignore[import-not-found]
        except ImportError as e:
            msg = "VectorBT is required for backtesting. Install with: pip install vectorbt"
            raise ImportError(msg) from e

        symbols = request.data.symbols
        asset_weights = request.asset_weights
        portfolio = request.portfolio
        pm = portfolio.config
        analyzer = request.analyzer or PerformanceAnalyzer()

        logger.info(
            f"Multi-asset backtest | {len(symbols)} symbols | strategy={request.strategy.name}"
        )

        # 1. 심볼별 전략 실행 (독립적)
        processed_dict: dict[str, pd.DataFrame] = {}
        raw_strengths: dict[str, pd.Series] = {}  # type: ignore[type-arg]

        for symbol in symbols:
            df = request.data.ohlcv[symbol]
            processed, signals = request.strategy.run(df)
            processed_dict[symbol] = processed
            raw_strengths[symbol] = signals.strength.copy()

        # 2. 자산 배분 비중 적용 + PM 규칙
        if request.asset_allocation is not None:
            close_df_for_alloc = request.data.close_matrix
            rolling_weights = self._compute_rolling_asset_weights_fast(
                close_df=close_df_for_alloc,
                config=request.asset_allocation,
                symbols=tuple(symbols),
                strengths_dict=raw_strengths,
            )
            target_weights_dict = self._scale_with_dynamic_weights(
                raw_strengths=raw_strengths,
                rolling_weights=rolling_weights,
                processed_dict=processed_dict,
                pm=pm,
                n_symbols=len(symbols),
            )
            # 마지막 bar 기준 weight 기록 (결과 config용)
            final_asset_weights = {s: float(rolling_weights[s].iloc[-1]) for s in symbols}
            alloc_method_name = request.asset_allocation.method.value
        else:
            target_weights_dict: dict[str, pd.Series] = {}  # type: ignore[type-arg,no-redef]
            for symbol in symbols:
                scaled = raw_strengths[symbol] * asset_weights[symbol]
                scaled = _apply_pm_rules_to_weights(scaled, processed_dict[symbol], pm)
                target_weights_dict[symbol] = scaled
            final_asset_weights = asset_weights
            alloc_method_name = None

        # 3. DataFrame으로 합성 (VectorBT 멀티에셋 입력)
        close_df = request.data.close_matrix
        weights_df = pd.DataFrame(target_weights_dict)

        # 인덱스 정렬 (심볼별 인덱스가 다를 수 있음)
        common_index = close_df.index.intersection(weights_df.index)
        close_df = close_df.loc[common_index]
        weights_df = weights_df.loc[common_index]

        # M-001: Aggregate leverage 검증 및 스케일링
        agg_leverage = weights_df.abs().sum(axis=1)
        max_agg_leverage = float(agg_leverage.max())
        if max_agg_leverage > pm.max_leverage_cap:
            scale_factor = pm.max_leverage_cap / agg_leverage
            scale_factor = scale_factor.clip(upper=1.0)
            weights_df = weights_df.multiply(scale_factor, axis=0)
            logger.warning(
                f"Aggregate Leverage Capped | Peak: {max_agg_leverage:.2f}x → {pm.max_leverage_cap:.1f}x"
            )

        # 4. price 결정
        if pm.price_type == "next_open":
            price_dict: dict[str, pd.Series] = {}  # type: ignore[type-arg]
            for symbol in symbols:
                # common_index로 필터 후 shift → 인덱스 불일치 방지
                open_series = processed_dict[symbol]["open"].reindex(common_index)
                price_dict[symbol] = open_series.shift(-1)  # type: ignore[assignment]
            price_df: pd.DataFrame = pd.DataFrame(price_dict)
        else:
            price_df = close_df

        # 5. VectorBT Portfolio 생성 (cash_sharing)
        vbt_portfolio = vbt.Portfolio.from_orders(
            close=close_df,
            size=weights_df,
            size_type="targetpercent",
            direction="both",
            price=price_df,
            fees=pm.cost_model.effective_fee,
            slippage=pm.cost_model.slippage + pm.cost_model.market_impact,
            init_cash=portfolio.initial_capital_float,
            cash_sharing=True,
            group_by=True,
            freq=request.data.freq,
        )

        # 6. 포트폴리오 전체 성과 분석
        portfolio_metrics = analyzer.analyze(vbt_portfolio)
        # H-001: 펀딩비 보정
        portfolio_metrics = self._adjust_metrics_for_funding(
            vbt_portfolio, portfolio_metrics, pm.cost_model, request.data.freq
        )

        # 7. 심볼별 성과 분석 (개별 수익률 기반)
        per_symbol_metrics: dict[str, PerformanceMetrics] = {}
        returns_dict: dict[str, pd.Series] = {}  # type: ignore[type-arg]
        for symbol in symbols:
            close_series = close_df[symbol]
            symbol_returns: pd.Series = close_series.pct_change().fillna(0)  # type: ignore[type-arg,assignment]
            # 심볼별 가중 수익률 (target_weight x market_return은 근사치)
            returns_dict[symbol] = symbol_returns
            # 간단한 B&H 성과 지표 계산
            per_symbol_metrics[symbol] = self._compute_simple_metrics(symbol_returns)

        # 8. 상관행렬
        returns_df = pd.DataFrame(returns_dict)
        corr_matrix = returns_df.corr()
        correlation_dict: dict[str, dict[str, float]] = {
            s1: {s2: float(corr_matrix.loc[s1, s2]) for s2 in symbols} for s1 in symbols
        }

        # 9. 가중 수익 기여도 (asset_weight x cumulative_return)
        contribution: dict[str, float] = {}
        for symbol in symbols:
            weight = final_asset_weights[symbol]
            symbol_cum_ret = float(returns_df[symbol].sum())
            contribution[symbol] = round(weight * symbol_cum_ret * 100, 4)

        # 10. 결과 조합
        config = MultiAssetConfig(
            strategy_name=request.strategy.name,
            symbols=tuple(symbols),
            timeframe=request.data.timeframe,
            start_date=request.data.start,
            end_date=request.data.end,
            initial_capital=portfolio.initial_capital,
            asset_weights=final_asset_weights,
            asset_allocation_method=alloc_method_name,
            maker_fee=pm.cost_model.maker_fee,
            taker_fee=pm.cost_model.taker_fee,
            slippage=pm.cost_model.slippage,
            strategy_params=request.strategy.params,
        )

        result = MultiAssetBacktestResult(
            config=config,
            portfolio_metrics=portfolio_metrics,
            per_symbol_metrics=per_symbol_metrics,
            correlation_matrix=correlation_dict,
            contribution=contribution,
        )

        logger.info(
            f"Multi-asset backtest complete | Sharpe={portfolio_metrics.sharpe_ratio:.2f}, CAGR={portfolio_metrics.cagr:.2f}%, MDD={portfolio_metrics.max_drawdown:.2f}%"
        )

        return result, vbt_portfolio

    @staticmethod
    def _adjust_metrics_for_funding(
        vbt_portfolio: Any,
        metrics: PerformanceMetrics,
        cost_model: Any,
        freq: str,
    ) -> PerformanceMetrics:
        """펀딩비 반영 및 365일 연환산으로 성과 지표 보정.

        VBT는 거래 수수료와 슬리피지만 반영하며 252일 기반 연환산을 사용하므로,
        (1) 선물 펀딩비 드래그 차감, (2) 365일 기반 연환산으로 재계산합니다.

        Args:
            vbt_portfolio: VBT Portfolio
            metrics: 원본 성과 지표
            cost_model: CostModel (funding_rate_8h 포함)
            freq: 데이터 주기 (예: "1d", "1h")

        Returns:
            보정된 PerformanceMetrics
        """
        returns = vbt_portfolio.returns()
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]

        hours_per_period = freq_to_hours(freq)
        periods_per_year = (365 * 24) / hours_per_period

        # 펀딩비 보정 (funding_rate_8h != 0인 경우만)
        has_funding = cost_model.funding_rate_8h != 0
        if has_funding:
            try:
                asset_value = vbt_portfolio.asset_value()
                total_value = vbt_portfolio.value()
                if isinstance(asset_value, pd.DataFrame):
                    asset_value = asset_value.sum(axis=1)
                if isinstance(total_value, pd.DataFrame):
                    total_value = total_value.iloc[:, 0]
                eff_weight = (asset_value / total_value).abs().fillna(0)
            except Exception:
                eff_weight = pd.Series(0.5, index=returns.index)

            funding_per_period = cost_model.funding_rate_8h * (hours_per_period / 8.0)
            funding_drag = eff_weight * funding_per_period
            adjusted_returns = returns - funding_drag
        else:
            adjusted_returns = returns
            funding_drag = pd.Series(0.0, index=returns.index)

        # 보정 지표 재계산 (365일 연환산)
        cum = (1 + adjusted_returns).cumprod()
        n = len(cum)
        if n == 0:
            return metrics

        adj_total_return = float((cum.iloc[-1] - 1) * 100)
        years = n / periods_per_year

        growth = float(cum.iloc[-1])
        adj_cagr = float((growth ** (1.0 / years) - 1.0) * 100) if years > 0 and growth > 0 else 0.0

        # Sharpe 재계산 (365일 기반)
        mean_ret = float(adjusted_returns.mean())
        std_ret = float(adjusted_returns.std())
        adj_sharpe = float((mean_ret / std_ret) * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

        # Sortino 재계산
        downside = adjusted_returns[adjusted_returns < 0]
        downside_std = float(downside.std()) if len(downside) > 0 else 0.0
        adj_sortino: float | None = (
            float((mean_ret / downside_std) * np.sqrt(periods_per_year))
            if downside_std > 0
            else None
        )

        # MDD 재계산 (양수 반환)
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        adj_mdd = float(abs(dd.min()) * 100)

        # Calmar 재계산 (MDD 양수 기준)
        adj_calmar: float | None = abs(adj_cagr / adj_mdd) if adj_mdd > 0 else None

        if has_funding:
            total_drag_pct = float(funding_drag.sum() * 100)
            logger.info(
                f"Funding Adjustment | Drag: {total_drag_pct:.2f}%, Sharpe: {metrics.sharpe_ratio:.2f} → {adj_sharpe:.2f}, CAGR: {metrics.cagr:.2f}% → {adj_cagr:.2f}%"
            )
        else:
            logger.debug(
                f"365-day Annualization | Sharpe: {metrics.sharpe_ratio:.2f} → {adj_sharpe:.2f}"
            )

        return PerformanceMetrics(
            total_return=adj_total_return,
            cagr=adj_cagr,
            sharpe_ratio=adj_sharpe,
            sortino_ratio=adj_sortino,
            calmar_ratio=adj_calmar,
            max_drawdown=adj_mdd,
            avg_drawdown=metrics.avg_drawdown,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            avg_win=metrics.avg_win,
            avg_loss=metrics.avg_loss,
            total_trades=metrics.total_trades,
            winning_trades=metrics.winning_trades,
            losing_trades=metrics.losing_trades,
        )

    @staticmethod
    def _compute_simple_metrics(returns: pd.Series) -> PerformanceMetrics:  # type: ignore[type-arg]
        """일별 수익률에서 간단한 성과 지표 계산 (심볼별 분석용).

        Args:
            returns: 일별 수익률 시리즈

        Returns:
            PerformanceMetrics
        """
        cum = (1 + returns).cumprod()
        total_return = float((cum.iloc[-1] - 1) * 100) if len(cum) > 0 else 0.0
        days = len(returns)
        years = days / 365.0 if days > 0 else 1.0
        cagr = float((cum.iloc[-1] ** (1 / years) - 1) * 100) if years > 0 and len(cum) > 0 else 0.0

        # MDD (양수 반환)
        running_max = cum.cummax()
        drawdowns = (cum - running_max) / running_max
        max_drawdown = float(abs(drawdowns.min()) * 100)

        # Sharpe
        ann_vol = float(returns.std() * np.sqrt(365) * 100) if len(returns) > 1 else 0.0
        sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

        return PerformanceMetrics(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
        )


def _apply_pm_rules_to_weights(
    weights: pd.Series,  # type: ignore[type-arg]
    df: pd.DataFrame,
    pm: Any,
) -> pd.Series:  # type: ignore[type-arg]
    """단일 심볼의 비중에 PM 규칙 적용.

    BacktestEngine._create_portfolio_from_orders()의 Step 2~4 로직을
    재사용 가능한 함수로 추출한 것입니다.

    Args:
        weights: 목표 비중 시리즈 (strength x asset_weight)
        df: 전처리된 OHLCV DataFrame (close, high, low, atr 포함)
        pm: PortfolioManagerConfig

    Returns:
        PM 규칙이 적용된 비중 시리즈
    """
    # 1. max_leverage_cap 적용
    weights = weights.clip(lower=-pm.max_leverage_cap, upper=pm.max_leverage_cap)

    # 1.5. execution_prices 준비 (SL/TS에서 체결가 기준 진입가 설정)
    if pm.price_type == "next_open" and "open" in df.columns:
        exec_prices = np.asarray(df["open"].shift(-1).fillna(df["close"]).values, dtype=np.float64)
    else:
        exec_prices = np.asarray(df["close"].values, dtype=np.float64)

    # 2. 시스템 손절매 (System Stop Loss)
    if pm.system_stop_loss is not None and pm.system_stop_loss > 0:
        weights_before = weights.copy()
        result_arr = apply_stop_loss_to_weights(
            np.asarray(weights.fillna(0).values, dtype=np.float64),
            np.asarray(df["close"].values, dtype=np.float64),
            np.asarray(df["high"].values, dtype=np.float64),
            np.asarray(df["low"].values, dtype=np.float64),
            pm.system_stop_loss,
            pm.use_intrabar_stop,
            exec_prices,
        )
        weights = pd.Series(result_arr, index=weights.index)
        weights = weights.where(weights_before.notna() | (weights != 0), np.nan)

    # 3. Trailing Stop
    if pm.use_trailing_stop and "atr" in df.columns:
        weights_before = weights.copy()
        result_arr = apply_trailing_stop_to_weights(
            np.asarray(weights.fillna(0).values, dtype=np.float64),
            np.asarray(df["close"].values, dtype=np.float64),
            np.asarray(df["high"].values, dtype=np.float64),
            np.asarray(df["low"].values, dtype=np.float64),
            np.asarray(df["atr"].fillna(0).values, dtype=np.float64),
            pm.trailing_stop_atr_multiplier,
            exec_prices,
        )
        weights = pd.Series(result_arr, index=weights.index)
        weights = weights.where(weights_before.notna() | (weights != 0), np.nan)

    # 4. Rebalance threshold
    result_arr = apply_rebalance_threshold_numba(
        np.asarray(weights.fillna(0).values, dtype=np.float64),
        pm.rebalance_threshold,
    )
    return pd.Series(result_arr, index=weights.index)


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
    from itertools import product

    from src.backtest.request import BacktestRequest

    engine = BacktestEngine()
    results = []
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for combination in product(*param_values):
        params = dict(zip(param_names, combination, strict=True))

        try:
            strategy = strategy_class.from_params(**params)

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
