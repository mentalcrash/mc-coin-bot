"""VW-TSMOM Preprocessor (Indicator Calculation).

이 모듈은 VW-TSMOM 전략에 필요한 모든 지표를 벡터화된 연산으로 계산합니다.
백테스팅과 라이브 트레이딩 모두에서 동일한 코드를 사용합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.strategy.tsmom.config import TSMOMConfig

logger = logging.getLogger(__name__)


def calculate_returns(
    close: pd.Series | Any,
    use_log: bool = True,
) -> pd.Series:
    """수익률 계산 (로그 또는 단순).

    Args:
        close: 종가 시리즈
        use_log: True면 로그 수익률, False면 단순 수익률

    Returns:
        수익률 시리즈 (첫 값은 NaN)

    Example:
        >>> returns = calculate_returns(df["close"], use_log=True)
    """
    # Series 타입 검증
    if not isinstance(close, pd.Series):
        msg = f"Expected pd.Series, got {type(close)}"
        raise TypeError(msg)

    if len(close) == 0:
        msg = "Empty Series provided"
        raise ValueError(msg)

    if use_log:
        # 로그 수익률: ln(P_t / P_{t-1})
        price_ratio = close / close.shift(1)
        return pd.Series(np.log(price_ratio), index=close.index, name="returns")
    # 단순 수익률: (P_t - P_{t-1}) / P_{t-1}
    return close.pct_change()


def calculate_realized_volatility(
    returns: pd.Series,
    window: int,
    annualization_factor: float = 8760.0,
    min_periods: int | None = None,
) -> pd.Series:
    """실현 변동성 계산 (연환산).

    Rolling standard deviation을 사용하여 실현 변동성을 계산합니다.
    결과는 연환산되어 반환됩니다.

    Args:
        returns: 수익률 시리즈
        window: Rolling 윈도우 크기
        annualization_factor: 연환산 계수 (시간봉: 8760)
        min_periods: 최소 관측치 수 (None이면 window 사용)

    Returns:
        연환산 변동성 시리즈

    Example:
        >>> vol = calculate_realized_volatility(returns, window=24)
    """
    if min_periods is None:
        min_periods = window

    # Rolling 표준편차 계산
    rolling_std = returns.rolling(window=window, min_periods=min_periods).std()

    # 연환산: vol_annual = vol_period * sqrt(periods_per_year)
    return rolling_std * np.sqrt(annualization_factor)


def calculate_volume_weighted_returns(
    returns: pd.Series,
    volume: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """거래량 가중 수익률 계산 (로그 스케일링 적용).

    각 기간의 수익률에 로그 거래량을 가중하여 평균합니다.
    로그 스케일링으로 거래량 이상치(패닉 셀링 등)의 과도한 영향력을 압축합니다.

    Log-Volume Scaling:
        - 거래량 100배 → 가중치 ln(100) ≈ 4.6배 (100배가 아님)
        - 패닉 셀링 한 방에 전체 추세가 뒤집히는 것을 방지

    Args:
        returns: 수익률 시리즈
        volume: 거래량 시리즈
        window: Rolling 윈도우 크기
        min_periods: 최소 관측치 수

    Returns:
        거래량 가중 수익률 시리즈 (로그 스케일링 적용)

    Example:
        >>> vw_returns = calculate_volume_weighted_returns(
        ...     df["returns"], df["volume"], window=24
        ... )
    """
    if min_periods is None:
        min_periods = window

    # 로그 스케일링: ln(volume + 1)로 이상치 영향력 압축
    # +1은 volume=0일 때 ln(0) = -inf 방지
    log_volume = np.log1p(volume)  # log1p(x) = ln(1 + x), 수치 안정성 우수

    # 가중 수익률: sum(return * ln_volume) / sum(ln_volume)
    weighted_returns: pd.Series = (  # type: ignore[assignment]
        (returns * log_volume).rolling(window=window, min_periods=min_periods).sum()
    )
    total_log_volume: pd.Series = log_volume.rolling(  # type: ignore[assignment]
        window=window, min_periods=min_periods
    ).sum()

    # 0으로 나누기 방지
    total_log_volume_safe = total_log_volume.replace(0, np.nan)
    return weighted_returns / total_log_volume_safe


def calculate_vw_momentum(
    returns: pd.Series,
    volume: pd.Series,
    lookback: int,
    smoothing: int | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """거래량 가중 모멘텀 계산.

    VW-TSMOM의 핵심 지표입니다. 거래량 가중 수익률의 누적 합계로
    모멘텀을 측정합니다.

    Args:
        returns: 수익률 시리즈
        volume: 거래량 시리즈
        lookback: 모멘텀 계산 기간
        smoothing: EMA 스무딩 윈도우 (선택적)
        min_periods: 최소 관측치 수

    Returns:
        모멘텀 시리즈

    Example:
        >>> momentum = calculate_vw_momentum(
        ...     df["returns"], df["volume"], lookback=24
        ... )
    """
    # 거래량 가중 수익률 계산
    vw_returns: pd.Series = calculate_volume_weighted_returns(
        returns, volume, lookback, min_periods
    )

    # 선택적 스무딩 (EMA)
    if smoothing is not None and smoothing > 1:
        vw_returns = vw_returns.ewm(span=smoothing, adjust=False).mean()  # type: ignore[assignment]

    return vw_returns


def calculate_zscore_momentum(
    returns: pd.Series,
    volume: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Z-Score 정규화된 거래량 가중 모멘텀 계산.

    모멘텀을 변동성으로 나누어 표준화합니다 (Risk-Adjusted Return).
    결과는 보통 -2 ~ +2 (Sigma) 범위의 값으로, 신호 강도를 명확히 표현합니다.

    Formula:
        cumulative_vw_return = sum(vw_returns over window)
        vol = std(returns) * sqrt(window)  # 기간 스케일링된 변동성
        z_score = cumulative_vw_return / vol

    Args:
        returns: 수익률 시리즈
        volume: 거래량 시리즈
        window: 룩백 윈도우
        min_periods: 최소 관측치 수

    Returns:
        Z-Score 정규화된 모멘텀 시리즈 (보통 -2 ~ +2 범위)

    Example:
        >>> zscore = calculate_zscore_momentum(returns, volume, window=60)
    """
    if min_periods is None:
        min_periods = window // 2  # 앙상블에서 더 빠르게 신호 생성

    # 1. 로그 볼륨 가중치 계산
    log_volume = np.log1p(volume)

    # 2. 가중 수익률의 **정규화된 누적 합계** (윈도우 기간 동안)
    # 🔧 FIX (H2): sum(returns * log_volume) / sum(log_volume) 으로 정규화
    weighted_returns = returns * log_volume
    sum_weighted_returns: pd.Series = weighted_returns.rolling(  # type: ignore[assignment]
        window=window, min_periods=min_periods
    ).sum()
    sum_log_volume: pd.Series = log_volume.rolling(  # type: ignore[assignment]
        window=window, min_periods=min_periods
    ).sum()
    # 정규화된 누적 수익률 (가중 평균)
    sum_log_volume_safe = sum_log_volume.replace(0, np.nan)
    cumulative_vw_ret: pd.Series = sum_weighted_returns / sum_log_volume_safe  # type: ignore[assignment]

    # 3. 변동성 계산
    # 🔧 FIX (H6): cumulative_vw_ret이 평균이므로 sqrt(window) 스케일링 불필요
    # sqrt(window)는 누적 합계를 사용할 때만 필요 (분산의 가산성)
    vol: pd.Series = returns.rolling(  # type: ignore[assignment]
        window=window, min_periods=min_periods
    ).std()

    # 4. Z-Score 계산: 정규화된 가중평균수익률 / 변동성
    # 0으로 나누기 방지
    vol_safe = vol.replace(0, np.nan)
    z_score: pd.Series = cumulative_vw_ret / vol_safe  # type: ignore[assignment]

    return z_score


def calculate_ensemble_momentum(
    returns: pd.Series,
    volume: pd.Series,
    windows: tuple[int, ...],
    clip_value: float = 2.0,
) -> pd.Series:
    """앙상블 모멘텀 계산 (여러 윈도우의 Z-Score 평균).

    여러 타임프레임의 모멘텀을 Z-Score로 정규화한 후 평균을 냅니다.
    효과: 단기 변동(휩쏘)에 덜 민감하고, 여러 시간대의 추세 합의를 반영.

    Example:
        windows = (60, 120, 240)  # 10일, 20일, 40일 (4시간봉 기준)
        - 10일 선이 꺾여도 40일 선이 살아있으면 롱 유지
        - 모든 윈도우가 같은 방향일 때만 강한 신호

    Args:
        returns: 수익률 시리즈
        volume: 거래량 시리즈
        windows: 앙상블 윈도우 튜플 (예: (60, 120, 240))
        clip_value: Z-Score 클리핑 범위 (기본 ±2.0 sigma)

    Returns:
        앙상블 모멘텀 시리즈 (클리핑된 Z-Score 평균)

    Example:
        >>> ensemble = calculate_ensemble_momentum(
        ...     returns, volume, windows=(60, 120, 240), clip_value=2.0
        ... )
    """
    if not windows:
        msg = "ensemble_windows must not be empty"
        raise ValueError(msg)

    # 각 윈도우별 Z-Score 계산
    z_scores: list[pd.Series] = []
    for w in windows:
        z = calculate_zscore_momentum(returns, volume, w)
        z_scores.append(z)

    # DataFrame으로 결합 후 행 평균 계산
    z_df = pd.concat(z_scores, axis=1)
    ensemble_mean: pd.Series = z_df.mean(axis=1)  # type: ignore[assignment]

    # 클리핑: 이상치 제거 (-clip ~ +clip)
    clipped: pd.Series = ensemble_mean.clip(lower=-clip_value, upper=clip_value)

    return clipped


def calculate_volatility_scalar(
    realized_vol: pd.Series,
    vol_target: float,
    min_volatility: float = 0.05,
) -> pd.Series:
    """변동성 스케일러 계산.

    목표 변동성 대비 실현 변동성의 비율을 계산합니다.
    변동성이 높을 때 포지션을 줄이고, 낮을 때 늘립니다.

    Args:
        realized_vol: 실현 변동성 시리즈
        vol_target: 연간 목표 변동성 (예: 0.15)
        min_volatility: 최소 변동성 클램프 (0으로 나누기 방지)

    Returns:
        변동성 스케일러 시리즈

    Example:
        >>> scalar = calculate_volatility_scalar(vol, vol_target=0.15)
    """
    # 최소 변동성으로 클램프 (0으로 나누기 방지)
    clamped_vol = realized_vol.clip(lower=min_volatility)

    # 스케일러 계산: target / realized
    return vol_target / clamped_vol


def preprocess(  # noqa: PLR0915
    df: pd.DataFrame,
    config: TSMOMConfig,
) -> pd.DataFrame:
    """VW-TSMOM 전처리 (모든 지표 계산).

    OHLCV DataFrame에 VW-TSMOM 전략에 필요한 모든 지표를 계산하여 추가합니다.
    모든 계산은 벡터화되어 있으며 for 루프를 사용하지 않습니다.

    Note:
        레버리지 클램핑과 시그널 필터링은 PortfolioManagerConfig에서 처리됩니다.
        전략은 순수한 raw_signal만 생성하고, PM이 max_leverage_cap과
        rebalance_threshold를 적용합니다.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - realized_vol: 실현 변동성 (연환산)
        - vw_momentum: 거래량 가중 모멘텀
        - vol_scalar: 변동성 스케일러
        - raw_signal: 원시 시그널 (방향 x 스케일러, 레버리지 무제한)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: close, volume
        config: TSMOM 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시

    Example:
        >>> config = TSMOMConfig(lookback=24, vol_target=0.15)
        >>> processed_df = preprocess(ohlcv_df, config)
        >>> processed_df["vw_momentum"]  # 모멘텀 시리즈
    """
    # 입력 검증
    required_cols = {"close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # 원본 보존 (복사본 생성)
    result = df.copy()

    # OHLCV 컬럼을 float64로 변환 (Decimal 타입 처리)
    # Parquet에서 Decimal로 저장된 경우 np.log() 등이 작동하지 않음
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출 (명시적 Series 타입)
    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = calculate_returns(
        close_series,
        use_log=config.use_log_returns,
    )

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 실현 변동성 계산 (연환산)
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 3. 거래량 가중 모멘텀 계산 (앙상블 또는 단일 윈도우)
    if config.use_zscore and config.ensemble_windows:
        # 🆕 앙상블 모드: 여러 윈도우의 Z-Score 정규화 평균
        result["vw_momentum"] = calculate_ensemble_momentum(
            returns_series,
            volume_series,
            windows=config.ensemble_windows,
            clip_value=config.zscore_clip,
        )
        logger.info(
            "🔄 Ensemble Mode | Windows: %s, Z-Score Clip: ±%.1f",
            config.ensemble_windows,
            config.zscore_clip,
        )
    else:
        # 기존 단일 윈도우 모드
        result["vw_momentum"] = calculate_vw_momentum(
            returns_series,
            volume_series,
            lookback=config.lookback,
            smoothing=config.momentum_smoothing,
        )

    # 4. 변동성 스케일러 계산
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 5. 원시 시그널 계산
    if config.use_zscore:
        # 🆕 Z-Score 모드: 모멘텀 자체가 이미 정규화됨
        # 모멘텀 강도를 직접 사용 (방향 포함)
        # vol_scalar로 목표 변동성에 맞춰 스케일링
        result["raw_signal"] = result["vw_momentum"] * result["vol_scalar"]
        logger.info("📈 Z-Score Signal | Momentum (normalized) used directly")
    else:
        # 기존 모드: 방향만 추출하고 vol_scalar로 크기 조절
        momentum_direction = np.sign(result["vw_momentum"])
        result["raw_signal"] = momentum_direction * result["vol_scalar"]

    # 6. 🆕 Trend Filter (국면 필터) - 메타데이터만 저장
    # 실제 필터링은 signal.py에서 shift(1) 후 적용
    if config.use_trend_filter:
        trend_ma: pd.Series = close_series.rolling(  # type: ignore[assignment]
            window=config.trend_ma_period, min_periods=config.trend_ma_period // 2
        ).mean()
        result["trend_ma"] = trend_ma

        # 추세 판단: 1 = 상승장, -1 = 하락장
        # signal.py에서 필터링할 때 사용할 메타데이터
        result["trend_regime"] = np.where(close_series > trend_ma, 1, -1)

        # 통계 로깅
        uptrend_count = int((result["trend_regime"] == 1).sum())
        downtrend_count = int((result["trend_regime"] == -1).sum())
        logger.info(
            "🎯 Trend Filter | MA(%d): Uptrend %d days, Downtrend %d days",
            config.trend_ma_period,
            uptrend_count,
            downtrend_count,
        )

    # 7. 🆕 Deadband (불감대)
    # 신호 강도가 임계값 이하면 중립 유지 (확실한 추세에서만 진입)
    if config.deadband_threshold > 0:
        momentum: pd.Series = result["vw_momentum"]  # type: ignore[assignment]

        # |momentum| < threshold 면 신호를 0으로 (Z-Score 기준)
        deadband_mask = np.abs(momentum) < config.deadband_threshold
        result["raw_signal"] = np.where(deadband_mask, 0, result["raw_signal"])

        # 통계 로깅
        filtered_count = int(deadband_mask.sum())
        total_count = len(momentum.dropna())
        if total_count > 0:
            filtered_pct = filtered_count / total_count * 100
            logger.info(
                "🚫 Deadband | Threshold: %.2f, Filtered: %d/%d (%.1f%%)",
                config.deadband_threshold,
                filtered_count,
                total_count,
                filtered_pct,
            )

    # 🔍 디버그: 지표 통계 (NaN 제외)
    valid_data = result.dropna()
    if len(valid_data) > 0:
        mom_min = valid_data["vw_momentum"].min()
        mom_max = valid_data["vw_momentum"].max()
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        sig_min = valid_data["raw_signal"].min()
        sig_max = valid_data["raw_signal"].max()
        logger.info(
            "📊 VW-TSMOM | Mom: [%.4f, %.4f] Vol: [%.2f, %.2f] Sig: [%.2f, %.2f]",
            mom_min,
            mom_max,
            vs_min,
            vs_max,
            sig_min,
            sig_max,
        )
        # 방향성 검증: 가격 vs 모멘텀
        price_change = (result["close"].iloc[-1] / result["close"].iloc[0] - 1) * 100
        avg_momentum = valid_data["vw_momentum"].mean()
        aligned = (price_change > 0 and avg_momentum > 0) or (
            price_change < 0 and avg_momentum < 0
        )
        status = "✅ Aligned" if aligned else "⚠️ Diverged"
        logger.info(
            "🎯 Direction Check | Price Change: %+.2f%%, Avg Momentum: %+.4f (%s)",
            price_change,
            avg_momentum,
            status,
        )

    return result


def preprocess_live(
    buffer: pd.DataFrame,
    config: TSMOMConfig,
    max_rows: int = 200,
) -> pd.DataFrame:
    """라이브 트레이딩용 전처리 (버퍼 기반).

    라이브 트레이딩에서는 전체 데이터가 아닌 최근 버퍼만 유지하며
    계산합니다. 메모리 효율적이며 실시간 처리에 적합합니다.

    Args:
        buffer: 최근 캔들 버퍼 (최신이 마지막)
        config: TSMOM 설정
        max_rows: 최대 버퍼 크기

    Returns:
        전처리된 버퍼 (마지막 행이 최신 시그널)

    Example:
        >>> # 라이브 트레이딩 루프에서
        >>> buffer = buffer.append(new_candle).tail(200)
        >>> processed = preprocess_live(buffer, config)
        >>> latest_signal = processed["raw_signal"].iloc[-1]
    """
    # 버퍼 크기 제한
    if len(buffer) > max_rows:
        buffer = buffer.tail(max_rows)

    # 일반 전처리 수행
    return preprocess(buffer, config)
