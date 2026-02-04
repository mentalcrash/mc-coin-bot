"""Performance Metrics Calculation.

이 모듈은 백테스트 결과에서 성과 지표를 계산하는 함수들을 제공합니다.
VectorBT 출력 및 일반 수익률 시리즈와 호환됩니다.

Rules Applied:
    - #25 QuantStats Standards: 암호화폐 기준 (365일)
    - #12 Data Engineering: Log returns for calculation
"""

# pyright: reportArgumentType=false, reportOperatorIssue=false
# pandas Scalar 타입은 실제로 숫자형이지만 타입 체커가 모든 가능성(complex, datetime 등)을 고려함

import numpy as np
import pandas as pd


def calculate_returns(
    prices: pd.Series,
    log_returns: bool = False,
) -> pd.Series:
    """수익률 계산.

    Args:
        prices: 가격 시리즈
        log_returns: True면 로그 수익률

    Returns:
        수익률 시리즈
    """
    if log_returns:
        # np.log 결과를 명시적으로 Series로 변환
        result = np.log(prices / prices.shift(1))
        return pd.Series(result, index=prices.index)
    return prices.pct_change()


def calculate_total_return(returns: pd.Series) -> float:
    """총 수익률 계산.

    Args:
        returns: 수익률 시리즈

    Returns:
        총 수익률 (%)
    """
    cumulative_value = (1 + returns.fillna(0)).prod()
    # pandas prod()는 Scalar 타입을 반환하므로 명시적으로 float 변환
    cumulative_float = float(cumulative_value)
    return (cumulative_float - 1) * 100


def calculate_cagr(
    returns: pd.Series,
    periods_per_year: int = 8760,  # 시간봉 기준
) -> float:
    """CAGR (연평균 복리 수익률) 계산.

    Args:
        returns: 수익률 시리즈
        periods_per_year: 연간 기간 수 (시간봉: 8760)

    Returns:
        CAGR (%)
    """
    if len(returns) == 0:
        return 0.0

    total_return_value = (1 + returns.fillna(0)).prod()
    total_return_float = float(total_return_value)
    years = len(returns) / periods_per_year

    if years <= 0 or total_return_float <= 0:
        return 0.0

    cagr = (total_return_float ** (1 / years) - 1) * 100
    return cagr


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,  # 연 5%
    periods_per_year: int = 8760,
) -> float:
    """샤프 비율 계산.

    Args:
        returns: 수익률 시리즈
        risk_free_rate: 무위험 수익률 (연율)
        periods_per_year: 연간 기간 수

    Returns:
        샤프 비율
    """
    returns_std = float(returns.std())
    if len(returns) == 0 or returns_std == 0:
        return 0.0

    # 기간별 무위험 수익률
    rf_per_period = risk_free_rate / periods_per_year

    # 초과 수익률
    excess_returns = returns - rf_per_period

    # 샤프 비율 (연환산)
    mean_excess = float(excess_returns.mean())
    std_excess = float(excess_returns.std())
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 8760,
) -> float:
    """소르티노 비율 계산.

    하방 변동성만 사용하여 위험 조정 수익을 측정합니다.

    Args:
        returns: 수익률 시리즈
        risk_free_rate: 무위험 수익률 (연율)
        periods_per_year: 연간 기간 수

    Returns:
        소르티노 비율
    """
    if len(returns) == 0:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period

    # 하방 수익률 (음수만)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float("inf")  # 손실 없음

    downside_std_value = float(downside_returns.std())
    if downside_std_value == 0:
        return 0.0

    mean_excess = float(excess_returns.mean())
    sortino = (mean_excess / downside_std_value) * np.sqrt(periods_per_year)
    return sortino


def calculate_max_drawdown(
    returns: pd.Series | None = None,
    equity_curve: pd.Series | None = None,
) -> float:
    """최대 낙폭 (MDD) 계산.

    Args:
        returns: 수익률 시리즈 (equity_curve 없으면 필수)
        equity_curve: 자산 곡선 (제공되면 직접 사용)

    Returns:
        최대 낙폭 (%, 음수)
    """
    curve: pd.Series
    if equity_curve is not None:
        curve = equity_curve
    elif returns is not None:
        curve = (1 + returns.fillna(0)).cumprod()
    else:
        return 0.0

    if len(curve) == 0:
        return 0.0

    # Running maximum
    running_max = curve.cummax()

    # Drawdown series
    drawdown = (curve - running_max) / running_max

    # Maximum drawdown
    mdd_value = float(drawdown.min())
    return mdd_value * 100


def calculate_calmar_ratio(
    cagr: float,
    max_drawdown: float,
) -> float | None:
    """칼마 비율 계산.

    CAGR / |MDD| 비율입니다.

    Args:
        cagr: 연평균 복리 수익률 (%)
        max_drawdown: 최대 낙폭 (%, 음수)

    Returns:
        칼마 비율 또는 None (계산 불가 시)
    """
    if max_drawdown >= 0:
        return None

    calmar = cagr / abs(max_drawdown)
    return float(calmar)


def calculate_win_rate(
    trade_returns: pd.Series,
) -> float:
    """승률 계산.

    Args:
        trade_returns: 개별 거래 수익률 시리즈

    Returns:
        승률 (%)
    """
    if len(trade_returns) == 0:
        return 0.0

    wins = int((trade_returns > 0).sum())
    total = len(trade_returns)

    return (wins / total) * 100


def calculate_profit_factor(
    trade_returns: pd.Series,
) -> float | None:
    """수익 팩터 계산.

    총 수익 / 총 손실 비율입니다.

    Args:
        trade_returns: 개별 거래 수익률 시리즈

    Returns:
        수익 팩터 또는 None
    """
    if len(trade_returns) == 0:
        return None

    gross_profit_value = float(trade_returns[trade_returns > 0].sum())
    gross_loss_value = float(abs(trade_returns[trade_returns < 0].sum()))

    if gross_loss_value == 0:
        return float("inf") if gross_profit_value > 0 else None

    return gross_profit_value / gross_loss_value


def calculate_volatility(
    returns: pd.Series,
    periods_per_year: int = 8760,
) -> float:
    """연환산 변동성 계산.

    Args:
        returns: 수익률 시리즈
        periods_per_year: 연간 기간 수

    Returns:
        연환산 변동성 (%)
    """
    if len(returns) == 0:
        return 0.0

    std_value = float(returns.std())
    vol = std_value * np.sqrt(periods_per_year) * 100
    return vol


MIN_SAMPLES_SKEWNESS = 3
MIN_SAMPLES_KURTOSIS = 4


def calculate_skewness(returns: pd.Series) -> float:
    """왜도 (Skewness) 계산.

    Args:
        returns: 수익률 시리즈

    Returns:
        왜도
    """
    if len(returns) < MIN_SAMPLES_SKEWNESS:
        return 0.0
    skew_value = returns.skew()
    return float(skew_value)


def calculate_kurtosis(returns: pd.Series) -> float:
    """첨도 (Kurtosis) 계산.

    초과 첨도 (Excess Kurtosis)를 반환합니다.
    정규분포는 0입니다.

    Args:
        returns: 수익률 시리즈

    Returns:
        첨도
    """
    if len(returns) < MIN_SAMPLES_KURTOSIS:
        return 0.0
    kurtosis_value = returns.kurtosis()
    return float(kurtosis_value)


def calculate_drawdown_series(
    returns: pd.Series | None = None,
    equity_curve: pd.Series | None = None,
) -> pd.Series:
    """낙폭 시리즈 계산.

    각 시점에서의 전고점 대비 낙폭을 계산합니다.

    Args:
        returns: 수익률 시리즈
        equity_curve: 자산 곡선

    Returns:
        낙폭 시리즈 (%, 음수 또는 0)
    """
    curve: pd.Series
    if equity_curve is not None:
        curve = equity_curve
    elif returns is not None:
        curve = (1 + returns.fillna(0)).cumprod()
    else:
        return pd.Series(dtype=float)

    running_max = curve.cummax()
    drawdown = (curve - running_max) / running_max * 100

    return drawdown


def calculate_underwater_periods(
    drawdown_series: pd.Series,
    threshold: float = -1.0,  # 1% 이상 손실
) -> pd.DataFrame:
    """수면 아래 기간 분석.

    연속적인 낙폭 기간을 분석합니다.

    Args:
        drawdown_series: 낙폭 시리즈 (%, 음수)
        threshold: 낙폭 임계값 (%)

    Returns:
        수면 아래 기간 분석 DataFrame
    """
    is_underwater = drawdown_series < threshold

    # 연속 기간 식별
    underwater_groups = (is_underwater != is_underwater.shift()).cumsum()
    underwater_groups = underwater_groups[is_underwater]

    if len(underwater_groups) == 0:
        return pd.DataFrame({"start": [], "end": [], "duration": [], "max_dd": []})

    periods: list[dict[str, object]] = []
    for group_id in underwater_groups.unique():
        group: pd.Series = drawdown_series[underwater_groups == group_id]  # type: ignore[assignment]
        periods.append({
            "start": group.index[0],
            "end": group.index[-1],
            "duration": len(group),
            "max_dd": group.min(),
        })

    return pd.DataFrame(periods)


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 168,  # 1주일 (시간봉)
    risk_free_rate: float = 0.05,
    periods_per_year: int = 8760,
) -> pd.Series:
    """롤링 샤프 비율 계산.

    Args:
        returns: 수익률 시리즈
        window: 롤링 윈도우
        risk_free_rate: 무위험 수익률
        periods_per_year: 연간 기간 수

    Returns:
        롤링 샤프 비율 시리즈
    """
    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period

    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = excess_returns.rolling(window=window).std()

    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)
    return rolling_sharpe


def calculate_all_metrics(
    returns: pd.Series,
    trade_returns: pd.Series | None = None,
    periods_per_year: int = 8760,
    risk_free_rate: float = 0.05,
) -> dict[str, float | None]:
    """모든 성과 지표 계산.

    Args:
        returns: 수익률 시리즈
        trade_returns: 개별 거래 수익률 시리즈 (선택적)
        periods_per_year: 연간 기간 수
        risk_free_rate: 무위험 수익률

    Returns:
        성과 지표 딕셔너리
    """
    total_return = calculate_total_return(returns)
    cagr = calculate_cagr(returns, periods_per_year)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    max_dd = calculate_max_drawdown(returns)
    calmar = calculate_calmar_ratio(cagr, max_dd)
    volatility = calculate_volatility(returns, periods_per_year)
    skewness = calculate_skewness(returns)
    kurtosis = calculate_kurtosis(returns)

    # 거래 기반 지표 (있으면)
    win_rate = None
    profit_factor = None
    total_trades = 0

    if trade_returns is not None and len(trade_returns) > 0:
        win_rate = calculate_win_rate(trade_returns)
        profit_factor = calculate_profit_factor(trade_returns)
        total_trades = len(trade_returns)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "volatility": volatility,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
    }
