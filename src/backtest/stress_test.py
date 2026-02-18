"""Stress Test Framework -- 합성 충격 주입 스트레스 테스트.

과거 데이터에 합성 충격(Black Swan, Liquidity Crisis, Funding Spike, Flash Crash)을
주입하여 전략/포트폴리오의 생존을 검증합니다.

Rules Applied:
    - BacktestEngine 직접 수정 없음 (DataFrame 변형만)
    - Pydantic V2 모델, frozen=True
    - injection_bar=None 시 DataFrame 중간 지점에 충격

Usage:
    >>> from src.backtest.stress_test import inject_shock, run_stress_test, BLACK_SWAN
    >>> shocked_df = inject_shock(df, BLACK_SWAN)
    >>> result = run_stress_test(df, BLACK_SWAN, strategy_fn)
    >>> print(result.survived, result.max_drawdown)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

# ─── Models ──────────────────────────────────────────────────────────


class StressScenario(BaseModel):
    """합성 충격 시나리오 정의.

    Attributes:
        name: 시나리오 이름
        price_shock_pct: 가격 충격 비율 (예: -0.30 = -30%)
        spread_multiplier: 스프레드(high-low) 배율
        volume_reduction_pct: 거래량 감소 비율 (0.0~1.0)
        duration_bars: 충격 지속 바 수
        funding_rate_override: 펀딩비 강제 설정 (None이면 미적용)
    """

    model_config = ConfigDict(frozen=True)

    name: str
    price_shock_pct: float = Field(ge=-1.0, le=1.0)
    spread_multiplier: float = Field(ge=1.0)
    volume_reduction_pct: float = Field(ge=0.0, le=1.0)
    duration_bars: int = Field(ge=1)
    funding_rate_override: float | None = None


class StressTestResult(BaseModel):
    """스트레스 테스트 결과.

    Attributes:
        scenario_name: 시나리오 이름
        survived: 생존 여부 (equity > 0)
        min_equity_pct: 최소 equity 비율 (초기 자본 대비 %)
        max_drawdown: 최대 낙폭 (%, 음수)
        bars_to_recover: 충격 후 전고점 회복까지 바 수 (미회복 시 None)
        metrics: 추가 지표 딕셔너리
    """

    model_config = ConfigDict(frozen=True)

    scenario_name: str
    survived: bool
    min_equity_pct: float
    max_drawdown: float
    bars_to_recover: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


# ─── Predefined Scenarios ────────────────────────────────────────────

BLACK_SWAN = StressScenario(
    name="BLACK_SWAN",
    price_shock_pct=-0.30,
    spread_multiplier=5.0,
    volume_reduction_pct=0.5,
    duration_bars=1,
)

LIQUIDITY_CRISIS = StressScenario(
    name="LIQUIDITY_CRISIS",
    price_shock_pct=-0.10,
    spread_multiplier=10.0,
    volume_reduction_pct=0.8,
    duration_bars=5,
)

FUNDING_SPIKE = StressScenario(
    name="FUNDING_SPIKE",
    price_shock_pct=0.0,
    spread_multiplier=1.0,
    volume_reduction_pct=0.0,
    duration_bars=24,
    funding_rate_override=0.003,
)

FLASH_CRASH = StressScenario(
    name="FLASH_CRASH",
    price_shock_pct=-0.15,
    spread_multiplier=3.0,
    volume_reduction_pct=0.3,
    duration_bars=1,
)

ALL_SCENARIOS: list[StressScenario] = [
    BLACK_SWAN,
    LIQUIDITY_CRISIS,
    FUNDING_SPIKE,
    FLASH_CRASH,
]


# ─── Pure Functions ──────────────────────────────────────────────────

_PRICE_COLS = ["open", "high", "low", "close"]


def inject_shock(
    df: pd.DataFrame,
    scenario: StressScenario,
    injection_bar: int | None = None,
) -> pd.DataFrame:
    """OHLCV DataFrame에 합성 충격을 주입.

    원본 DataFrame을 수정하지 않고, 충격이 적용된 복사본을 반환합니다.

    Args:
        df: OHLCV DataFrame (open, high, low, close, volume 컬럼 필수)
        scenario: 적용할 충격 시나리오
        injection_bar: 충격 시작 바 인덱스. None이면 DataFrame 중간 지점 사용

    Returns:
        충격이 적용된 DataFrame 복사본

    Raises:
        ValueError: injection_bar가 DataFrame 범위를 벗어날 때
    """
    result = df.copy()
    n = len(result)

    if n == 0:
        return result

    # injection_bar 결정
    start = injection_bar if injection_bar is not None else n // 2

    if start < 0 or start >= n:
        msg = f"injection_bar={start} is out of range [0, {n - 1}]"
        raise ValueError(msg)

    # 충격 적용 구간 계산 (DataFrame 범위 내로 클램프)
    end = min(start + scenario.duration_bars, n)

    # 1) 가격 충격 적용
    if scenario.price_shock_pct != 0.0:
        shock_factor = 1.0 + scenario.price_shock_pct
        for col in _PRICE_COLS:
            if col in result.columns:
                result.iloc[start:end, result.columns.get_loc(col)] = (
                    result.iloc[start:end][col].to_numpy() * shock_factor
                )

    # 2) 스프레드 확대
    if scenario.spread_multiplier > 1.0 and "high" in result.columns and "low" in result.columns:
        for i in range(start, end):
            mid = (result.iloc[i]["high"] + result.iloc[i]["low"]) / 2.0
            half_spread = (result.iloc[i]["high"] - result.iloc[i]["low"]) / 2.0
            expanded_half = half_spread * scenario.spread_multiplier
            result.iloc[i, result.columns.get_loc("high")] = mid + expanded_half
            result.iloc[i, result.columns.get_loc("low")] = mid - expanded_half

    # 3) 거래량 감소
    if scenario.volume_reduction_pct > 0.0 and "volume" in result.columns:
        volume_factor = 1.0 - scenario.volume_reduction_pct
        result.iloc[start:end, result.columns.get_loc("volume")] = (
            result.iloc[start:end]["volume"].to_numpy() * volume_factor
        )

    # 4) Flash Crash 특수 처리: 충격 직후 바에서 가격 회복
    is_flash_crash = scenario.name == "FLASH_CRASH" and scenario.price_shock_pct != 0.0
    if is_flash_crash and end < n:
        # 충격 전 가격 수준으로 회복
        pre_shock_close = df.iloc[start - 1]["close"] if start > 0 else df.iloc[0]["close"]
        recovery_bar = end
        for col in _PRICE_COLS:
            if col in result.columns:
                result.iloc[recovery_bar, result.columns.get_loc(col)] = pre_shock_close

    return result


def _calculate_equity_curve(
    prices: pd.Series,
    weights: pd.Series,
    initial_capital: float,
) -> pd.Series:
    """가격과 weight로 equity curve 계산.

    Args:
        prices: close 가격 시리즈
        weights: 포지션 weight 시리즈 (1.0=fully long, -1.0=fully short, 0=flat)
        initial_capital: 초기 자본

    Returns:
        equity curve 시리즈
    """
    returns = prices.pct_change().fillna(0.0)
    # weight는 이전 바 기준 (lookahead bias 방지)
    shifted_weights = weights.shift(1).fillna(0.0)
    strategy_returns = returns * shifted_weights
    cumulative = (strategy_returns + 1.0).cumprod()
    equity = cumulative * initial_capital
    return equity


def _calculate_max_drawdown(equity: pd.Series) -> float:
    """Equity curve에서 최대 낙폭 계산 (%, 음수).

    Args:
        equity: equity curve 시리즈

    Returns:
        최대 낙폭 (%, 음수)
    """
    if len(equity) == 0:
        return 0.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min()) * 100.0


def _calculate_bars_to_recover(
    equity: pd.Series,
    injection_bar: int,
) -> int | None:
    """충격 후 전고점 회복까지의 바 수 계산.

    Args:
        equity: equity curve 시리즈
        injection_bar: 충격 시작 바 인덱스

    Returns:
        회복까지 바 수, 미회복 시 None
    """
    if injection_bar >= len(equity):
        return None

    pre_shock_peak = (
        float(equity.iloc[:injection_bar].max()) if injection_bar > 0 else float(equity.iloc[0])
    )

    for i in range(injection_bar, len(equity)):
        if float(equity.iloc[i]) >= pre_shock_peak:
            return i - injection_bar

    return None


def run_stress_test(
    df: pd.DataFrame,
    scenario: StressScenario,
    strategy_fn: Callable[[pd.DataFrame], pd.Series],
    initial_capital: float = 10000.0,
    injection_bar: int | None = None,
) -> StressTestResult:
    """충격 주입 후 간이 백테스트를 실행.

    strategy_fn은 OHLCV DataFrame을 받아 weight Series를 반환해야 합니다.
    (1.0=fully long, -1.0=fully short, 0=flat)

    Args:
        df: OHLCV DataFrame
        scenario: 적용할 충격 시나리오
        strategy_fn: 전략 함수 (DataFrame -> weight Series)
        initial_capital: 초기 자본
        injection_bar: 충격 시작 바 인덱스 (None이면 중간 지점)

    Returns:
        StressTestResult
    """
    n = len(df)
    actual_injection = injection_bar if injection_bar is not None else n // 2

    # 충격 주입
    shocked_df = inject_shock(df, scenario, injection_bar=actual_injection)

    # 전략 실행
    weights = strategy_fn(shocked_df)

    # Equity curve 계산
    close_series: pd.Series = shocked_df["close"]  # type: ignore[assignment]
    equity = _calculate_equity_curve(
        prices=close_series,
        weights=weights,
        initial_capital=initial_capital,
    )

    # 생존 판정
    min_equity = float(equity.min())
    survived = min_equity > 0.0

    # 최소 equity 비율
    min_equity_pct = (min_equity / initial_capital) * 100.0

    # 최대 낙폭
    max_dd = _calculate_max_drawdown(equity)

    # 회복 바 수
    bars_to_recover = _calculate_bars_to_recover(equity, actual_injection)

    # 추가 지표
    final_equity = float(equity.iloc[-1])
    total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100.0

    # 충격 구간 수익률
    shock_end = min(actual_injection + scenario.duration_bars, n)
    equity_at_shock_start = float(equity.iloc[actual_injection])
    equity_at_shock_end = (
        float(equity.iloc[shock_end - 1]) if shock_end > 0 else equity_at_shock_start
    )
    shock_return_pct = (
        ((equity_at_shock_end - equity_at_shock_start) / equity_at_shock_start) * 100.0
        if equity_at_shock_start > 0
        else 0.0
    )

    # Annualized volatility (daily 기준 근사)
    returns = equity.pct_change().dropna()
    annualized_vol = float(np.std(returns)) * np.sqrt(365) * 100.0 if len(returns) > 0 else 0.0

    metrics = {
        "final_equity": final_equity,
        "total_return_pct": round(total_return_pct, 4),
        "shock_return_pct": round(shock_return_pct, 4),
        "annualized_vol": round(annualized_vol, 4),
    }

    return StressTestResult(
        scenario_name=scenario.name,
        survived=survived,
        min_equity_pct=round(min_equity_pct, 4),
        max_drawdown=round(max_dd, 4),
        bars_to_recover=bars_to_recover,
        metrics=metrics,
    )
