"""Chaos Test -- Funding rate spike 시나리오.

극단적 funding rate 환경에서 스트레스 테스트 프레임워크가 올바르게 동작하는지 검증합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.stress_test import (
    FLASH_CRASH,
    FUNDING_SPIKE,
    StressScenario,
    StressTestResult,
    inject_shock,
    run_stress_test,
)

pytestmark = pytest.mark.chaos


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """100일 합성 OHLCV (상승 추세)."""
    rng = np.random.default_rng(42)
    n = 100
    prices = 50000 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": prices * (1 + rng.uniform(-0.005, 0.005, n)),
            "high": prices * (1 + rng.uniform(0.005, 0.02, n)),
            "low": prices * (1 - rng.uniform(0.005, 0.02, n)),
            "close": prices,
            "volume": rng.uniform(1e6, 1e8, n),
        },
        index=dates,
    )


def _simple_long_strategy(df: pd.DataFrame) -> pd.Series:
    """항상 100% 롱 포지션."""
    return pd.Series(1.0, index=df.index)


def _momentum_strategy(df: pd.DataFrame) -> pd.Series:
    """단순 20일 모멘텀 전략."""
    returns = df["close"].pct_change(20)
    weights = (returns > 0).astype(float)
    return weights


# ── Tests: Funding Spike ──────────────────────────────────────────


class TestFundingSpike:
    """Funding rate spike 시나리오."""

    def test_funding_spike_scenario_defined(self) -> None:
        """FUNDING_SPIKE 시나리오 정의 확인."""
        assert FUNDING_SPIKE.funding_rate_override == 0.003
        assert FUNDING_SPIKE.duration_bars == 24
        assert FUNDING_SPIKE.price_shock_pct == 0.0  # 가격 변화 없음

    def test_funding_spike_no_price_change(self, sample_ohlcv: pd.DataFrame) -> None:
        """FUNDING_SPIKE는 가격에 영향 없음."""
        shocked = inject_shock(sample_ohlcv, FUNDING_SPIKE)
        # price_shock_pct=0이므로 close가 동일해야 함
        pd.testing.assert_series_equal(sample_ohlcv["close"], shocked["close"], check_names=False)

    def test_funding_spike_stress_test(self, sample_ohlcv: pd.DataFrame) -> None:
        """FUNDING_SPIKE + long strategy → survived."""
        result = run_stress_test(
            sample_ohlcv,
            FUNDING_SPIKE,
            _simple_long_strategy,
            initial_capital=10000.0,
        )
        assert isinstance(result, StressTestResult)
        assert result.survived is True
        assert result.scenario_name == "FUNDING_SPIKE"


# ── Tests: Flash Crash ────────────────────────────────────────────


class TestFlashCrash:
    """Flash crash 시나리오."""

    def test_flash_crash_price_recovery(self, sample_ohlcv: pd.DataFrame) -> None:
        """FLASH_CRASH 주입 후 회복 바에서 가격 복원."""
        injection = 50
        shocked = inject_shock(sample_ohlcv, FLASH_CRASH, injection_bar=injection)

        # 충격 바: 가격 하락
        assert shocked.iloc[injection]["close"] < sample_ohlcv.iloc[injection]["close"]

        # 회복 바 (injection + duration_bars): 충격 전 가격 수준으로 복귀
        recovery = injection + FLASH_CRASH.duration_bars
        if recovery < len(sample_ohlcv):
            pre_shock = sample_ohlcv.iloc[injection - 1]["close"]
            assert shocked.iloc[recovery]["close"] == pytest.approx(pre_shock, rel=0.01)

    def test_flash_crash_long_position_survived(self, sample_ohlcv: pd.DataFrame) -> None:
        """Flash crash에서 롱 포지션 생존 확인."""
        result = run_stress_test(
            sample_ohlcv,
            FLASH_CRASH,
            _simple_long_strategy,
            initial_capital=10000.0,
        )
        assert result.survived is True

    def test_flash_crash_drawdown_limited(self, sample_ohlcv: pd.DataFrame) -> None:
        """Flash crash MDD가 충격 크기와 비례."""
        result = run_stress_test(
            sample_ohlcv,
            FLASH_CRASH,
            _simple_long_strategy,
            initial_capital=10000.0,
        )
        # -15% 충격 → MDD ≈ -15% 이내 (비용 감안)
        assert result.max_drawdown < 0  # 음수 (손실)
        assert result.max_drawdown > -30  # -30% 이내


# ── Tests: Custom Scenario ────────────────────────────────────────


class TestCustomScenario:
    """Custom stress scenario 테스트."""

    def test_extreme_shock(self, sample_ohlcv: pd.DataFrame) -> None:
        """-50% 극단 충격 시 포트폴리오 생존 여부."""
        extreme = StressScenario(
            name="EXTREME",
            price_shock_pct=-0.50,
            spread_multiplier=10.0,
            volume_reduction_pct=0.9,
            duration_bars=3,
        )
        result = run_stress_test(
            sample_ohlcv,
            extreme,
            _simple_long_strategy,
            initial_capital=10000.0,
        )
        assert isinstance(result, StressTestResult)
        # -50% 충격 → 생존할 수도 있고 아닐 수도 있음
        # 중요한 것은 에러 없이 결과 반환
        assert result.max_drawdown < 0

    def test_momentum_strategy_resilience(self, sample_ohlcv: pd.DataFrame) -> None:
        """모멘텀 전략의 충격 복원력."""
        result = run_stress_test(
            sample_ohlcv,
            FLASH_CRASH,
            _momentum_strategy,
            initial_capital=10000.0,
        )
        assert isinstance(result, StressTestResult)
        assert result.survived is True

    def test_injection_at_boundary(self, sample_ohlcv: pd.DataFrame) -> None:
        """injection_bar=0 (첫 번째 바)에서 충격."""
        result = run_stress_test(
            sample_ohlcv,
            FLASH_CRASH,
            _simple_long_strategy,
            injection_bar=0,
        )
        assert isinstance(result, StressTestResult)

    def test_injection_at_end(self, sample_ohlcv: pd.DataFrame) -> None:
        """injection_bar=마지막 바에서 충격."""
        last_bar = len(sample_ohlcv) - 1
        result = run_stress_test(
            sample_ohlcv,
            FLASH_CRASH,
            _simple_long_strategy,
            injection_bar=last_bar,
        )
        assert isinstance(result, StressTestResult)

    def test_invalid_injection_bar(self, sample_ohlcv: pd.DataFrame) -> None:
        """범위 밖 injection_bar → ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            inject_shock(sample_ohlcv, FLASH_CRASH, injection_bar=len(sample_ohlcv))
