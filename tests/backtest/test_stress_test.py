"""Tests for src/backtest/stress_test.py -- Stress Test Framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.stress_test import (
    ALL_SCENARIOS,
    BLACK_SWAN,
    FLASH_CRASH,
    FUNDING_SPIKE,
    LIQUIDITY_CRISIS,
    StressScenario,
    StressTestResult,
    inject_shock,
    run_stress_test,
)

# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """100행 OHLCV DataFrame (가격 ~100, 거래량 1000)."""
    rng = np.random.default_rng(42)
    n = 100
    # 안정적인 가격: 100 근처에서 작은 변동
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + rng.uniform(0.5, 1.5, n)
    low = close - rng.uniform(0.5, 1.5, n)
    open_ = close + rng.uniform(-0.5, 0.5, n)
    volume = np.full(n, 1000.0)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def always_long_strategy() -> object:
    """항상 100% 롱 포지션 전략."""

    def strategy_fn(df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index)

    return strategy_fn


@pytest.fixture
def flat_strategy() -> object:
    """항상 flat(0%) 전략."""

    def strategy_fn(df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=df.index)

    return strategy_fn


# ─── TestStressScenario ──────────────────────────────────────────────


class TestStressScenario:
    """StressScenario 모델 테스트."""

    def test_creation(self) -> None:
        """커스텀 시나리오 생성."""
        scenario = StressScenario(
            name="custom",
            price_shock_pct=-0.20,
            spread_multiplier=2.0,
            volume_reduction_pct=0.5,
            duration_bars=3,
        )
        assert scenario.name == "custom"
        assert scenario.price_shock_pct == -0.20
        assert scenario.spread_multiplier == 2.0
        assert scenario.volume_reduction_pct == 0.5
        assert scenario.duration_bars == 3
        assert scenario.funding_rate_override is None

    def test_creation_with_funding_override(self) -> None:
        """funding_rate_override가 있는 시나리오 생성."""
        scenario = StressScenario(
            name="funding_test",
            price_shock_pct=0.0,
            spread_multiplier=1.0,
            volume_reduction_pct=0.0,
            duration_bars=10,
            funding_rate_override=0.005,
        )
        assert scenario.funding_rate_override == 0.005

    def test_frozen(self) -> None:
        """frozen=True: 필드 수정 불가."""
        with pytest.raises(Exception):  # noqa: B017
            BLACK_SWAN.name = "modified"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "scenario",
        ALL_SCENARIOS,
        ids=[s.name for s in ALL_SCENARIOS],
    )
    def test_predefined_scenarios_valid(self, scenario: StressScenario) -> None:
        """사전 정의 시나리오 유효성."""
        assert scenario.name
        assert -1.0 <= scenario.price_shock_pct <= 1.0
        assert scenario.spread_multiplier >= 1.0
        assert 0.0 <= scenario.volume_reduction_pct <= 1.0
        assert scenario.duration_bars >= 1

    def test_black_swan_values(self) -> None:
        """BLACK_SWAN 상수 값 확인."""
        assert BLACK_SWAN.price_shock_pct == -0.30
        assert BLACK_SWAN.spread_multiplier == 5.0
        assert BLACK_SWAN.volume_reduction_pct == 0.5
        assert BLACK_SWAN.duration_bars == 1

    def test_liquidity_crisis_values(self) -> None:
        """LIQUIDITY_CRISIS 상수 값 확인."""
        assert LIQUIDITY_CRISIS.price_shock_pct == -0.10
        assert LIQUIDITY_CRISIS.spread_multiplier == 10.0
        assert LIQUIDITY_CRISIS.volume_reduction_pct == 0.8
        assert LIQUIDITY_CRISIS.duration_bars == 5

    def test_funding_spike_values(self) -> None:
        """FUNDING_SPIKE 상수 값 확인."""
        assert FUNDING_SPIKE.price_shock_pct == 0.0
        assert FUNDING_SPIKE.funding_rate_override == 0.003
        assert FUNDING_SPIKE.duration_bars == 24

    def test_flash_crash_values(self) -> None:
        """FLASH_CRASH 상수 값 확인."""
        assert FLASH_CRASH.price_shock_pct == -0.15
        assert FLASH_CRASH.spread_multiplier == 3.0
        assert FLASH_CRASH.duration_bars == 1

    def test_all_scenarios_count(self) -> None:
        """ALL_SCENARIOS에 4개 시나리오가 포함."""
        assert len(ALL_SCENARIOS) == 4


# ─── TestStressTestResult ────────────────────────────────────────────


class TestStressTestResult:
    """StressTestResult 모델 테스트."""

    def test_creation(self) -> None:
        """기본 결과 생성."""
        result = StressTestResult(
            scenario_name="test",
            survived=True,
            min_equity_pct=80.0,
            max_drawdown=-15.0,
            bars_to_recover=10,
            metrics={"final_equity": 9500.0},
        )
        assert result.survived is True
        assert result.min_equity_pct == 80.0
        assert result.max_drawdown == -15.0
        assert result.bars_to_recover == 10
        assert result.metrics["final_equity"] == 9500.0

    def test_no_recovery(self) -> None:
        """미회복 시 bars_to_recover=None."""
        result = StressTestResult(
            scenario_name="severe",
            survived=True,
            min_equity_pct=30.0,
            max_drawdown=-70.0,
        )
        assert result.bars_to_recover is None

    def test_frozen(self) -> None:
        """frozen=True: 필드 수정 불가."""
        result = StressTestResult(
            scenario_name="test",
            survived=True,
            min_equity_pct=90.0,
            max_drawdown=-5.0,
        )
        with pytest.raises(Exception):  # noqa: B017
            result.survived = False  # type: ignore[misc]


# ─── TestInjectShock ─────────────────────────────────────────────────


class TestInjectShock:
    """inject_shock 함수 테스트."""

    def test_copy_semantics(self, sample_ohlcv: pd.DataFrame) -> None:
        """원본 DataFrame은 수정되지 않음."""
        original_close = sample_ohlcv["close"].copy()
        _ = inject_shock(sample_ohlcv, BLACK_SWAN)
        pd.testing.assert_series_equal(sample_ohlcv["close"], original_close)

    def test_price_shock_applied(self, sample_ohlcv: pd.DataFrame) -> None:
        """가격 충격이 올바르게 적용됨."""
        injection_bar = 50
        result = inject_shock(sample_ohlcv, BLACK_SWAN, injection_bar=injection_bar)

        original_close = sample_ohlcv.iloc[injection_bar]["close"]
        shocked_close = result.iloc[injection_bar]["close"]

        # -30% 충격
        expected = original_close * (1.0 + BLACK_SWAN.price_shock_pct)
        assert shocked_close == pytest.approx(expected, rel=1e-6)

    def test_volume_reduced(self, sample_ohlcv: pd.DataFrame) -> None:
        """거래량이 올바르게 감소됨."""
        injection_bar = 50
        result = inject_shock(sample_ohlcv, BLACK_SWAN, injection_bar=injection_bar)

        original_vol = sample_ohlcv.iloc[injection_bar]["volume"]
        shocked_vol = result.iloc[injection_bar]["volume"]

        expected = original_vol * (1.0 - BLACK_SWAN.volume_reduction_pct)
        assert shocked_vol == pytest.approx(expected, rel=1e-6)

    def test_spread_multiplied(self, sample_ohlcv: pd.DataFrame) -> None:
        """스프레드가 확대됨."""
        injection_bar = 50
        result = inject_shock(sample_ohlcv, BLACK_SWAN, injection_bar=injection_bar)

        # 원본 mid 계산 (충격 전 가격 기준)
        shocked_factor = 1.0 + BLACK_SWAN.price_shock_pct
        original_high = sample_ohlcv.iloc[injection_bar]["high"] * shocked_factor
        original_low = sample_ohlcv.iloc[injection_bar]["low"] * shocked_factor
        original_spread = original_high - original_low

        shocked_spread = result.iloc[injection_bar]["high"] - result.iloc[injection_bar]["low"]

        # 스프레드가 확대되었는지 확인
        assert shocked_spread > original_spread * (BLACK_SWAN.spread_multiplier - 0.1)

    def test_injection_bar_none_uses_middle(self, sample_ohlcv: pd.DataFrame) -> None:
        """injection_bar=None이면 중간 지점 사용."""
        result = inject_shock(sample_ohlcv, BLACK_SWAN)
        mid = len(sample_ohlcv) // 2

        original_close = sample_ohlcv.iloc[mid]["close"]
        shocked_close = result.iloc[mid]["close"]

        expected = original_close * (1.0 + BLACK_SWAN.price_shock_pct)
        assert shocked_close == pytest.approx(expected, rel=1e-6)

    def test_non_shock_bars_unchanged(self, sample_ohlcv: pd.DataFrame) -> None:
        """충격 구간 외 바는 변경 없음."""
        injection_bar = 50
        result = inject_shock(sample_ohlcv, BLACK_SWAN, injection_bar=injection_bar)

        # 충격 전 구간 (0~49)은 동일
        pd.testing.assert_frame_equal(
            result.iloc[:injection_bar],
            sample_ohlcv.iloc[:injection_bar],
        )

    def test_duration_bars_applied(self, sample_ohlcv: pd.DataFrame) -> None:
        """duration_bars만큼 충격 적용."""
        injection_bar = 40
        result = inject_shock(sample_ohlcv, LIQUIDITY_CRISIS, injection_bar=injection_bar)

        # LIQUIDITY_CRISIS: duration=5, 40~44까지 충격
        for i in range(injection_bar, injection_bar + LIQUIDITY_CRISIS.duration_bars):
            original_close = sample_ohlcv.iloc[i]["close"]
            shocked_close = result.iloc[i]["close"]
            expected = original_close * (1.0 + LIQUIDITY_CRISIS.price_shock_pct)
            assert shocked_close == pytest.approx(expected, rel=1e-6)

    def test_flash_crash_recovery(self, sample_ohlcv: pd.DataFrame) -> None:
        """FLASH_CRASH: 충격 직후 바에서 가격 회복."""
        injection_bar = 50
        result = inject_shock(sample_ohlcv, FLASH_CRASH, injection_bar=injection_bar)

        # 충격 전 가격
        pre_shock_close = sample_ohlcv.iloc[injection_bar - 1]["close"]

        # 충격 바: 가격 하락
        shocked_close = result.iloc[injection_bar]["close"]
        assert shocked_close < pre_shock_close

        # 회복 바 (injection_bar + duration_bars): 충격 전 수준으로 복구
        recovery_bar = injection_bar + FLASH_CRASH.duration_bars
        recovery_close = result.iloc[recovery_bar]["close"]
        assert recovery_close == pytest.approx(pre_shock_close, rel=1e-6)

    def test_funding_spike_no_price_change(self, sample_ohlcv: pd.DataFrame) -> None:
        """FUNDING_SPIKE: price_shock=0이므로 가격 변동 없음."""
        injection_bar = 30
        result = inject_shock(sample_ohlcv, FUNDING_SPIKE, injection_bar=injection_bar)

        # 가격은 동일 (price_shock_pct=0.0, spread_multiplier=1.0)
        pd.testing.assert_series_equal(
            result["close"],
            sample_ohlcv["close"],
        )

    def test_out_of_range_injection_bar(self, sample_ohlcv: pd.DataFrame) -> None:
        """범위 밖 injection_bar는 ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            inject_shock(sample_ohlcv, BLACK_SWAN, injection_bar=200)

    def test_negative_injection_bar(self, sample_ohlcv: pd.DataFrame) -> None:
        """음수 injection_bar는 ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            inject_shock(sample_ohlcv, BLACK_SWAN, injection_bar=-1)

    def test_edge_last_bar(self, sample_ohlcv: pd.DataFrame) -> None:
        """마지막 바에 충격 주입."""
        last_bar = len(sample_ohlcv) - 1
        result = inject_shock(sample_ohlcv, BLACK_SWAN, injection_bar=last_bar)

        original_close = sample_ohlcv.iloc[last_bar]["close"]
        shocked_close = result.iloc[last_bar]["close"]
        expected = original_close * (1.0 + BLACK_SWAN.price_shock_pct)
        assert shocked_close == pytest.approx(expected, rel=1e-6)

    def test_empty_dataframe(self) -> None:
        """빈 DataFrame 처리."""
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = inject_shock(empty_df, BLACK_SWAN)
        assert len(result) == 0

    @pytest.mark.parametrize(
        "scenario",
        ALL_SCENARIOS,
        ids=[s.name for s in ALL_SCENARIOS],
    )
    def test_all_scenarios_produce_valid_output(
        self,
        sample_ohlcv: pd.DataFrame,
        scenario: StressScenario,
    ) -> None:
        """모든 사전 정의 시나리오가 유효한 출력 생성."""
        result = inject_shock(sample_ohlcv, scenario)
        assert len(result) == len(sample_ohlcv)
        assert list(result.columns) == list(sample_ohlcv.columns)


# ─── TestRunStressTest ───────────────────────────────────────────────


class TestRunStressTest:
    """run_stress_test 함수 테스트."""

    def test_survival_on_mild_shock(
        self,
        sample_ohlcv: pd.DataFrame,
        always_long_strategy: object,
    ) -> None:
        """경미한 충격에서 생존."""
        mild = StressScenario(
            name="MILD",
            price_shock_pct=-0.01,
            spread_multiplier=1.0,
            volume_reduction_pct=0.0,
            duration_bars=1,
        )
        result = run_stress_test(
            sample_ohlcv,
            mild,
            always_long_strategy,  # type: ignore[arg-type]
            initial_capital=10000.0,
        )
        assert result.survived is True
        assert result.min_equity_pct > 0.0
        assert result.scenario_name == "MILD"

    def test_drawdown_on_severe_shock(
        self,
        sample_ohlcv: pd.DataFrame,
        always_long_strategy: object,
    ) -> None:
        """심각한 충격에서 큰 drawdown 발생."""
        result = run_stress_test(
            sample_ohlcv,
            BLACK_SWAN,
            always_long_strategy,  # type: ignore[arg-type]
            initial_capital=10000.0,
        )
        # -30% 충격이므로 상당한 drawdown
        assert result.max_drawdown < -10.0
        assert result.scenario_name == "BLACK_SWAN"

    def test_flat_strategy_survives_any_shock(
        self,
        sample_ohlcv: pd.DataFrame,
        flat_strategy: object,
    ) -> None:
        """flat 전략은 어떤 충격에도 equity 불변."""
        result = run_stress_test(
            sample_ohlcv,
            BLACK_SWAN,
            flat_strategy,  # type: ignore[arg-type]
            initial_capital=10000.0,
        )
        assert result.survived is True
        # flat 전략: equity 변동 없음
        assert result.min_equity_pct == pytest.approx(100.0, abs=0.1)
        assert result.max_drawdown == pytest.approx(0.0, abs=0.1)

    def test_flash_crash_recovery_detected(
        self,
        sample_ohlcv: pd.DataFrame,
        always_long_strategy: object,
    ) -> None:
        """FLASH_CRASH 후 회복 감지."""
        result = run_stress_test(
            sample_ohlcv,
            FLASH_CRASH,
            always_long_strategy,  # type: ignore[arg-type]
            initial_capital=10000.0,
        )
        assert result.survived is True
        assert result.scenario_name == "FLASH_CRASH"

    def test_result_has_metrics(
        self,
        sample_ohlcv: pd.DataFrame,
        always_long_strategy: object,
    ) -> None:
        """결과에 추가 지표 포함."""
        result = run_stress_test(
            sample_ohlcv,
            BLACK_SWAN,
            always_long_strategy,  # type: ignore[arg-type]
        )
        assert "final_equity" in result.metrics
        assert "total_return_pct" in result.metrics
        assert "shock_return_pct" in result.metrics
        assert "annualized_vol" in result.metrics

    def test_custom_initial_capital(
        self,
        sample_ohlcv: pd.DataFrame,
        flat_strategy: object,
    ) -> None:
        """초기 자본 커스터마이징."""
        result = run_stress_test(
            sample_ohlcv,
            BLACK_SWAN,
            flat_strategy,  # type: ignore[arg-type]
            initial_capital=50000.0,
        )
        # flat 전략이므로 equity 불변
        assert result.metrics["final_equity"] == pytest.approx(50000.0, abs=1.0)

    def test_custom_injection_bar(
        self,
        sample_ohlcv: pd.DataFrame,
        always_long_strategy: object,
    ) -> None:
        """injection_bar 커스터마이징."""
        result = run_stress_test(
            sample_ohlcv,
            BLACK_SWAN,
            always_long_strategy,  # type: ignore[arg-type]
            injection_bar=20,
        )
        assert result.survived is True
        assert result.scenario_name == "BLACK_SWAN"

    @pytest.mark.parametrize(
        "scenario",
        ALL_SCENARIOS,
        ids=[s.name for s in ALL_SCENARIOS],
    )
    def test_all_predefined_scenarios(
        self,
        sample_ohlcv: pd.DataFrame,
        always_long_strategy: object,
        scenario: StressScenario,
    ) -> None:
        """모든 사전 정의 시나리오 실행 가능."""
        result = run_stress_test(
            sample_ohlcv,
            scenario,
            always_long_strategy,  # type: ignore[arg-type]
        )
        assert isinstance(result, StressTestResult)
        assert result.scenario_name == scenario.name
        assert isinstance(result.survived, bool)
        assert isinstance(result.min_equity_pct, float)
        assert isinstance(result.max_drawdown, float)
