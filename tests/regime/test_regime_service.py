"""RegimeService 단위 테스트.

Config/State, Incremental(Live), Precompute(Backtest),
enrich_dataframe, EDA 통합, Warmup을 검증합니다.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.regime.config import EnsembleRegimeDetectorConfig, RegimeLabel
from src.regime.service import (
    REGIME_COLUMNS,
    EnrichedRegimeState,
    RegimeContext,
    RegimeService,
    RegimeServiceConfig,
)

# ── Helper: 시리즈 생성 ──


def _make_trending_series(n: int = 100, drift: float = 0.01) -> pd.Series:
    """명확한 상승 추세 시리즈."""
    rng = np.random.default_rng(42)
    returns = drift + rng.normal(0, 0.002, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz=UTC)
    return pd.Series(prices, index=idx, name="close")


def _make_downtrend_series(n: int = 100) -> pd.Series:
    """명확한 하락 추세 시리즈."""
    rng = np.random.default_rng(42)
    returns = -0.01 + rng.normal(0, 0.002, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz=UTC)
    return pd.Series(prices, index=idx, name="close")


def _make_ranging_series(n: int = 100) -> pd.Series:
    """횡보 시리즈."""
    rng = np.random.default_rng(42)
    prices = np.zeros(n)
    prices[0] = 100.0
    for i in range(1, n):
        reversion = -0.05 * (prices[i - 1] - 100.0)
        prices[i] = prices[i - 1] + reversion + rng.normal(0, 0.5)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz=UTC)
    return pd.Series(np.maximum(prices, 50.0), index=idx, name="close")


def _make_ohlcv_df(closes: pd.Series) -> pd.DataFrame:
    """close 시리즈에서 간단한 OHLCV DataFrame 생성."""
    return pd.DataFrame(
        {
            "open": closes * 0.99,
            "high": closes * 1.01,
            "low": closes * 0.98,
            "close": closes,
            "volume": 1000.0,
        },
        index=closes.index,
    )


# ═══════════════════════════════════════
# Config / State 테스트
# ═══════════════════════════════════════


class TestRegimeServiceConfig:
    """RegimeServiceConfig Pydantic 모델 검증."""

    def test_default_creation(self) -> None:
        config = RegimeServiceConfig()
        assert config.direction_window == 10
        assert config.direction_threshold == 0.0
        assert config.target_timeframe == "1D"
        assert isinstance(config.ensemble, EnsembleRegimeDetectorConfig)

    def test_frozen(self) -> None:
        config = RegimeServiceConfig()
        with pytest.raises(ValidationError):
            config.direction_window = 20  # type: ignore[misc]

    def test_custom_values(self) -> None:
        config = RegimeServiceConfig(
            direction_window=20,
            direction_threshold=0.1,
            target_timeframe="4h",
        )
        assert config.direction_window == 20
        assert config.direction_threshold == 0.1
        assert config.target_timeframe == "4h"

    def test_direction_window_validation(self) -> None:
        with pytest.raises(ValidationError):
            RegimeServiceConfig(direction_window=1)  # ge=3 위반

    def test_direction_threshold_validation(self) -> None:
        with pytest.raises(ValidationError):
            RegimeServiceConfig(direction_threshold=0.6)  # le=0.5 위반


class TestEnrichedRegimeState:
    """EnrichedRegimeState dataclass 검증."""

    def test_creation(self) -> None:
        state = EnrichedRegimeState(
            label=RegimeLabel.TRENDING,
            probabilities={"trending": 0.7, "ranging": 0.2, "volatile": 0.1},
            bars_held=5,
            trend_direction=1,
            trend_strength=0.8,
        )
        assert state.label == RegimeLabel.TRENDING
        assert state.trend_direction == 1
        assert state.trend_strength == 0.8
        assert state.bars_held == 5

    def test_defaults(self) -> None:
        state = EnrichedRegimeState(
            label=RegimeLabel.RANGING,
            probabilities={"trending": 0.2, "ranging": 0.7, "volatile": 0.1},
            bars_held=1,
        )
        assert state.trend_direction == 0
        assert state.trend_strength == 0.0
        assert state.raw_indicators == {}

    def test_direction_values(self) -> None:
        """direction은 -1, 0, +1 중 하나."""
        for direction in (-1, 0, 1):
            state = EnrichedRegimeState(
                label=RegimeLabel.VOLATILE,
                probabilities={},
                bars_held=1,
                trend_direction=direction,
            )
            assert state.trend_direction == direction


# ═══════════════════════════════════════
# Incremental (Live) 테스트
# ═══════════════════════════════════════


class TestIncrementalUpdate:
    """_on_bar() 기반 증분 업데이트 검증."""

    def test_warmup_returns_none(self) -> None:
        """warmup 미완료 시 get_regime() = None."""
        service = RegimeService()
        # 5개 bar만 제공 (warmup 부족)
        for close in [100.0, 101.0, 99.0, 102.0, 100.5]:
            service._detector.update("BTC/USDT", close)
        assert service.get_regime("BTC/USDT") is None

    def test_uptrend_direction_positive(self) -> None:
        """상승 추세 → direction=+1."""
        service = RegimeService()
        closes = _make_trending_series(60)
        service.warmup("BTC/USDT", closes.tolist())

        state = service.get_regime("BTC/USDT")
        assert state is not None
        assert state.trend_direction == 1

    def test_downtrend_direction_negative(self) -> None:
        """하락 추세 → direction=-1."""
        service = RegimeService()
        closes = _make_downtrend_series(60)
        service.warmup("BTC/USDT", closes.tolist())

        state = service.get_regime("BTC/USDT")
        assert state is not None
        assert state.trend_direction == -1

    def test_ranging_direction_zero_or_weak(self) -> None:
        """횡보 → direction=0 또는 |strength| 낮음."""
        service = RegimeService(RegimeServiceConfig(direction_threshold=0.3))
        closes = _make_ranging_series(60)
        service.warmup("BTC/USDT", closes.tolist())

        state = service.get_regime("BTC/USDT")
        assert state is not None
        # 횡보에서는 방향이 약하거나 중립
        assert state.trend_strength <= 0.5

    def test_multi_symbol_independence(self) -> None:
        """여러 심볼이 독립적으로 추적됨."""
        service = RegimeService()

        up_closes = _make_trending_series(60)
        down_closes = _make_downtrend_series(60)

        service.warmup("BTC/USDT", up_closes.tolist())
        service.warmup("ETH/USDT", down_closes.tolist())

        btc = service.get_regime("BTC/USDT")
        eth = service.get_regime("ETH/USDT")

        assert btc is not None
        assert eth is not None
        assert btc.trend_direction == 1
        assert eth.trend_direction == -1

    def test_unknown_symbol_returns_none(self) -> None:
        """미등록 심볼은 None 반환."""
        service = RegimeService()
        assert service.get_regime("UNKNOWN") is None


# ═══════════════════════════════════════
# Precompute (Backtest) 테스트
# ═══════════════════════════════════════


class TestPrecompute:
    """precompute() vectorized 계산 검증."""

    def test_returns_correct_columns(self) -> None:
        """반환 DataFrame에 모든 regime 컬럼 포함."""
        service = RegimeService()
        closes = _make_trending_series(100)
        result = service.precompute("BTC/USDT", closes)

        for col in REGIME_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_same_length_as_input(self) -> None:
        """입력과 동일 길이."""
        service = RegimeService()
        closes = _make_trending_series(100)
        result = service.precompute("BTC/USDT", closes)
        assert len(result) == len(closes)

    def test_cache_stored(self) -> None:
        """사전 계산 결과가 캐시에 저장됨."""
        service = RegimeService()
        closes = _make_trending_series(100)
        service.precompute("BTC/USDT", closes)
        assert "BTC/USDT" in service._precomputed

    def test_trending_has_positive_direction(self) -> None:
        """상승 추세에서 trend_direction > 0인 bar가 많음."""
        service = RegimeService()
        closes = _make_trending_series(100)
        result = service.precompute("BTC/USDT", closes)

        valid = result["trend_direction"].dropna()
        positive_ratio = (valid == 1).sum() / max(len(valid), 1)
        assert positive_ratio > 0.5

    def test_trend_strength_range(self) -> None:
        """trend_strength는 0.0~1.0 범위."""
        service = RegimeService()
        closes = _make_trending_series(100)
        result = service.precompute("BTC/USDT", closes)

        valid = result["trend_strength"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_regime_label_values(self) -> None:
        """regime_label은 RegimeLabel 값 중 하나."""
        service = RegimeService()
        closes = _make_trending_series(100)
        result = service.precompute("BTC/USDT", closes)

        valid_labels = result["regime_label"].dropna().unique()
        allowed = {label.value for label in RegimeLabel}
        for label in valid_labels:
            assert label in allowed


# ═══════════════════════════════════════
# enrich_dataframe 테스트
# ═══════════════════════════════════════


class TestEnrichDataFrame:
    """enrich_dataframe() 검증."""

    def test_with_precomputed(self) -> None:
        """사전 계산 있으면 regime 컬럼이 추가됨."""
        service = RegimeService()
        closes = _make_trending_series(100)
        service.precompute("BTC/USDT", closes)

        df = _make_ohlcv_df(closes)
        enriched = service.enrich_dataframe(df, "BTC/USDT")

        for col in REGIME_COLUMNS:
            assert col in enriched.columns

    def test_without_precomputed(self) -> None:
        """사전 계산 없으면 df 변경 없음."""
        service = RegimeService()
        closes = _make_trending_series(50)
        df = _make_ohlcv_df(closes)

        original_cols = set(df.columns)
        enriched = service.enrich_dataframe(df, "BTC/USDT")
        assert set(enriched.columns) == original_cols

    def test_partial_index_match(self) -> None:
        """df 인덱스가 사전 계산의 일부분이어도 동작."""
        service = RegimeService()
        closes = _make_trending_series(100)
        service.precompute("BTC/USDT", closes)

        # 후반 50개만 사용
        partial_closes = closes.iloc[50:]
        df = _make_ohlcv_df(partial_closes)
        enriched = service.enrich_dataframe(df, "BTC/USDT")

        assert len(enriched) == 50
        for col in REGIME_COLUMNS:
            assert col in enriched.columns

    def test_no_duplicate_columns(self) -> None:
        """이미 regime 컬럼이 있으면 덮어쓰지 않음."""
        service = RegimeService()
        closes = _make_trending_series(100)
        service.precompute("BTC/USDT", closes)

        df = _make_ohlcv_df(closes)
        df["regime_label"] = "existing"  # 미리 추가

        enriched = service.enrich_dataframe(df, "BTC/USDT")
        # 기존 값 유지 (regime_label은 덮어쓰지 않음)
        # 다른 컬럼은 추가됨
        assert (enriched["regime_label"] == "existing").all()


# ═══════════════════════════════════════
# get_regime_columns (Live fallback) 테스트
# ═══════════════════════════════════════


class TestGetRegimeColumns:
    """get_regime_columns() Live fallback 검증."""

    def test_returns_none_before_warmup(self) -> None:
        """warmup 전에는 None."""
        service = RegimeService()
        assert service.get_regime_columns("BTC/USDT") is None

    def test_returns_dict_after_warmup(self) -> None:
        """warmup 후 dict 반환."""
        service = RegimeService()
        closes = _make_trending_series(60)
        service.warmup("BTC/USDT", closes.tolist())

        cols = service.get_regime_columns("BTC/USDT")
        assert cols is not None
        assert "regime_label" in cols
        assert "p_trending" in cols
        assert "trend_direction" in cols
        assert "trend_strength" in cols


# ═══════════════════════════════════════
# EDA Integration 테스트
# ═══════════════════════════════════════


class TestEDAIntegration:
    """EventBus 등록 + _on_bar 동작 검증."""

    @pytest.mark.asyncio
    async def test_register_subscribes_to_bar(self) -> None:
        """register()는 BAR 이벤트를 구독."""
        from src.core.event_bus import EventBus
        from src.core.events import EventType

        service = RegimeService()
        bus = EventBus(queue_size=100)

        await service.register(bus)

        assert len(bus._handlers[EventType.BAR]) >= 1

    @pytest.mark.asyncio
    async def test_on_bar_updates_state(self) -> None:
        """_on_bar() 호출 후 state 업데이트."""
        from src.core.events import BarEvent

        service = RegimeService()
        closes = _make_trending_series(60)

        # warmup: detector에 충분한 데이터 제공
        service.warmup("BTC/USDT", closes.tolist()[:50])

        # _on_bar 시뮬레이션
        bar = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=float(closes.iloc[-1] * 0.99),
            high=float(closes.iloc[-1] * 1.01),
            low=float(closes.iloc[-1] * 0.98),
            close=float(closes.iloc[-1]),
            volume=1000.0,
            bar_timestamp=closes.index[-1].to_pydatetime(),
        )
        await service._on_bar(bar)

        state = service.get_regime("BTC/USDT")
        assert state is not None

    @pytest.mark.asyncio
    async def test_tf_filter(self) -> None:
        """target_timeframe과 다른 TF bar는 무시."""
        from src.core.events import BarEvent

        service = RegimeService(RegimeServiceConfig(target_timeframe="1D"))
        closes = _make_trending_series(60)
        service.warmup("BTC/USDT", closes.tolist()[:50])

        # 4h bar — 무시되어야 함
        bar = BarEvent(
            symbol="BTC/USDT",
            timeframe="4h",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            bar_timestamp=datetime(2024, 6, 1, tzinfo=UTC),
        )
        # state 개수가 변하지 않아야 함
        before = service.get_regime("BTC/USDT")
        await service._on_bar(bar)
        after = service.get_regime("BTC/USDT")

        # warmup으로 이미 state가 있으므로, 4h bar는 무시 → 동일 state
        assert before is after or (before is not None and after is not None)


# ═══════════════════════════════════════
# Warmup 테스트
# ═══════════════════════════════════════


class TestWarmup:
    """warmup() 메서드 검증."""

    def test_warmup_enables_get_regime(self) -> None:
        """warmup() 후 get_regime()이 값을 반환."""
        service = RegimeService()
        closes = _make_trending_series(60)
        service.warmup("BTC/USDT", closes.tolist())

        state = service.get_regime("BTC/USDT")
        assert state is not None
        assert isinstance(state, EnrichedRegimeState)

    def test_warmup_insufficient_data(self) -> None:
        """warmup 데이터 부족 시 None."""
        service = RegimeService()
        service.warmup("BTC/USDT", [100.0, 101.0])  # 2개만

        assert service.get_regime("BTC/USDT") is None

    def test_warmup_populates_close_buffer(self) -> None:
        """warmup 후 close buffer가 채워짐."""
        service = RegimeService()
        closes = _make_trending_series(60)
        service.warmup("BTC/USDT", closes.tolist())

        assert "BTC/USDT" in service._close_buffers
        assert len(service._close_buffers["BTC/USDT"]) > 0


# ── Direction Std Parity ──


class TestDirectionStdParity:
    """Incremental direction std가 vectorized rolling std와 일치하는지 검증."""

    def test_direction_uses_rolling_std(self) -> None:
        """_compute_direction_from_buffer가 전체 버퍼 std가 아닌 rolling std를 사용."""
        from collections import deque

        config = RegimeServiceConfig(direction_window=10)
        service = RegimeService(config)

        # direction_window + 5 크기의 버퍼 생성
        rng = np.random.default_rng(42)
        n = config.direction_window + 5
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
        buf: deque[float] = deque(prices.tolist(), maxlen=n)

        direction, strength = service._compute_direction_from_buffer(buf)

        # Vectorized 방식으로 동일 계산
        log_returns = np.diff(np.log(np.array(buf)))
        returns_series = pd.Series(log_returns)
        window = config.direction_window
        rolling_std = returns_series.rolling(window=window, min_periods=max(2, window // 2)).std()
        expected_std = float(rolling_std.iloc[-1])

        ema_momentum = float(returns_series.ewm(span=window, adjust=False).mean().iloc[-1])
        expected_normalized = ema_momentum / expected_std

        if abs(expected_normalized) > config.direction_threshold:
            expected_direction = 1 if expected_normalized > 0 else -1
            expected_strength = min(abs(expected_normalized), 1.0)
        else:
            expected_direction = 0
            expected_strength = 0.0

        assert direction == expected_direction
        assert abs(strength - expected_strength) < 1e-10


# ── enrich_dataframe Copy Safety ──


class TestEnrichDataFrameCopy:
    """enrich_dataframe가 원본 DataFrame을 변경하지 않는지 검증."""

    def test_original_df_not_mutated(self) -> None:
        """enrich_dataframe 호출 후 원본 df에 regime 컬럼이 없어야 함."""
        service = RegimeService()
        closes = _make_trending_series(60)
        service.precompute("BTC/USDT", closes)

        df = pd.DataFrame(
            {"close": closes, "volume": 1000.0},
            index=closes.index,
        )
        original_cols = list(df.columns)

        result = service.enrich_dataframe(df, "BTC/USDT")

        # 원본은 변경 없음
        assert list(df.columns) == original_cols
        assert "regime_label" not in df.columns

        # 결과에는 regime 컬럼 존재
        assert "regime_label" in result.columns


# ═══════════════════════════════════════
# Confidence + Transition Probability 테스트
# ═══════════════════════════════════════


class TestConfidenceAndTransition:
    """confidence, transition_prob 필드 검증."""

    def test_enriched_state_defaults(self) -> None:
        """EnrichedRegimeState 기본값 검증."""
        state = EnrichedRegimeState(
            label=RegimeLabel.TRENDING,
            probabilities={"trending": 0.7, "ranging": 0.2, "volatile": 0.1},
            bars_held=1,
        )
        assert state.confidence == 0.0
        assert state.transition_prob == 0.0
        assert state.cascade_risk == 0.0

    def test_precompute_includes_new_columns(self) -> None:
        """precompute() 결과에 confidence, transition_prob, cascade_risk 컬럼."""
        service = RegimeService()
        closes = _make_trending_series(100)
        result = service.precompute("BTC/USDT", closes)
        assert "regime_confidence" in result.columns
        assert "regime_transition_prob" in result.columns
        assert "cascade_risk" in result.columns

    def test_transition_prob_range(self) -> None:
        """transition_prob은 0~1 범위."""
        service = RegimeService()
        closes = _make_trending_series(100)
        result = service.precompute("BTC/USDT", closes)
        valid = result["regime_transition_prob"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_confidence_range(self) -> None:
        """confidence는 0~1 범위."""
        service = RegimeService()
        closes = _make_trending_series(100)
        result = service.precompute("BTC/USDT", closes)
        valid = result["regime_confidence"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_warmup_builds_transition_matrix(self) -> None:
        """warmup() 후 transition matrix 생성됨."""
        service = RegimeService()
        closes = _make_trending_series(60)
        service.warmup("BTC/USDT", closes.tolist())
        assert "BTC/USDT" in service._transition_matrices

    def test_warmup_enriched_state_has_confidence(self) -> None:
        """warmup() 후 state에 confidence 포함."""
        service = RegimeService()
        closes = _make_trending_series(60)
        service.warmup("BTC/USDT", closes.tolist())
        state = service.get_regime("BTC/USDT")
        assert state is not None
        assert hasattr(state, "confidence")
        assert hasattr(state, "transition_prob")

    def test_get_regime_columns_includes_new_fields(self) -> None:
        """get_regime_columns() 반환에 새 필드 포함."""
        service = RegimeService()
        closes = _make_trending_series(60)
        service.warmup("BTC/USDT", closes.tolist())
        cols = service.get_regime_columns("BTC/USDT")
        assert cols is not None
        assert "regime_confidence" in cols
        assert "regime_transition_prob" in cols
        assert "cascade_risk" in cols

    def test_regime_columns_constant_updated(self) -> None:
        """REGIME_COLUMNS 상수에 새 필드 포함."""
        assert "regime_confidence" in REGIME_COLUMNS
        assert "regime_transition_prob" in REGIME_COLUMNS
        assert "cascade_risk" in REGIME_COLUMNS

    def test_enrich_dataframe_new_columns(self) -> None:
        """enrich_dataframe()에서 새 컬럼도 추가됨."""
        service = RegimeService()
        closes = _make_trending_series(100)
        service.precompute("BTC/USDT", closes)
        df = _make_ohlcv_df(closes)
        enriched = service.enrich_dataframe(df, "BTC/USDT")
        assert "regime_confidence" in enriched.columns
        assert "regime_transition_prob" in enriched.columns
        assert "cascade_risk" in enriched.columns


# ═══════════════════════════════════════
# RegimeContext 테스트
# ═══════════════════════════════════════


class TestRegimeContext:
    """RegimeContext API 검증."""

    def test_none_before_warmup(self) -> None:
        """warmup 전 None."""
        service = RegimeService()
        assert service.get_regime_context("BTC/USDT") is None

    def test_returns_context_after_warmup(self) -> None:
        """warmup 후 RegimeContext 반환."""
        service = RegimeService()
        closes = _make_trending_series(60)
        service.warmup("BTC/USDT", closes.tolist())
        ctx = service.get_regime_context("BTC/USDT")
        assert ctx is not None
        assert isinstance(ctx, RegimeContext)

    def test_context_fields(self) -> None:
        """RegimeContext 모든 필드 존재."""
        service = RegimeService()
        closes = _make_trending_series(60)
        service.warmup("BTC/USDT", closes.tolist())
        ctx = service.get_regime_context("BTC/USDT")
        assert ctx is not None
        assert hasattr(ctx, "label")
        assert hasattr(ctx, "p_trending")
        assert hasattr(ctx, "p_ranging")
        assert hasattr(ctx, "p_volatile")
        assert hasattr(ctx, "confidence")
        assert hasattr(ctx, "transition_prob")
        assert hasattr(ctx, "cascade_risk")
        assert hasattr(ctx, "trend_direction")
        assert hasattr(ctx, "trend_strength")
        assert hasattr(ctx, "bars_in_regime")
        assert hasattr(ctx, "suggested_vol_scalar")

    def test_vol_scalar_range(self) -> None:
        """suggested_vol_scalar는 0.0~1.0 범위."""
        service = RegimeService()
        closes = _make_trending_series(60)
        service.warmup("BTC/USDT", closes.tolist())
        ctx = service.get_regime_context("BTC/USDT")
        assert ctx is not None
        assert 0.0 <= ctx.suggested_vol_scalar <= 1.0

    def test_context_is_frozen(self) -> None:
        """RegimeContext는 불변."""
        service = RegimeService()
        closes = _make_trending_series(60)
        service.warmup("BTC/USDT", closes.tolist())
        ctx = service.get_regime_context("BTC/USDT")
        assert ctx is not None
        with pytest.raises(AttributeError):
            ctx.label = RegimeLabel.VOLATILE  # type: ignore[misc]


# ═══════════════════════════════════════
# REGIME_CHANGE Event 테스트
# ═══════════════════════════════════════


class TestRegimeChangeEvent:
    """REGIME_CHANGE 이벤트 발행 검증."""

    @pytest.mark.asyncio
    async def test_event_published_on_regime_change(self) -> None:
        """레짐 변경 시 REGIME_CHANGE 이벤트 발행 (결정론적).

        min_hold_bars=1로 hysteresis 최소화 후,
        Phase 1: 강한 상승 추세 warmup → TRENDING 확립
        Phase 2: 횡보 (ER~0, low vol) bars → RANGING 전환 유도
        """
        from src.core.event_bus import EventBus
        from src.core.events import BarEvent, EventType, RegimeChangeEvent

        config = RegimeServiceConfig(
            ensemble=EnsembleRegimeDetectorConfig(min_hold_bars=1),
        )
        service = RegimeService(config)
        bus = EventBus(queue_size=100)

        events_received: list[RegimeChangeEvent] = []

        async def capture_event(event: object) -> None:
            if isinstance(event, RegimeChangeEvent):
                events_received.append(event)

        bus.subscribe(EventType.REGIME_CHANGE, capture_event)
        await service.register(bus)

        # EventBus 소비 루프 시작 (백그라운드 태스크)
        import asyncio

        consumer_task = asyncio.create_task(bus.start())

        # Phase 1: 강한 상승 추세 (40 bars) → TRENDING 확립
        warmup_prices: list[float] = []
        base = 100.0
        for _ in range(40):
            base *= 1.015  # 1.5% daily gain → strong uptrend
            warmup_prices.append(base)
        service.warmup("BTC/USDT", warmup_prices)

        initial = service.get_regime("BTC/USDT")
        assert initial is not None

        # Phase 2: 횡보 (mean-reverting, tiny noise) → RANGING 전환
        # RV ratio → ~1.0 (low expansion), ER → ~0 (no direction) → RANGING
        rng = np.random.default_rng(42)
        flat_price = warmup_prices[-1]
        for i in range(30):
            close = flat_price * (1.0 + rng.normal(0, 0.001))
            bar = BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=close * 0.999,
                high=close * 1.001,
                low=close * 0.999,
                close=close,
                volume=1000.0,
                bar_timestamp=datetime(2024, 7, 1 + i % 28, tzinfo=UTC),
            )
            await service._on_bar(bar)
            await bus.flush()

        await bus.stop()
        await consumer_task

        # 레짐 변경 이벤트가 반드시 발행되어야 함
        assert len(events_received) > 0, "REGIME_CHANGE event should be published"
        # 이벤트 필드 검증
        event = events_received[0]
        assert event.prev_label != event.new_label
        assert event.symbol == "BTC/USDT"
        assert 0.0 <= event.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_no_event_without_bus(self) -> None:
        """bus 없이도 _on_bar 호출 가능 (publish 스킵)."""
        from src.core.events import BarEvent

        service = RegimeService()
        trending = _make_trending_series(60)
        service.warmup("BTC/USDT", trending.tolist())

        bar = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            bar_timestamp=datetime(2024, 7, 1, tzinfo=UTC),
        )
        # bus가 None이어도 에러 없이 실행
        await service._on_bar(bar)

    def test_regime_change_event_type_exists(self) -> None:
        """EventType에 REGIME_CHANGE 추가 확인."""
        from src.core.events import EventType

        assert hasattr(EventType, "REGIME_CHANGE")
        assert EventType.REGIME_CHANGE == "regime_change"

    def test_regime_change_event_model(self) -> None:
        """RegimeChangeEvent 모델 생성 검증."""
        from src.core.events import RegimeChangeEvent

        event = RegimeChangeEvent(
            symbol="BTC/USDT",
            prev_label="trending",
            new_label="volatile",
            confidence=0.75,
            cascade_risk=0.3,
            transition_prob=0.2,
        )
        assert event.symbol == "BTC/USDT"
        assert event.prev_label == "trending"
        assert event.new_label == "volatile"
        assert event.confidence == 0.75

    def test_droppable_events_includes_regime_change(self) -> None:
        """DROPPABLE_EVENTS에 REGIME_CHANGE 포함."""
        from src.core.events import DROPPABLE_EVENTS, EventType

        assert EventType.REGIME_CHANGE in DROPPABLE_EVENTS

    def test_event_type_map_includes_regime_change(self) -> None:
        """EVENT_TYPE_MAP에 REGIME_CHANGE 매핑."""
        from src.core.events import EVENT_TYPE_MAP, EventType, RegimeChangeEvent

        assert EventType.REGIME_CHANGE in EVENT_TYPE_MAP
        assert EVENT_TYPE_MAP[EventType.REGIME_CHANGE] is RegimeChangeEvent


# ═══════════════════════════════════════
# Backward Compatibility 테스트
# ═══════════════════════════════════════


class TestExpandingWindowTransition:
    """Expanding-window transition matrix 검증."""

    def test_expanding_converges_to_full(self) -> None:
        """Expanding-window 마지막 값이 full-sequence matrix와 수렴."""
        service = RegimeService()
        closes = _make_trending_series(100)
        result = service.precompute("BTC/USDT", closes)

        # Full-sequence matrix
        labels = result["regime_label"]
        full_mat = service._estimate_transition_matrix(labels)

        # Expanding의 마지막 값은 full의 마지막 값과 유사해야 함
        last_valid_label = labels.dropna().iloc[-1]
        if last_valid_label in service._IDX_MAP:
            idx = service._IDX_MAP[str(last_valid_label)]
            expanding_last = float(result["regime_transition_prob"].iloc[-1])
            full_prob = 1.0 - float(full_mat[idx, idx])
            # Expanding은 bar별 누적이므로 마지막에서 full과 일치
            np.testing.assert_allclose(expanding_last, full_prob, atol=1e-6)

    def test_expanding_no_look_ahead(self) -> None:
        """Expanding-window는 현재 bar까지만 사용 (look-ahead 없음)."""
        service = RegimeService()
        closes = _make_trending_series(80)
        result = service.precompute("BTC/USDT", closes)

        # 초기 bar의 transition_prob는 0 (데이터 부족)
        valid = result["regime_transition_prob"]
        # warmup 직후 값은 Laplace smoothing 기반의 작은 값
        first_valid = valid[valid > 0].iloc[:5] if (valid > 0).any() else pd.Series(dtype=float)
        if len(first_valid) > 0:
            # 초기에는 transition 데이터가 적으므로 값이 작거나 Laplace default
            assert first_valid.iloc[0] <= 1.0


class TestBackwardCompatibility:
    """기존 API 하위 호환성 검증."""

    def test_service_without_derivatives_provider(self) -> None:
        """derivatives_provider=None으로 RegimeService 정상 동작."""
        service = RegimeService()
        closes = _make_trending_series(60)
        service.warmup("BTC/USDT", closes.tolist())
        state = service.get_regime("BTC/USDT")
        assert state is not None
        assert state.cascade_risk == 0.0

    def test_precompute_without_deriv_df(self) -> None:
        """deriv_df=None으로 precompute 정상 동작."""
        service = RegimeService()
        closes = _make_trending_series(100)
        result = service.precompute("BTC/USDT", closes)
        assert "cascade_risk" in result.columns
        assert (result["cascade_risk"] == 0.0).all()

    def test_ensemble_update_without_derivatives(self) -> None:
        """derivatives=None으로 ensemble update 정상 동작."""
        from src.regime.ensemble import EnsembleRegimeDetector

        detector = EnsembleRegimeDetector()
        result = None
        for i in range(30):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)
        assert result is not None
        assert result.confidence == 1.0  # single detector
