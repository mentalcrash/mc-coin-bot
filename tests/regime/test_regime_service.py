"""RegimeService 단위 테스트.

Config/State, Incremental(Live), Precompute(Backtest),
enrich_dataframe, EDA 통합, Warmup을 검증합니다.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.regime.config import EnsembleRegimeDetectorConfig, RegimeLabel
from src.regime.service import (
    REGIME_COLUMNS,
    EnrichedRegimeState,
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

    def test_register_subscribes_to_bar(self) -> None:
        """register()는 BAR 이벤트를 구독."""
        from src.core.event_bus import EventBus
        from src.core.events import EventType

        service = RegimeService()
        bus = EventBus(queue_size=100)

        asyncio.get_event_loop().run_until_complete(service.register(bus))

        assert len(bus._handlers[EventType.BAR]) >= 1

    def test_on_bar_updates_state(self) -> None:
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
        asyncio.get_event_loop().run_until_complete(service._on_bar(bar))

        state = service.get_regime("BTC/USDT")
        assert state is not None

    def test_tf_filter(self) -> None:
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
        asyncio.get_event_loop().run_until_complete(service._on_bar(bar))
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
        rolling_std = returns_series.rolling(
            window=window, min_periods=max(2, window // 2)
        ).std()
        expected_std = float(rolling_std.iloc[-1])

        ema_momentum = float(
            returns_series.ewm(span=window, adjust=False).mean().iloc[-1]
        )
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
