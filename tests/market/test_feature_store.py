"""Tests for FeatureStore service."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.core.event_bus import EventBus
from src.core.events import BarEvent, EventType
from src.market.feature_store import (
    DEFAULT_SPECS,
    FeatureStore,
    FeatureStoreConfig,
    IndicatorSpec,
    _call_indicator,
    _resolve_column_name,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """200일 OHLCV 샘플 데이터."""
    np.random.seed(42)
    n = 200
    base = 50000.0
    noise = np.cumsum(np.random.randn(n) * 300)
    close = base + noise - noise.mean()
    close = np.maximum(close, base * 0.8)
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 100
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float) * 1000,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D", tz=UTC),
    )


@pytest.fixture
def default_store() -> FeatureStore:
    """기본 설정 FeatureStore."""
    return FeatureStore()


@pytest.fixture
def custom_store() -> FeatureStore:
    """커스텀 설정 FeatureStore."""
    config = FeatureStoreConfig(
        specs=(
            IndicatorSpec("rsi", {"period": 7}, column_name="rsi_7"),
            IndicatorSpec("atr", {"period": 20}),
        ),
        target_timeframe="4h",
    )
    return FeatureStore(config)


# ---------------------------------------------------------------------------
# IndicatorSpec & helpers
# ---------------------------------------------------------------------------


class TestIndicatorSpec:
    """IndicatorSpec dataclass 테스트."""

    def test_frozen(self) -> None:
        """Frozen dataclass는 속성 변경 불가."""
        spec = IndicatorSpec("atr", {"period": 14})
        with pytest.raises(AttributeError):
            spec.name = "rsi"  # type: ignore[misc]

    def test_default_params(self) -> None:
        """params 기본값은 빈 dict."""
        spec = IndicatorSpec("drawdown")
        assert spec.params == {}
        assert spec.column_name is None


class TestResolveColumnName:
    """_resolve_column_name 테스트."""

    def test_explicit_name(self) -> None:
        """column_name 설정 시 그대로 반환."""
        spec = IndicatorSpec("rsi", {"period": 14}, column_name="my_rsi")
        assert _resolve_column_name(spec) == "my_rsi"

    def test_auto_name_with_params(self) -> None:
        """params의 첫 값으로 자동 생성."""
        spec = IndicatorSpec("atr", {"period": 14})
        assert _resolve_column_name(spec) == "atr_14"

    def test_auto_name_no_params(self) -> None:
        """params 없으면 함수명 그대로."""
        spec = IndicatorSpec("drawdown")
        assert _resolve_column_name(spec) == "drawdown"

    def test_auto_name_float_param(self) -> None:
        """float 파라미터도 문자열로 변환."""
        spec = IndicatorSpec("realized_volatility", {"window": 30, "annualization_factor": 365.0})
        assert _resolve_column_name(spec) == "realized_volatility_30"


class TestCallIndicator:
    """_call_indicator 테스트."""

    def test_atr(self, sample_ohlcv: pd.DataFrame) -> None:
        """ATR 지표 계산."""
        spec = IndicatorSpec("atr", {"period": 14})
        result = _call_indicator(spec, sample_ohlcv)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_rsi(self, sample_ohlcv: pd.DataFrame) -> None:
        """RSI 지표 계산."""
        spec = IndicatorSpec("rsi", {"period": 14})
        result = _call_indicator(spec, sample_ohlcv)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_drawdown(self, sample_ohlcv: pd.DataFrame) -> None:
        """Drawdown 지표 계산."""
        spec = IndicatorSpec("drawdown", {})
        result = _call_indicator(spec, sample_ohlcv)
        valid = result.dropna()
        assert (valid <= 0).all()

    def test_realized_volatility(self, sample_ohlcv: pd.DataFrame) -> None:
        """Realized Volatility 계산 (returns 자동 파생)."""
        spec = IndicatorSpec("realized_volatility", {"window": 30, "annualization_factor": 365.0})
        result = _call_indicator(spec, sample_ohlcv)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_tuple_return_takes_first(self, sample_ohlcv: pd.DataFrame) -> None:
        """Tuple 반환 지표(bollinger_bands)는 첫 번째 시리즈만 사용."""
        spec = IndicatorSpec("bollinger_bands", {"period": 20, "std_dev": 2.0})
        result = _call_indicator(spec, sample_ohlcv)
        assert isinstance(result, pd.Series)  # upper band
        valid = result.dropna()
        assert len(valid) > 0

    def test_invalid_indicator_raises(self, sample_ohlcv: pd.DataFrame) -> None:
        """존재하지 않는 지표명은 AttributeError."""
        spec = IndicatorSpec("nonexistent_indicator", {})
        with pytest.raises(AttributeError):
            _call_indicator(spec, sample_ohlcv)


# ---------------------------------------------------------------------------
# FeatureStoreConfig
# ---------------------------------------------------------------------------


class TestFeatureStoreConfig:
    """FeatureStoreConfig 테스트."""

    def test_default_specs(self) -> None:
        """기본 specs는 5개 지표."""
        config = FeatureStoreConfig()
        assert len(config.specs) == 8
        names = [s.name for s in config.specs]
        assert "atr" in names
        assert "rsi" in names
        assert "adx" in names

    def test_custom_specs(self) -> None:
        """커스텀 specs 설정."""
        config = FeatureStoreConfig(
            specs=(IndicatorSpec("rsi", {"period": 7}),),
            target_timeframe="4h",
        )
        assert len(config.specs) == 1
        assert config.target_timeframe == "4h"

    def test_frozen(self) -> None:
        """Frozen config는 변경 불가."""
        config = FeatureStoreConfig()
        with pytest.raises(Exception):  # noqa: B017
            config.target_timeframe = "1h"  # type: ignore[misc]

    def test_default_specs_matches_module_constant(self) -> None:
        """기본 specs는 모듈 상수 DEFAULT_SPECS와 동일."""
        config = FeatureStoreConfig()
        assert config.specs == DEFAULT_SPECS


# ---------------------------------------------------------------------------
# FeatureStore — Backtest
# ---------------------------------------------------------------------------


class TestFeatureStorePrecompute:
    """FeatureStore.precompute() 테스트."""

    def test_adds_columns(self, default_store: FeatureStore, sample_ohlcv: pd.DataFrame) -> None:
        """precompute 후 캐시에 지표 컬럼 존재."""
        default_store.precompute("BTC/USDT", sample_ohlcv)
        cached = default_store._cache["BTC/USDT"]
        expected_cols = {"atr_14", "rsi_14", "adx_14", "realized_volatility_30", "drawdown"}
        assert expected_cols.issubset(set(cached.columns))

    def test_index_matches(self, default_store: FeatureStore, sample_ohlcv: pd.DataFrame) -> None:
        """캐시 인덱스가 입력과 동일."""
        default_store.precompute("BTC/USDT", sample_ohlcv)
        cached = default_store._cache["BTC/USDT"]
        pd.testing.assert_index_equal(cached.index, sample_ohlcv.index)

    def test_cache_isolation(self, default_store: FeatureStore, sample_ohlcv: pd.DataFrame) -> None:
        """심볼별 캐시 독립."""
        default_store.precompute("BTC/USDT", sample_ohlcv)
        default_store.precompute("ETH/USDT", sample_ohlcv.iloc[:100])
        assert len(default_store._cache["BTC/USDT"]) == 200
        assert len(default_store._cache["ETH/USDT"]) == 100

    def test_custom_specs(self, custom_store: FeatureStore, sample_ohlcv: pd.DataFrame) -> None:
        """커스텀 specs로 지표 계산."""
        custom_store.precompute("BTC/USDT", sample_ohlcv)
        cached = custom_store._cache["BTC/USDT"]
        assert "rsi_7" in cached.columns
        assert "atr_20" in cached.columns
        assert "adx_14" not in cached.columns  # 기본 specs가 아님


class TestFeatureStoreEnrich:
    """FeatureStore.enrich_dataframe() 테스트."""

    def test_joins_cached(self, default_store: FeatureStore, sample_ohlcv: pd.DataFrame) -> None:
        """캐시된 컬럼을 DataFrame에 join."""
        default_store.precompute("BTC/USDT", sample_ohlcv)
        result = default_store.enrich_dataframe(sample_ohlcv, "BTC/USDT")
        assert "atr_14" in result.columns
        assert "rsi_14" in result.columns

    def test_no_cache_returns_original(
        self, default_store: FeatureStore, sample_ohlcv: pd.DataFrame
    ) -> None:
        """캐시 없으면 원본 DataFrame 반환."""
        result = default_store.enrich_dataframe(sample_ohlcv, "UNKNOWN")
        assert result is sample_ohlcv

    def test_no_overwrite(self, default_store: FeatureStore, sample_ohlcv: pd.DataFrame) -> None:
        """기존 컬럼은 덮어쓰지 않음."""
        default_store.precompute("BTC/USDT", sample_ohlcv)
        df = sample_ohlcv.copy()
        df["atr_14"] = 999.0
        result = default_store.enrich_dataframe(df, "BTC/USDT")
        assert (result["atr_14"] == 999.0).all()

    def test_partial_index(self, default_store: FeatureStore, sample_ohlcv: pd.DataFrame) -> None:
        """df 인덱스가 캐시의 부분집합이어도 정상 join."""
        default_store.precompute("BTC/USDT", sample_ohlcv)
        subset = sample_ohlcv.iloc[50:100]
        result = default_store.enrich_dataframe(subset, "BTC/USDT")
        assert len(result) == 50
        assert "atr_14" in result.columns

    def test_original_not_modified(
        self, default_store: FeatureStore, sample_ohlcv: pd.DataFrame
    ) -> None:
        """원본 DataFrame 미변경."""
        default_store.precompute("BTC/USDT", sample_ohlcv)
        original_cols = list(sample_ohlcv.columns)
        default_store.enrich_dataframe(sample_ohlcv, "BTC/USDT")
        assert list(sample_ohlcv.columns) == original_cols


# ---------------------------------------------------------------------------
# FeatureStore — Live (async)
# ---------------------------------------------------------------------------


class TestFeatureStoreLive:
    """FeatureStore live 모드 테스트."""

    @pytest.mark.asyncio
    async def test_register_subscribes_bar(self) -> None:
        """register() 후 BAR 이벤트 구독."""
        store = FeatureStore()
        bus = EventBus(queue_size=100)
        await store.register(bus)
        assert len(bus._handlers[EventType.BAR]) >= 1

    @pytest.mark.asyncio
    async def test_on_bar_updates_latest(self) -> None:
        """_on_bar가 _latest를 업데이트."""
        store = FeatureStore()
        bus = EventBus(queue_size=100)
        await store.register(bus)

        # 충분한 바 데이터 생성 (최소 30개 — realized_volatility window)
        np.random.seed(42)
        for i in range(50):
            bar = BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000.0 + i * 10,
                high=50500.0 + i * 10,
                low=49500.0 + i * 10,
                close=50200.0 + i * 10,
                volume=1000000.0,
                bar_timestamp=datetime(2024, 1, 1 + i, tzinfo=UTC)
                if i < 28
                else datetime(2024, 2, i - 27, tzinfo=UTC),
                source="test",
            )
            await store._on_bar(bar)

        latest = store.get_feature_columns("BTC/USDT")
        assert latest is not None
        assert "atr_14" in latest
        assert "rsi_14" in latest
        assert latest["atr_14"] > 0
        assert 0 <= latest["rsi_14"] <= 100

    @pytest.mark.asyncio
    async def test_on_bar_filters_timeframe(self) -> None:
        """다른 TF의 바는 무시."""
        store = FeatureStore(FeatureStoreConfig(target_timeframe="1D"))
        bar = BarEvent(
            symbol="BTC/USDT",
            timeframe="1m",  # TF 불일치
            open=50000.0,
            high=50500.0,
            low=49500.0,
            close=50200.0,
            volume=1000000.0,
            bar_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            source="test",
        )
        await store._on_bar(bar)
        assert store.get_feature_columns("BTC/USDT") is None

    @pytest.mark.asyncio
    async def test_on_bar_buffer_size_limit(self) -> None:
        """버퍼 크기가 max_buffer_size를 초과하지 않음."""
        config = FeatureStoreConfig(max_buffer_size=10, target_timeframe="1D")
        store = FeatureStore(config)

        for i in range(20):
            bar = BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000.0,
                high=50500.0,
                low=49500.0,
                close=50200.0,
                volume=1000000.0,
                bar_timestamp=datetime(2024, 1, 1 + i, tzinfo=UTC)
                if i < 28
                else datetime(2024, 2, i - 27, tzinfo=UTC),
                source="test",
            )
            await store._on_bar(bar)

        assert len(store._buffers["BTC/USDT"]) <= 10

    def test_get_feature_columns_empty(self) -> None:
        """데이터 없으면 None 반환."""
        store = FeatureStore()
        assert store.get_feature_columns("BTC/USDT") is None


class TestFeatureStoreWarmup:
    """FeatureStore.warmup() 테스트."""

    def test_initializes_cache(
        self, default_store: FeatureStore, sample_ohlcv: pd.DataFrame
    ) -> None:
        """warmup이 캐시를 초기화."""
        default_store.warmup("BTC/USDT", sample_ohlcv)
        assert "BTC/USDT" in default_store._cache

    def test_initializes_buffer(
        self, default_store: FeatureStore, sample_ohlcv: pd.DataFrame
    ) -> None:
        """warmup이 버퍼를 초기화."""
        default_store.warmup("BTC/USDT", sample_ohlcv)
        assert "BTC/USDT" in default_store._buffers
        assert len(default_store._buffers["BTC/USDT"]) == min(500, len(sample_ohlcv))

    def test_initializes_latest(
        self, default_store: FeatureStore, sample_ohlcv: pd.DataFrame
    ) -> None:
        """warmup이 latest 값을 설정."""
        default_store.warmup("BTC/USDT", sample_ohlcv)
        latest = default_store.get_feature_columns("BTC/USDT")
        assert latest is not None
        assert "atr_14" in latest

    def test_buffer_respects_max_size(self, sample_ohlcv: pd.DataFrame) -> None:
        """warmup 버퍼가 max_buffer_size를 초과하지 않음."""
        config = FeatureStoreConfig(max_buffer_size=50)
        store = FeatureStore(config)
        store.warmup("BTC/USDT", sample_ohlcv)
        assert len(store._buffers["BTC/USDT"]) == 50


# ---------------------------------------------------------------------------
# FeatureStorePort Protocol
# ---------------------------------------------------------------------------


class TestFeatureStorePort:
    """FeatureStorePort Protocol 만족 테스트."""

    def test_protocol_satisfied(self) -> None:
        """FeatureStore는 FeatureStorePort Protocol을 만족."""
        from src.eda.ports import FeatureStorePort

        store = FeatureStore()
        assert isinstance(store, FeatureStorePort)


# ---------------------------------------------------------------------------
# column_names property
# ---------------------------------------------------------------------------


class TestColumnNames:
    """column_names 프로퍼티 테스트."""

    def test_default_names(self) -> None:
        """기본 specs의 컬럼명."""
        store = FeatureStore()
        names = store.column_names
        assert "atr_14" in names
        assert "rsi_14" in names
        assert "adx_14" in names
        assert "realized_volatility_30" in names
        assert "drawdown" in names

    def test_custom_names(self) -> None:
        """커스텀 column_name 반영."""
        config = FeatureStoreConfig(
            specs=(
                IndicatorSpec("rsi", {"period": 7}, column_name="my_rsi"),
                IndicatorSpec("atr", {"period": 20}),
            ),
        )
        store = FeatureStore(config)
        assert store.column_names == ["my_rsi", "atr_20"]


# ---------------------------------------------------------------------------
# register_specs
# ---------------------------------------------------------------------------


class TestRegisterSpecs:
    """register_specs() 동적 확장 테스트."""

    def test_register_specs_adds_new(self) -> None:
        """새 spec이 정상 등록되는지 확인."""
        store = FeatureStore(FeatureStoreConfig(specs=(IndicatorSpec("rsi", {"period": 14}),)))
        assert len(store.column_names) == 1

        store.register_specs([IndicatorSpec("atr", {"period": 14})])
        assert len(store.column_names) == 2
        assert "atr_14" in store.column_names

    def test_register_specs_dedup(self) -> None:
        """중복 name 등록 시 무시."""
        store = FeatureStore(FeatureStoreConfig(specs=(IndicatorSpec("rsi", {"period": 14}),)))
        store.register_specs(
            [
                IndicatorSpec("rsi", {"period": 7}),  # 같은 name → 무시
                IndicatorSpec("atr", {"period": 14}),
            ]
        )
        assert len(store.column_names) == 2
        # 기존 rsi_14 유지 (rsi_7 아님)
        assert "rsi_14" in store.column_names

    def test_register_specs_empty(self) -> None:
        """빈 리스트 등록 시 변경 없음."""
        store = FeatureStore()
        original_count = len(store.column_names)
        store.register_specs([])
        assert len(store.column_names) == original_count

    def test_precompute_with_extra_specs(self, sample_ohlcv: pd.DataFrame) -> None:
        """추가 spec이 precompute 캐시에 포함되는지 확인."""
        store = FeatureStore(FeatureStoreConfig(specs=(IndicatorSpec("rsi", {"period": 14}),)))
        store.register_specs([IndicatorSpec("momentum", {"period": 10})])

        store.precompute("BTC/USDT", sample_ohlcv)
        enriched = store.enrich_dataframe(sample_ohlcv.copy(), "BTC/USDT")

        assert "rsi_14" in enriched.columns
        assert "momentum_10" in enriched.columns
