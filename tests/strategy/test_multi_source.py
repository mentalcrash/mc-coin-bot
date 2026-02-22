"""Tests for src/strategy/multi_source/ — Multi-Source 전략 템플릿."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.multi_source.config import (
    MultiSourceConfig,
    ShortMode,
    SignalCombineMethod,
    SubSignalSpec,
    SubSignalTransform,
)
from src.strategy.multi_source.preprocessor import preprocess
from src.strategy.multi_source.signal import generate_signals
from src.strategy.multi_source.strategy import MultiSourceStrategy

# ─── Test Fixtures ───────────────────────────────────────────────


def _make_enriched_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """테스트용 enriched DataFrame 생성."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))

    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.5, n),
            "high": close + abs(rng.normal(0, 1, n)),
            "low": close - abs(rng.normal(0, 1, n)),
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
            "oc_fear_greed": rng.uniform(10, 90, n),
            "oc_mvrv": rng.normal(2.0, 0.5, n),
            "macro_dxy": rng.normal(100, 5, n),
        },
        index=dates,
    )


def _default_config() -> MultiSourceConfig:
    """2-signal 기본 설정."""
    return MultiSourceConfig(
        signals=(
            SubSignalSpec(column="oc_fear_greed", transform=SubSignalTransform.ZSCORE, window=20),
            SubSignalSpec(column="oc_mvrv", transform=SubSignalTransform.ZSCORE, window=20),
        ),
        entry_threshold=1.0,
        exit_threshold=0.3,
    )


# ─── Config Validation ──────────────────────────────────────────


class TestMultiSourceConfig:
    def test_valid_config(self) -> None:
        config = _default_config()
        assert len(config.signals) == 2
        assert config.combine_method == SignalCombineMethod.ZSCORE_SUM

    def test_min_signals_validation(self) -> None:
        """최소 2개 서브시그널 필요."""
        with pytest.raises(ValueError, match="least 2"):
            MultiSourceConfig(
                signals=(SubSignalSpec(column="oc_fear_greed"),),
            )

    def test_max_signals_validation(self) -> None:
        """최대 5개 서브시그널."""
        with pytest.raises(ValueError, match="most 5"):
            MultiSourceConfig(
                signals=tuple(SubSignalSpec(column=f"col_{i}") for i in range(6)),
            )

    def test_exit_must_be_less_than_entry(self) -> None:
        with pytest.raises(ValueError, match="exit_threshold"):
            MultiSourceConfig(
                signals=(
                    SubSignalSpec(column="a"),
                    SubSignalSpec(column="b"),
                ),
                entry_threshold=1.0,
                exit_threshold=1.5,
            )

    def test_frozen_config(self) -> None:
        config = _default_config()
        with pytest.raises(Exception):  # noqa: B017
            config.entry_threshold = 2.0  # type: ignore[misc]

    def test_warmup_periods(self) -> None:
        config = MultiSourceConfig(
            signals=(
                SubSignalSpec(column="a", window=50),
                SubSignalSpec(column="b", window=20),
            ),
        )
        assert config.warmup_periods() >= 50

    def test_short_mode_default(self) -> None:
        config = _default_config()
        assert config.short_mode == ShortMode.HEDGE_ONLY


class TestSubSignalSpec:
    def test_defaults(self) -> None:
        spec = SubSignalSpec(column="oc_mvrv")
        assert spec.transform == SubSignalTransform.ZSCORE
        assert spec.window == 30
        assert spec.weight == 1.0
        assert spec.invert is False

    def test_contrarian(self) -> None:
        spec = SubSignalSpec(column="oc_fear_greed", invert=True)
        assert spec.invert is True

    def test_frozen(self) -> None:
        spec = SubSignalSpec(column="test")
        with pytest.raises(Exception):  # noqa: B017
            spec.column = "other"  # type: ignore[misc]


# ─── Preprocessor ────────────────────────────────────────────────


class TestPreprocess:
    def test_output_has_sub_columns(self) -> None:
        df = _make_enriched_df()
        config = _default_config()
        result = preprocess(df, config)
        assert "_sub_0" in result.columns
        assert "_sub_1" in result.columns

    def test_vol_scalar_computed(self) -> None:
        df = _make_enriched_df()
        config = _default_config()
        result = preprocess(df, config)
        assert "_vol_scalar" in result.columns
        assert not result["_vol_scalar"].isna().all()

    def test_does_not_modify_original(self) -> None:
        df = _make_enriched_df()
        original_cols = set(df.columns)
        config = _default_config()
        preprocess(df, config)
        assert set(df.columns) == original_cols

    def test_missing_column_graceful(self) -> None:
        """서브시그널 컬럼이 없으면 NaN으로 채움."""
        df = _make_enriched_df()
        config = MultiSourceConfig(
            signals=(
                SubSignalSpec(column="oc_fear_greed"),
                SubSignalSpec(column="nonexistent_col"),
            ),
        )
        result = preprocess(df, config)
        assert result["_sub_1"].isna().all()

    def test_invert_flips_sign(self) -> None:
        df = _make_enriched_df()
        config_normal = MultiSourceConfig(
            signals=(
                SubSignalSpec(column="oc_fear_greed", window=20),
                SubSignalSpec(column="oc_mvrv", window=20),
            ),
        )
        config_inverted = MultiSourceConfig(
            signals=(
                SubSignalSpec(column="oc_fear_greed", window=20, invert=True),
                SubSignalSpec(column="oc_mvrv", window=20),
            ),
        )
        normal = preprocess(df, config_normal)
        inverted = preprocess(df, config_inverted)

        # inverted sub_0 should be opposite sign (where not NaN)
        valid_mask = normal["_sub_0"].notna() & inverted["_sub_0"].notna()
        if valid_mask.any():
            np.testing.assert_array_almost_equal(
                normal["_sub_0"][valid_mask].values,
                -inverted["_sub_0"][valid_mask].values,
            )

    def test_missing_close_raises(self) -> None:
        df = pd.DataFrame({"volume": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3))
        config = _default_config()
        with pytest.raises(ValueError, match="Missing"):
            preprocess(df, config)

    def test_percentile_transform(self) -> None:
        df = _make_enriched_df()
        config = MultiSourceConfig(
            signals=(
                SubSignalSpec(
                    column="oc_fear_greed", transform=SubSignalTransform.PERCENTILE, window=20
                ),
                SubSignalSpec(column="oc_mvrv", window=20),
            ),
        )
        result = preprocess(df, config)
        valid = result["_sub_0"].dropna()
        assert len(valid) > 0
        # Percentile는 -1 ~ 1 범위
        assert valid.min() >= -1.01
        assert valid.max() <= 1.01

    def test_ma_cross_transform(self) -> None:
        df = _make_enriched_df()
        config = MultiSourceConfig(
            signals=(
                SubSignalSpec(
                    column="oc_fear_greed", transform=SubSignalTransform.MA_CROSS, window=20
                ),
                SubSignalSpec(column="oc_mvrv", window=20),
            ),
        )
        result = preprocess(df, config)
        valid = result["_sub_0"].dropna()
        assert len(valid) > 0


# ─── Signal Generation ───────────────────────────────────────────


class TestGenerateSignals:
    def test_returns_strategy_signals(self) -> None:
        df = _make_enriched_df()
        config = _default_config()
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_direction_values(self) -> None:
        """direction은 -1, 0, 1만 포함."""
        df = _make_enriched_df()
        config = _default_config()
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_shift_1_no_lookahead(self) -> None:
        """첫 번째 bar는 시그널 없음 (shift(1) 때문)."""
        df = _make_enriched_df()
        config = _default_config()
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # 첫 bar는 shift(1) → NaN → direction 0
        assert signals.direction.iloc[0] == 0

    def test_disabled_short_mode(self) -> None:
        """DISABLED 모드에서 short 시그널 없음."""
        df = _make_enriched_df()
        config = MultiSourceConfig(
            signals=(
                SubSignalSpec(column="oc_fear_greed", window=20),
                SubSignalSpec(column="oc_mvrv", window=20),
            ),
            entry_threshold=0.5,
            exit_threshold=0.1,
            short_mode=ShortMode.DISABLED,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction >= 0).all()

    def test_majority_vote_method(self) -> None:
        df = _make_enriched_df()
        config = MultiSourceConfig(
            signals=(
                SubSignalSpec(column="oc_fear_greed", window=20),
                SubSignalSpec(column="oc_mvrv", window=20),
            ),
            combine_method=SignalCombineMethod.MAJORITY_VOTE,
            entry_threshold=0.5,
            exit_threshold=0.1,
            min_agreement=0.6,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_weighted_sum_method(self) -> None:
        df = _make_enriched_df()
        config = MultiSourceConfig(
            signals=(
                SubSignalSpec(column="oc_fear_greed", window=20, weight=2.0),
                SubSignalSpec(column="oc_mvrv", window=20, weight=1.0),
            ),
            combine_method=SignalCombineMethod.WEIGHTED_SUM,
            entry_threshold=0.5,
            exit_threshold=0.1,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        assert len(signals.direction) == len(df)


# ─── Strategy Class ──────────────────────────────────────────────


class TestMultiSourceStrategy:
    def test_name(self) -> None:
        strategy = MultiSourceStrategy()
        assert strategy.name == "multi-source"

    def test_required_columns(self) -> None:
        config = _default_config()
        strategy = MultiSourceStrategy(config=config)
        cols = strategy.required_columns
        assert "close" in cols
        assert "oc_fear_greed" in cols
        assert "oc_mvrv" in cols

    def test_from_params(self) -> None:
        params = {
            "signals": (
                SubSignalSpec(column="oc_fear_greed"),
                SubSignalSpec(column="oc_mvrv"),
            ),
        }
        strategy = MultiSourceStrategy.from_params(**params)
        assert strategy.name == "multi-source"

    def test_run(self) -> None:
        """전체 파이프라인 (preprocess + generate_signals)."""
        df = _make_enriched_df()
        config = _default_config()
        strategy = MultiSourceStrategy(config=config)
        processed, signals = strategy.run(df)
        assert "_sub_0" in processed.columns
        assert len(signals.direction) == len(df)

    def test_registry_registered(self) -> None:
        from src.strategy.registry import is_registered

        assert is_registered("multi-source")

    def test_get_startup_info(self) -> None:
        strategy = MultiSourceStrategy(config=_default_config())
        info = strategy.get_startup_info()
        assert "combine_method" in info
        assert "n_signals" in info
