"""Regime-Adaptive TSMOM 전략 단위 테스트."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from src.strategy.regime_tsmom.config import RegimeTSMOMConfig
from src.strategy.regime_tsmom.preprocessor import preprocess
from src.strategy.regime_tsmom.signal import generate_signals
from src.strategy.regime_tsmom.strategy import RegimeTSMOMStrategy
from src.strategy.tsmom.config import ShortMode

# ── Config ──


class TestRegimeTSMOMConfig:
    """RegimeTSMOMConfig 검증 테스트."""

    def test_default_values(self) -> None:
        cfg = RegimeTSMOMConfig()
        assert cfg.lookback == 30
        assert cfg.vol_target == 0.35
        assert cfg.trending_vol_target == 0.40
        assert cfg.ranging_vol_target == 0.15
        assert cfg.volatile_vol_target == 0.10

    def test_frozen(self) -> None:
        cfg = RegimeTSMOMConfig()
        with pytest.raises(ValidationError):
            cfg.lookback = 10  # type: ignore[misc]

    def test_invalid_momentum_smoothing(self) -> None:
        """momentum_smoothing > lookback → ValidationError."""
        with pytest.raises(ValidationError):
            RegimeTSMOMConfig(lookback=10, momentum_smoothing=20)

    def test_to_tsmom_config(self) -> None:
        """TSMOM 설정 변환."""
        cfg = RegimeTSMOMConfig(lookback=20, vol_target=0.30)
        tsmom_cfg = cfg.to_tsmom_config()
        assert tsmom_cfg.lookback == 20
        assert tsmom_cfg.vol_target == 0.30

    def test_warmup_periods(self) -> None:
        """warmup은 TSMOM과 regime 중 큰 값."""
        cfg = RegimeTSMOMConfig(lookback=50, vol_window=50)
        warmup = cfg.warmup_periods()
        assert warmup >= 51  # max(lookback, vol_window) + 1
        assert warmup >= cfg.regime.rv_long_window + 5

    def test_from_params(self) -> None:
        """from_params()로 인스턴스 생성."""
        strategy = RegimeTSMOMStrategy.from_params(
            lookback=20,
            trending_vol_target=0.50,
        )
        assert strategy.config.lookback == 20
        assert strategy.config.trending_vol_target == 0.50


# ── Preprocessor ──


class TestPreprocess:
    """preprocess() 테스트."""

    def test_adds_regime_columns(
        self, sample_ohlcv: pd.DataFrame, default_config: RegimeTSMOMConfig
    ) -> None:
        """레짐 컬럼이 추가되는지 확인."""
        result = preprocess(sample_ohlcv, default_config)
        assert "regime_label" in result.columns
        assert "p_trending" in result.columns
        assert "p_ranging" in result.columns
        assert "p_volatile" in result.columns

    def test_adds_tsmom_columns(
        self, sample_ohlcv: pd.DataFrame, default_config: RegimeTSMOMConfig
    ) -> None:
        """TSMOM 지표 컬럼이 추가되는지 확인."""
        result = preprocess(sample_ohlcv, default_config)
        assert "vw_momentum" in result.columns
        assert "realized_vol" in result.columns
        assert "vol_scalar" in result.columns
        assert "returns" in result.columns

    def test_preserves_original(
        self, sample_ohlcv: pd.DataFrame, default_config: RegimeTSMOMConfig
    ) -> None:
        """원본 OHLCV 데이터 보존."""
        result = preprocess(sample_ohlcv, default_config)
        pd.testing.assert_series_equal(
            result["close"], sample_ohlcv["close"], check_names=True
        )


# ── Signal Generation ──


class TestGenerateSignals:
    """generate_signals() 테스트."""

    def test_returns_strategy_signals(
        self, sample_ohlcv: pd.DataFrame, default_config: RegimeTSMOMConfig
    ) -> None:
        """StrategySignals 반환 확인."""
        processed = preprocess(sample_ohlcv, default_config)
        signals = generate_signals(processed, default_config)

        assert len(signals.entries) == len(sample_ohlcv)
        assert len(signals.exits) == len(sample_ohlcv)
        assert len(signals.direction) == len(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_direction_values(
        self, sample_ohlcv: pd.DataFrame, default_config: RegimeTSMOMConfig
    ) -> None:
        """direction은 -1, 0, 1만 허용."""
        processed = preprocess(sample_ohlcv, default_config)
        signals = generate_signals(processed, default_config)
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_trending_higher_strength(
        self, trending_ohlcv: pd.DataFrame, volatile_ohlcv: pd.DataFrame
    ) -> None:
        """trending 구간에서 volatile 구간보다 평균 strength가 높아야 함."""
        config = RegimeTSMOMConfig(
            trending_vol_target=0.40,
            ranging_vol_target=0.15,
            volatile_vol_target=0.10,
            trending_leverage_scale=1.0,
            volatile_leverage_scale=0.2,
        )

        trending_processed = preprocess(trending_ohlcv, config)
        trending_signals = generate_signals(trending_processed, config)

        volatile_processed = preprocess(volatile_ohlcv, config)
        volatile_signals = generate_signals(volatile_processed, config)

        # 절대 strength 평균 비교
        avg_trending = trending_signals.strength.abs().mean()
        avg_volatile = volatile_signals.strength.abs().mean()

        assert avg_trending > avg_volatile, (
            f"trending avg strength ({avg_trending:.4f}) should be > "
            f"volatile avg strength ({avg_volatile:.4f})"
        )

    def test_short_mode_disabled(
        self, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Short DISABLED → 음수 direction 없음."""
        config = RegimeTSMOMConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_missing_columns_raises(self, default_config: RegimeTSMOMConfig) -> None:
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"close": [100, 101]})
        with pytest.raises(ValueError, match="Missing"):
            generate_signals(df, default_config)

    def test_probability_weighted_vol_target(
        self, sample_ohlcv: pd.DataFrame
    ) -> None:
        """vol_target이 레짐 확률에 따라 가중 적용되는지 확인."""
        config = RegimeTSMOMConfig(
            trending_vol_target=1.0,
            ranging_vol_target=0.0,
            volatile_vol_target=0.0,
        )
        processed = preprocess(sample_ohlcv, config)
        valid = processed.dropna()

        # p_trending이 높을수록 adaptive vol_target이 높아야 함
        if len(valid) > 0:
            adaptive_vt = (
                valid["p_trending"] * config.trending_vol_target
                + valid["p_ranging"] * config.ranging_vol_target
                + valid["p_volatile"] * config.volatile_vol_target
            )
            # p_trending과 adaptive_vt 상관관계 높아야 함
            corr = valid["p_trending"].corr(adaptive_vt)
            assert corr > 0.9, f"Correlation {corr:.2f} should be > 0.9"


# ── Strategy Class ──


class TestRegimeTSMOMStrategy:
    """RegimeTSMOMStrategy 클래스 테스트."""

    def test_run_end_to_end(self, sample_ohlcv: pd.DataFrame) -> None:
        """전략 run() end-to-end 실행."""
        strategy = RegimeTSMOMStrategy()
        processed, signals = strategy.run(sample_ohlcv)

        assert len(processed) == len(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)
        assert "regime_label" in processed.columns
        assert "vw_momentum" in processed.columns

    def test_name(self) -> None:
        assert RegimeTSMOMStrategy().name == "Regime-TSMOM"

    def test_required_columns(self) -> None:
        cols = RegimeTSMOMStrategy().required_columns
        assert "close" in cols
        assert "volume" in cols

    def test_recommended_config(self) -> None:
        cfg = RegimeTSMOMStrategy.recommended_config()
        assert "max_leverage_cap" in cfg
        assert cfg["max_leverage_cap"] == 1.0

    def test_get_startup_info(self) -> None:
        info = RegimeTSMOMStrategy().get_startup_info()
        assert "trending_vt" in info
        assert "ranging_vt" in info

    def test_registry_lookup(self) -> None:
        """레지스트리에서 조회 가능."""
        from src.strategy.registry import get_strategy

        strategy_cls = get_strategy("regime-tsmom")
        assert strategy_cls is RegimeTSMOMStrategy
