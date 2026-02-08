"""Tests for ADX Regime Filter Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.adx_regime.config import ADXRegimeConfig
from src.strategy.adx_regime.preprocessor import preprocess


class TestPreprocess:
    """전처리 메인 함수 테스트."""

    def test_preprocess_adds_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 필수 컬럼이 모두 존재."""
        config = ADXRegimeConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = [
            "adx",
            "returns",
            "realized_vol",
            "vol_scalar",
            "vw_momentum",
            "z_score",
            "drawdown",
            "atr",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_z_score_mean_zero(self, sample_ohlcv: pd.DataFrame) -> None:
        """z_score는 warmup 이후 대략 평균 0."""
        config = ADXRegimeConfig()
        result = preprocess(sample_ohlcv, config)

        warmup = config.warmup_periods()
        z_score_after_warmup = result["z_score"].iloc[warmup:].dropna()

        # z_score는 (close - SMA) / std이므로 완전히 0은 아니지만 대략 0 근처
        assert len(z_score_after_warmup) > 0
        assert abs(z_score_after_warmup.mean()) < 1.0

    def test_adx_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """ADX는 warmup 이후 >= 0."""
        config = ADXRegimeConfig()
        result = preprocess(sample_ohlcv, config)

        warmup = config.warmup_periods()
        adx_after_warmup = result["adx"].iloc[warmup:].dropna()

        assert len(adx_after_warmup) > 0
        assert (adx_after_warmup >= 0).all()

    def test_missing_columns_raises(self) -> None:
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        config = ADXRegimeConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_original_not_modified(self, sample_ohlcv: pd.DataFrame) -> None:
        """원본 DataFrame이 수정되지 않음."""
        config = ADXRegimeConfig()
        original_cols = list(sample_ohlcv.columns)
        preprocess(sample_ohlcv, config)
        assert list(sample_ohlcv.columns) == original_cols

    def test_vol_scalar_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """vol_scalar는 warmup 이후 항상 양수."""
        config = ADXRegimeConfig()
        result = preprocess(sample_ohlcv, config)

        warmup = config.warmup_periods()
        vol_scalar = result["vol_scalar"].iloc[warmup:].dropna()
        assert (vol_scalar > 0).all()

    def test_returns_has_values(self, sample_ohlcv: pd.DataFrame) -> None:
        """returns 컬럼이 계산됨."""
        config = ADXRegimeConfig()
        result = preprocess(sample_ohlcv, config)

        # 첫 번째 값은 NaN (로그 수익률), 이후 값 존재
        returns = result["returns"].dropna()
        assert len(returns) > 0
        assert np.isfinite(returns).all()

    def test_atr_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """ATR은 warmup 이후 양수."""
        config = ADXRegimeConfig()
        result = preprocess(sample_ohlcv, config)

        warmup = config.warmup_periods()
        atr = result["atr"].iloc[warmup:].dropna()
        assert len(atr) > 0
        assert (atr > 0).all()

    def test_drawdown_non_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """drawdown은 항상 <= 0."""
        config = ADXRegimeConfig()
        result = preprocess(sample_ohlcv, config)

        drawdown = result["drawdown"].dropna()
        assert len(drawdown) > 0
        assert (drawdown <= 0).all()
