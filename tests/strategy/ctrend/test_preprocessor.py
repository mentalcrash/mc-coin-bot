"""Tests for CTREND preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.market.indicators import (
    bb_position,
    cci,
    chaikin_money_flow,
    macd,
    obv,
    roc,
    rsi,
    stochastic,
    volume_macd,
    williams_r,
)
from src.strategy.ctrend.config import CTRENDConfig
from src.strategy.ctrend.preprocessor import (
    compute_all_features,
    preprocess,
)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (400일, training_window=252 고려)."""
    np.random.seed(42)
    n = 400

    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5) + 0.5
    low = close - np.abs(np.random.randn(n) * 1.5) - 0.5
    open_ = close + np.random.randn(n) * 0.5

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def default_config() -> CTRENDConfig:
    """기본 CTRENDConfig."""
    return CTRENDConfig()


class TestPreprocess:
    """preprocess 함수 테스트."""

    def test_preprocess_columns(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: CTRENDConfig,
    ) -> None:
        """preprocess 출력에 필수 컬럼 존재."""
        result = preprocess(sample_ohlcv_df, default_config)

        # 기본 컬럼
        assert "returns" in result.columns
        assert "realized_vol" in result.columns
        assert "vol_scalar" in result.columns
        assert "forward_return" in result.columns

        # feature 컬럼 (28개)
        feat_cols = [c for c in result.columns if c.startswith("feat_")]
        assert len(feat_cols) == 28

    def test_forward_return_calculation(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: CTRENDConfig,
    ) -> None:
        """forward_return이 올바르게 계산되었는지 확인."""
        result = preprocess(sample_ohlcv_df, default_config)

        # forward_return의 마지막 prediction_horizon 개는 NaN
        fwd = result["forward_return"]
        assert fwd.iloc[-default_config.prediction_horizon :].isna().all()

        # 앞 부분은 NaN이 아닌 값이 존재
        assert (
            fwd.iloc[default_config.prediction_horizon : -default_config.prediction_horizon]
            .notna()
            .any()
        )

    def test_numeric_conversion(
        self,
        default_config: CTRENDConfig,
    ) -> None:
        """Decimal 타입 등을 float로 변환."""
        n = 400
        np.random.seed(42)

        close = 100 + np.cumsum(np.random.randn(n) * 2)
        df = pd.DataFrame(
            {
                "open": close + 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close.astype(str),  # 문자열로 저장
                "volume": np.full(n, "5000"),  # 문자열
            },
            index=pd.date_range("2023-01-01", periods=n, freq="D"),
        )

        result = preprocess(df, default_config)
        assert result["close"].dtype == np.float64

    def test_missing_columns_raises(
        self,
        default_config: CTRENDConfig,
    ) -> None:
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, default_config)

    def test_empty_df_raises(
        self,
        default_config: CTRENDConfig,
    ) -> None:
        """빈 DataFrame 시 에러."""
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
        )

        with pytest.raises(ValueError, match="empty"):
            preprocess(df, default_config)

    def test_original_not_modified(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: CTRENDConfig,
    ) -> None:
        """원본 DataFrame이 수정되지 않음."""
        original_cols = list(sample_ohlcv_df.columns)
        original_values = sample_ohlcv_df["close"].values.copy()

        _ = preprocess(sample_ohlcv_df, default_config)

        assert list(sample_ohlcv_df.columns) == original_cols
        np.testing.assert_array_equal(sample_ohlcv_df["close"].values, original_values)

    def test_feature_names(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: CTRENDConfig,
    ) -> None:
        """feature 컬럼 이름이 feat_ 접두사로 시작."""
        result = preprocess(sample_ohlcv_df, default_config)

        feat_cols = [c for c in result.columns if c.startswith("feat_")]
        for col in feat_cols:
            assert col.startswith("feat_")

        # 주요 feature 이름 확인
        expected_features = [
            "feat_macd",
            "feat_macd_signal",
            "feat_macd_hist",
            "feat_rsi_14",
            "feat_rsi_7",
            "feat_cci_20",
            "feat_stoch_k",
            "feat_stoch_d",
            "feat_obv_norm",
            "feat_cmf_20",
            "feat_bb_pos_20",
            "feat_volume_macd",
            "feat_sma_cross_5_20",
            "feat_roc_5",
            "feat_atr_ratio_14",
            "feat_vol_ratio",
        ]
        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"


class TestComputeAllFeatures:
    """compute_all_features 함수 테스트."""

    def test_shape_28_features(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """28개 feature 컬럼 확인."""
        features = compute_all_features(sample_ohlcv_df)
        assert features.shape[1] == 28
        assert features.shape[0] == len(sample_ohlcv_df)


class TestIndividualFeatures:
    """개별 feature 계산 함수 테스트."""

    @pytest.fixture
    def close_series(self) -> pd.Series:
        """샘플 종가 시리즈."""
        np.random.seed(42)
        n = 200
        return pd.Series(
            100 + np.cumsum(np.random.randn(n) * 2),
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

    @pytest.fixture
    def ohlcv_series(self) -> dict[str, pd.Series]:
        """샘플 OHLCV 시리즈 딕셔너리."""
        np.random.seed(42)
        n = 200
        close = pd.Series(
            100 + np.cumsum(np.random.randn(n) * 2),
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        high = close + np.abs(np.random.randn(n) * 1.5) + 0.5
        low = close - np.abs(np.random.randn(n) * 1.5) - 0.5
        volume = pd.Series(
            np.random.randint(1000, 10000, n).astype(float),
            index=close.index,
        )
        return {"close": close, "high": high, "low": low, "volume": volume}

    def test_macd(self, close_series: pd.Series) -> None:
        """MACD 계산 테스트."""
        macd_line, signal_line, histogram = macd(close_series)

        assert len(macd_line) == len(close_series)
        assert len(signal_line) == len(close_series)
        assert len(histogram) == len(close_series)

        # histogram = macd_line - signal_line
        valid = macd_line.dropna().index.intersection(signal_line.dropna().index)
        np.testing.assert_allclose(
            histogram.loc[valid].values,
            (macd_line.loc[valid] - signal_line.loc[valid]).values,
            atol=1e-10,
        )

    def test_rsi(self, close_series: pd.Series) -> None:
        """RSI 계산 테스트 (0-100 범위)."""
        rsi_result = rsi(close_series, period=14)

        valid_rsi = rsi_result.dropna()
        assert len(valid_rsi) > 0
        assert valid_rsi.min() >= 0.0
        assert valid_rsi.max() <= 100.0

    def test_cci(self, ohlcv_series: dict[str, pd.Series]) -> None:
        """CCI 계산 테스트."""
        cci_result = cci(
            ohlcv_series["high"],
            ohlcv_series["low"],
            ohlcv_series["close"],
            period=20,
        )

        valid_cci = cci_result.dropna()
        assert len(valid_cci) > 0
        # CCI는 unbounded지만 대부분 -200 ~ +200 범위
        assert valid_cci.min() < 0
        assert valid_cci.max() > 0

    def test_williams_r(self, ohlcv_series: dict[str, pd.Series]) -> None:
        """Williams %R 계산 테스트 (-100 ~ 0)."""
        wr = williams_r(
            ohlcv_series["high"],
            ohlcv_series["low"],
            ohlcv_series["close"],
            period=14,
        )

        valid_wr = wr.dropna()
        assert len(valid_wr) > 0
        assert valid_wr.min() >= -100.0
        assert valid_wr.max() <= 0.0

    def test_stochastic(self, ohlcv_series: dict[str, pd.Series]) -> None:
        """Stochastic 계산 테스트 (0-100)."""
        k, _d = stochastic(
            ohlcv_series["high"],
            ohlcv_series["low"],
            ohlcv_series["close"],
        )

        valid_k = k.dropna()
        assert len(valid_k) > 0
        assert valid_k.min() >= 0.0
        assert valid_k.max() <= 100.0

    def test_obv(self, close_series: pd.Series) -> None:
        """OBV 계산 테스트."""
        np.random.seed(42)
        volume = pd.Series(
            np.random.randint(1000, 10000, len(close_series)).astype(float),
            index=close_series.index,
        )
        obv_result = obv(close_series, volume)

        assert len(obv_result) == len(close_series)
        # OBV는 cumulative이므로 단조 증가/감소는 아님

    def test_bb_position(self, close_series: pd.Series) -> None:
        """Bollinger Band position 테스트."""
        bb_pos = bb_position(close_series, period=20)

        valid_bb = bb_pos.dropna()
        assert len(valid_bb) > 0
        # 대부분 0~1 범위이지만 극단 상황에서 벗어날 수 있음

    def test_roc(self, close_series: pd.Series) -> None:
        """Rate of Change 테스트."""
        roc_result = roc(close_series, period=5)

        valid_roc = roc_result.dropna()
        assert len(valid_roc) > 0
        # ROC는 pct_change이므로 unbounded

    def test_volume_macd(self) -> None:
        """Volume MACD 테스트."""
        np.random.seed(42)
        n = 200
        volume = pd.Series(
            np.random.randint(1000, 10000, n).astype(float),
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        v_macd = volume_macd(volume)

        assert len(v_macd) == n
        # Volume MACD는 EMA 차이이므로 양수/음수 모두 가능

    def test_chaikin_money_flow(self, ohlcv_series: dict[str, pd.Series]) -> None:
        """Chaikin Money Flow 테스트."""
        cmf = chaikin_money_flow(
            ohlcv_series["high"],
            ohlcv_series["low"],
            ohlcv_series["close"],
            ohlcv_series["volume"],
            period=20,
        )

        valid_cmf = cmf.dropna()
        assert len(valid_cmf) > 0
        # CMF는 -1 ~ +1 범위
        assert valid_cmf.min() >= -1.0
        assert valid_cmf.max() <= 1.0
