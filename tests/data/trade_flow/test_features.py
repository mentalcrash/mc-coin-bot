"""Tests for trade flow feature computation (compute_bar_features + compute_vpin)."""

import numpy as np
import pandas as pd
import pytest

from src.data.trade_flow.features import compute_bar_features, compute_vpin

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_trades() -> pd.DataFrame:
    """기본 aggTrades DataFrame — 60% buy, 40% sell."""
    return pd.DataFrame(
        {
            "quantity": [1.0, 2.0, 3.0, 1.5, 2.5],
            "is_buyer_maker": [False, False, False, True, True],
            # buy: 1+2+3 = 6.0, sell: 1.5+2.5 = 4.0, total = 10.0
        }
    )


@pytest.fixture
def all_buy_trades() -> pd.DataFrame:
    """모두 taker buy인 trades."""
    return pd.DataFrame(
        {
            "quantity": [1.0, 2.0, 3.0],
            "is_buyer_maker": [False, False, False],
        }
    )


@pytest.fixture
def all_sell_trades() -> pd.DataFrame:
    """모두 taker sell인 trades."""
    return pd.DataFrame(
        {
            "quantity": [1.0, 2.0, 3.0],
            "is_buyer_maker": [True, True, True],
        }
    )


@pytest.fixture
def large_trades() -> pd.DataFrame:
    """대형 거래를 포함한 trades (30건 이상)."""
    rng = np.random.default_rng(42)
    n = 100
    quantities = rng.exponential(scale=1.0, size=n)
    # 상위 5%에 대형 거래 추가
    quantities[-5:] = quantities[-5:] * 50
    return pd.DataFrame(
        {
            "quantity": quantities,
            "is_buyer_maker": rng.choice([True, False], size=n),
        }
    )


# ---------------------------------------------------------------------------
# compute_bar_features tests
# ---------------------------------------------------------------------------


class TestComputeBarFeatures:
    def test_empty_dataframe(self) -> None:
        """빈 DataFrame → 모든 피처 0.0."""
        df = pd.DataFrame(columns=["quantity", "is_buyer_maker"])
        result = compute_bar_features(df)

        assert result["tflow_cvd"] == 0.0
        assert result["tflow_buy_ratio"] == 0.0
        assert result["tflow_intensity"] == 0.0
        assert result["tflow_large_ratio"] == 0.0
        assert result["tflow_abs_order_imbalance"] == 0.0

    def test_cvd_calculation(self, sample_trades: pd.DataFrame) -> None:
        """CVD = (buy_vol - sell_vol) / total_vol."""
        result = compute_bar_features(sample_trades)
        # buy=6.0, sell=4.0, total=10.0 → CVD = (6-4)/10 = 0.2
        assert result["tflow_cvd"] == pytest.approx(0.2)

    def test_buy_ratio(self, sample_trades: pd.DataFrame) -> None:
        """buy_ratio = buy_vol / total_vol."""
        result = compute_bar_features(sample_trades)
        # buy=6.0, total=10.0 → 0.6
        assert result["tflow_buy_ratio"] == pytest.approx(0.6)

    def test_intensity(self, sample_trades: pd.DataFrame) -> None:
        """intensity = trade_count / bar_hours."""
        result = compute_bar_features(sample_trades, bar_hours=12.0)
        # 5 trades / 12 hours
        assert result["tflow_intensity"] == pytest.approx(5 / 12)

    def test_all_buy_cvd_is_one(self, all_buy_trades: pd.DataFrame) -> None:
        """모두 buy → CVD = 1.0."""
        result = compute_bar_features(all_buy_trades)
        assert result["tflow_cvd"] == pytest.approx(1.0)
        assert result["tflow_buy_ratio"] == pytest.approx(1.0)

    def test_all_sell_cvd_is_minus_one(self, all_sell_trades: pd.DataFrame) -> None:
        """모두 sell → CVD = -1.0."""
        result = compute_bar_features(all_sell_trades)
        assert result["tflow_cvd"] == pytest.approx(-1.0)
        assert result["tflow_buy_ratio"] == pytest.approx(0.0)

    def test_abs_order_imbalance(self, sample_trades: pd.DataFrame) -> None:
        """abs_order_imbalance = |buy - sell| / total."""
        result = compute_bar_features(sample_trades)
        # |6 - 4| / 10 = 0.2
        assert result["tflow_abs_order_imbalance"] == pytest.approx(0.2)

    def test_large_ratio_with_enough_trades(self, large_trades: pd.DataFrame) -> None:
        """충분한 trades가 있을 때 large_ratio > 0."""
        result = compute_bar_features(large_trades)
        assert result["tflow_large_ratio"] > 0.0
        assert result["tflow_large_ratio"] <= 1.0

    def test_large_ratio_insufficient_trades(self) -> None:
        """trades < 20 → large_ratio = 0.0."""
        df = pd.DataFrame(
            {
                "quantity": [1.0] * 10,
                "is_buyer_maker": [False] * 10,
            }
        )
        result = compute_bar_features(df)
        assert result["tflow_large_ratio"] == 0.0

    def test_custom_bar_hours(self, sample_trades: pd.DataFrame) -> None:
        """bar_hours 변경 시 intensity 변화."""
        result_12h = compute_bar_features(sample_trades, bar_hours=12.0)
        result_4h = compute_bar_features(sample_trades, bar_hours=4.0)
        assert result_4h["tflow_intensity"] == pytest.approx(result_12h["tflow_intensity"] * 3)

    def test_returns_all_expected_keys(self, sample_trades: pd.DataFrame) -> None:
        """모든 피처 키 반환 확인."""
        result = compute_bar_features(sample_trades)
        expected_keys = {
            "tflow_cvd",
            "tflow_buy_ratio",
            "tflow_intensity",
            "tflow_large_ratio",
            "tflow_abs_order_imbalance",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# compute_vpin tests
# ---------------------------------------------------------------------------


class TestComputeVpin:
    def test_basic_vpin(self) -> None:
        """기본 VPIN rolling mean 계산."""
        bar_features = pd.DataFrame(
            {
                "tflow_abs_order_imbalance": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        vpin = compute_vpin(bar_features, window=3)

        assert vpin.name == "tflow_vpin"
        assert len(vpin) == 5
        # 마지막 값: mean(0.3, 0.4, 0.5) = 0.4
        assert vpin.iloc[-1] == pytest.approx(0.4)

    def test_vpin_min_periods(self) -> None:
        """min_periods=1 → 첫 번째 값도 NaN이 아님."""
        bar_features = pd.DataFrame(
            {
                "tflow_abs_order_imbalance": [0.5, 0.3],
            }
        )
        vpin = compute_vpin(bar_features, window=10)
        assert not vpin.isna().any()
        assert vpin.iloc[0] == pytest.approx(0.5)

    def test_vpin_empty_dataframe(self) -> None:
        """빈 DataFrame → 빈 Series."""
        vpin = compute_vpin(pd.DataFrame())
        assert len(vpin) == 0
        assert vpin.name == "tflow_vpin"

    def test_vpin_missing_column(self) -> None:
        """필수 컬럼 없음 → 빈 Series."""
        bar_features = pd.DataFrame({"other_col": [1, 2, 3]})
        vpin = compute_vpin(bar_features)
        assert len(vpin) == 0

    def test_vpin_range(self) -> None:
        """VPIN은 [0, 1] 범위."""
        bar_features = pd.DataFrame(
            {
                "tflow_abs_order_imbalance": np.random.default_rng(42).uniform(0, 1, size=50),
            }
        )
        vpin = compute_vpin(bar_features, window=10)
        assert vpin.min() >= 0.0
        assert vpin.max() <= 1.0
