"""Unit tests for TSMOM diagnostics module.

Rules Applied:
    - #17 Testing Standards: Pytest, parametrize
    - #11 Pydantic Modeling: Model validation tests
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.models.backtest import SignalDiagnosticRecord
from src.strategy.tsmom.diagnostics import (
    collect_diagnostics_from_signals,
    log_diagnostic_summary,
)


class TestSignalDiagnosticRecord:
    """SignalDiagnosticRecord 모델 테스트."""

    def test_create_valid_record(self) -> None:
        """유효한 진단 레코드 생성 테스트."""
        record = SignalDiagnosticRecord(
            timestamp=datetime.now(UTC),
            symbol="BTC/USDT",
            close_price=Decimal(50000),
            realized_vol_annualized=0.65,
            benchmark_return=0.02,
            raw_momentum=0.15,
            vol_scalar=0.62,
            scaled_momentum=0.093,
            trend_regime=1,
            signal_before_trend_filter=0.093,
            signal_after_trend_filter=0.093,
            deadband_applied=False,
            signal_after_deadband=0.093,
            raw_target_weight=0.093,
            leverage_capped_weight=0.093,
            final_target_weight=0.093,
            rebalance_triggered=True,
            stop_loss_triggered=False,
        )

        assert record.symbol == "BTC/USDT"
        assert record.close_price == Decimal(50000)
        assert record.signal_suppression_reason == "none"
        assert record.is_signal_suppressed is False

    def test_computed_fields(self) -> None:
        """계산 필드 테스트."""
        record = SignalDiagnosticRecord(
            timestamp=datetime.now(UTC),
            symbol="BTC/USDT",
            close_price=Decimal(50000),
            realized_vol_annualized=0.65,
            benchmark_return=0.02,
            raw_momentum=0.15,
            vol_scalar=0.62,
            scaled_momentum=0.093,
            trend_regime=1,
            signal_before_trend_filter=0.093,
            signal_after_trend_filter=0.0,  # Trend filter로 억제됨
            deadband_applied=False,
            signal_after_deadband=0.0,
            raw_target_weight=0.093,
            leverage_capped_weight=0.093,
            final_target_weight=0.5,
            rebalance_triggered=True,
            stop_loss_triggered=False,
            signal_suppression_reason="trend_filter",
        )

        assert record.is_signal_suppressed is True
        # beta_contribution = final_target_weight * benchmark_return
        assert record.beta_contribution == pytest.approx(0.5 * 0.02)

    @pytest.mark.parametrize(
        ("trend_regime", "expected"),
        [
            (1, 1),
            (-1, -1),
            (0, 0),
        ],
    )
    def test_trend_regime_values(self, trend_regime: int, expected: int) -> None:
        """Trend regime 값 테스트."""
        record = SignalDiagnosticRecord(
            timestamp=datetime.now(UTC),
            symbol="BTC/USDT",
            close_price=Decimal(50000),
            realized_vol_annualized=0.65,
            benchmark_return=0.02,
            raw_momentum=0.15,
            vol_scalar=0.62,
            scaled_momentum=0.093,
            trend_regime=trend_regime,  # type: ignore[arg-type]
            signal_before_trend_filter=0.093,
            signal_after_trend_filter=0.093,
            deadband_applied=False,
            signal_after_deadband=0.093,
            raw_target_weight=0.093,
            leverage_capped_weight=0.093,
            final_target_weight=0.093,
            rebalance_triggered=True,
            stop_loss_triggered=False,
        )

        assert record.trend_regime == expected


class TestCollectDiagnosticsFromSignals:
    """collect_diagnostics_from_signals 함수 테스트."""

    @pytest.fixture
    def _processed_df(self) -> pd.DataFrame:
        """테스트용 전처리된 DataFrame."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz=UTC)
        rng = np.random.default_rng(42)
        close = 50000.0 + np.cumsum(rng.normal(0, 500, n))

        return pd.DataFrame(
            {
                "close": close,
                "realized_vol": rng.uniform(0.3, 0.8, n),
                "vw_momentum": rng.uniform(-0.5, 0.5, n),
                "vol_scalar": rng.uniform(0.3, 1.5, n),
            },
            index=idx,
        )

    def test_returns_dataframe_with_expected_columns(self, _processed_df: pd.DataFrame) -> None:
        """반환 DataFrame의 컬럼 확인."""
        weights = pd.Series(0.1, index=_processed_df.index)

        result = collect_diagnostics_from_signals(
            processed_df=_processed_df,
            symbol="BTC/USDT",
            final_weights=weights,
        )

        expected_cols = {
            "symbol",
            "close_price",
            "realized_vol_annualized",
            "benchmark_return",
            "raw_momentum",
            "vol_scalar",
            "scaled_momentum",
            "final_target_weight",
        }
        assert expected_cols == set(result.columns)

    def test_index_preserved(self, _processed_df: pd.DataFrame) -> None:
        """입력 DataFrame의 인덱스가 유지되는지 확인."""
        weights = pd.Series(0.1, index=_processed_df.index)

        result = collect_diagnostics_from_signals(
            processed_df=_processed_df,
            symbol="BTC/USDT",
            final_weights=weights,
        )

        pd.testing.assert_index_equal(result.index, _processed_df.index)

    def test_length_matches_input(self, _processed_df: pd.DataFrame) -> None:
        """반환 DataFrame 길이가 입력과 동일."""
        weights = pd.Series(0.5, index=_processed_df.index)

        result = collect_diagnostics_from_signals(
            processed_df=_processed_df,
            symbol="BTC/USDT",
            final_weights=weights,
        )

        assert len(result) == len(_processed_df)

    def test_symbol_column_filled(self, _processed_df: pd.DataFrame) -> None:
        """symbol 컬럼이 올바르게 채워지는지 확인."""
        weights = pd.Series(0.1, index=_processed_df.index)

        result = collect_diagnostics_from_signals(
            processed_df=_processed_df,
            symbol="ETH/USDT",
            final_weights=weights,
        )

        assert (result["symbol"] == "ETH/USDT").all()

    def test_benchmark_return_is_pct_change(self, _processed_df: pd.DataFrame) -> None:
        """benchmark_return이 close의 pct_change인지 확인."""
        weights = pd.Series(0.1, index=_processed_df.index)

        result = collect_diagnostics_from_signals(
            processed_df=_processed_df,
            symbol="BTC/USDT",
            final_weights=weights,
        )

        expected = _processed_df["close"].pct_change().fillna(0)
        pd.testing.assert_series_equal(result["benchmark_return"], expected, check_names=False)

    def test_final_weights_mapped_correctly(self, _processed_df: pd.DataFrame) -> None:
        """final_weights가 final_target_weight와 scaled_momentum에 반영되는지 확인."""
        rng = np.random.default_rng(99)
        weights = pd.Series(rng.uniform(-1, 1, len(_processed_df)), index=_processed_df.index)

        result = collect_diagnostics_from_signals(
            processed_df=_processed_df,
            symbol="BTC/USDT",
            final_weights=weights,
        )

        pd.testing.assert_series_equal(result["final_target_weight"], weights, check_names=False)
        pd.testing.assert_series_equal(result["scaled_momentum"], weights, check_names=False)


class TestLogDiagnosticSummary:
    """log_diagnostic_summary 함수 테스트."""

    def test_empty_dataframe_logs_warning(self) -> None:
        """빈 DataFrame은 warning 로그를 출력."""
        empty_df = pd.DataFrame()

        with patch("src.strategy.tsmom.diagnostics.get_diagnostic_logger") as mock_logger:
            log_diagnostic_summary(empty_df, "BTC/USDT")
            mock_logger.return_value.warning.assert_called_once()

    def test_summary_with_mixed_weights(self) -> None:
        """Long/Short/Neutral 혼합 weights에 대한 요약 로깅."""
        df = pd.DataFrame({"final_target_weight": [0.5, -0.3, 0.0, 0.8, -0.1]})

        with patch("src.strategy.tsmom.diagnostics.get_diagnostic_logger") as mock_logger:
            log_diagnostic_summary(df, "BTC/USDT")
            mock_logger.return_value.info.assert_called_once()

    def test_summary_without_weight_column(self) -> None:
        """final_target_weight 컬럼이 없으면 info 로그 미출력."""
        df = pd.DataFrame({"close_price": [50000, 51000]})

        with patch("src.strategy.tsmom.diagnostics.get_diagnostic_logger") as mock_logger:
            log_diagnostic_summary(df, "BTC/USDT")
            mock_logger.return_value.info.assert_not_called()
