"""Unit tests for TSMOM diagnostics module.

Rules Applied:
    - #17 Testing Standards: Pytest, parametrize
    - #11 Pydantic Modeling: Model validation tests
"""

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import pytest

from src.models.backtest import SignalDiagnosticRecord
from src.strategy.tsmom.diagnostics import (
    DiagnosticCollector,
    collect_diagnostics_from_pipeline,
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


class TestDiagnosticCollector:
    """DiagnosticCollector 클래스 테스트."""

    def test_init(self) -> None:
        """초기화 테스트."""
        collector = DiagnosticCollector("BTC/USDT")
        assert collector.symbol == "BTC/USDT"
        assert len(collector) == 0

    def test_collect_single_record(self) -> None:
        """단일 레코드 수집 테스트."""
        collector = DiagnosticCollector("BTC/USDT")

        record = collector.collect(
            timestamp=datetime.now(UTC),
            close_price=50000.0,
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
        )

        assert len(collector) == 1
        assert isinstance(record, SignalDiagnosticRecord)
        assert record.symbol == "BTC/USDT"

    def test_to_dataframe(self) -> None:
        """DataFrame 변환 테스트."""
        collector = DiagnosticCollector("BTC/USDT")

        # 3개 레코드 수집
        for i in range(3):
            collector.collect(
                timestamp=datetime(2024, 1, i + 1, tzinfo=UTC),
                close_price=50000.0 + i * 100,
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
            )

        df = collector.to_dataframe()

        assert len(df) == 3
        assert "symbol" in df.columns
        assert "close_price" in df.columns
        assert df.index.name == "timestamp"

    def test_suppression_reason_determination(self) -> None:
        """억제 원인 결정 로직 테스트."""
        collector = DiagnosticCollector("BTC/USDT")

        # Trend filter로 억제된 경우
        record = collector.collect(
            timestamp=datetime.now(UTC),
            close_price=50000.0,
            realized_vol_annualized=0.65,
            benchmark_return=0.02,
            raw_momentum=0.15,
            vol_scalar=0.62,
            scaled_momentum=0.093,
            trend_regime=1,
            signal_before_trend_filter=0.5,  # 시그널 있음
            signal_after_trend_filter=0.0,  # 필터로 0이 됨
            deadband_applied=False,
            signal_after_deadband=0.0,
            raw_target_weight=0.0,
            leverage_capped_weight=0.0,
            final_target_weight=0.0,
            rebalance_triggered=True,
        )

        assert record.signal_suppression_reason == "trend_filter"

    def test_clear(self) -> None:
        """레코드 초기화 테스트."""
        collector = DiagnosticCollector("BTC/USDT")

        collector.collect(
            timestamp=datetime.now(UTC),
            close_price=50000.0,
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
        )

        assert len(collector) == 1
        collector.clear()
        assert len(collector) == 0


class TestCollectDiagnosticsFromPipeline:
    """collect_diagnostics_from_pipeline 함수 테스트."""

    def test_with_sample_data(self, sample_processed_df: pd.DataFrame) -> None:
        """샘플 데이터로 진단 수집 테스트."""
        n = len(sample_processed_df)

        # 가상 시그널 데이터 생성
        signal_before_trend = pd.Series(0.1, index=sample_processed_df.index)
        signal_after_trend = pd.Series(0.1, index=sample_processed_df.index)
        signal_after_deadband = pd.Series(0.1, index=sample_processed_df.index)
        deadband_mask = pd.Series(False, index=sample_processed_df.index)
        final_weights = pd.Series(0.1, index=sample_processed_df.index)

        diagnostics_df = collect_diagnostics_from_pipeline(
            processed_df=sample_processed_df,
            symbol="BTC/USDT",
            signal_before_trend=signal_before_trend,
            signal_after_trend=signal_after_trend,
            signal_after_deadband=signal_after_deadband,
            deadband_mask=deadband_mask,
            final_weights=final_weights,
        )

        assert len(diagnostics_df) == n
        assert "symbol" in diagnostics_df.columns
        assert "signal_suppression_reason" in diagnostics_df.columns

    def test_suppression_reasons_vectorized(
        self, sample_processed_df: pd.DataFrame
    ) -> None:
        """벡터화된 억제 원인 결정 테스트."""
        n = len(sample_processed_df)

        # 첫 번째 절반은 trend_filter로 억제
        signal_before_trend = pd.Series(0.5, index=sample_processed_df.index)
        signal_after_trend = pd.Series(
            [0.0] * (n // 2) + [0.5] * (n - n // 2),
            index=sample_processed_df.index,
        )
        signal_after_deadband = signal_after_trend.copy()
        deadband_mask = pd.Series(False, index=sample_processed_df.index)
        final_weights = signal_after_trend.copy()

        diagnostics_df = collect_diagnostics_from_pipeline(
            processed_df=sample_processed_df,
            symbol="BTC/USDT",
            signal_before_trend=signal_before_trend,
            signal_after_trend=signal_after_trend,
            signal_after_deadband=signal_after_deadband,
            deadband_mask=deadband_mask,
            final_weights=final_weights,
        )

        # 첫 절반은 trend_filter로 억제
        trend_filter_count = (
            diagnostics_df["signal_suppression_reason"] == "trend_filter"
        ).sum()
        assert trend_filter_count >= n // 2 - 1  # 약간의 오차 허용
