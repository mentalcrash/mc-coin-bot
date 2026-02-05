"""Loss concentration analyzer.

손실이 특정 시간대, 요일, 패턴에 집중되는지 분석합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.backtest.advisor.models import LossConcentration

if TYPE_CHECKING:
    from src.models.backtest import BacktestResult


class LossAnalyzer:
    """손실 집중 분석기.

    손실이 특정 시간대, 요일, 패턴에 집중되는지 분석합니다.
    Stateless 설계를 따릅니다.

    Example:
        >>> analyzer = LossAnalyzer()
        >>> result = analyzer.analyze(backtest_result, returns_series)
        >>> print(result.worst_hours)
    """

    def __init__(
        self,
        large_loss_threshold: float = -0.02,
    ) -> None:
        """LossAnalyzer 초기화.

        Args:
            large_loss_threshold: 대규모 손실 임계값 (기본값 -2%)
        """
        self._large_loss_threshold = large_loss_threshold

    def analyze(
        self,
        result: BacktestResult,
        returns: pd.Series,  # type: ignore[type-arg]
    ) -> LossConcentration:
        """손실 집중 분석 수행.

        Args:
            result: 백테스트 결과 (메타데이터용)
            returns: 수익률 시리즈 (DatetimeIndex)

        Returns:
            LossConcentration 분석 결과
        """
        # 손실만 필터링
        losses = returns[returns < 0]

        # 시간대별 PnL 분석
        hourly_pnl = self._analyze_hourly(returns)
        worst_hours = self._get_worst_hours(hourly_pnl, top_n=3)

        # 요일별 PnL 분석
        weekday_pnl = self._analyze_weekday(returns)
        worst_weekdays = self._get_worst_weekdays(weekday_pnl, top_n=2)

        # 연속 손실 분석
        max_consecutive, avg_consecutive = self._analyze_consecutive_losses(returns)

        # 대규모 손실 분석
        large_losses = losses[losses < self._large_loss_threshold]
        large_loss_count = len(large_losses)
        large_loss_total = float(large_losses.sum() * 100) if len(large_losses) > 0 else 0.0

        return LossConcentration(
            hourly_pnl=hourly_pnl,
            worst_hours=tuple(worst_hours),
            weekday_pnl=weekday_pnl,
            worst_weekdays=tuple(worst_weekdays),
            max_consecutive_losses=max_consecutive,
            avg_consecutive_losses=avg_consecutive,
            large_loss_threshold=self._large_loss_threshold * 100,
            large_loss_count=large_loss_count,
            large_loss_total=large_loss_total,
        )

    def _analyze_hourly(
        self,
        returns: pd.Series,  # type: ignore[type-arg]
    ) -> dict[int, float]:
        """시간대별 PnL 분석."""
        if not isinstance(returns.index, pd.DatetimeIndex):
            return dict.fromkeys(range(24), 0.0)

        # DatetimeIndex.hour는 실제로 존재하지만 pyright stubs에서 인식 못함
        hourly_returns = returns.groupby(
            returns.index.hour  # type: ignore[union-attr]
        ).sum()
        return {
            int(h): float(v * 100)  # type: ignore[arg-type]
            for h, v in hourly_returns.items()
        }

    def _analyze_weekday(
        self,
        returns: pd.Series,  # type: ignore[type-arg]
    ) -> dict[int, float]:
        """요일별 PnL 분석."""
        if not isinstance(returns.index, pd.DatetimeIndex):
            return dict.fromkeys(range(7), 0.0)

        # DatetimeIndex.dayofweek는 실제로 존재하지만 pyright stubs에서 인식 못함
        weekday_returns = returns.groupby(
            returns.index.dayofweek  # type: ignore[union-attr]
        ).sum()
        return {
            int(d): float(v * 100)  # type: ignore[arg-type]
            for d, v in weekday_returns.items()
        }

    def _get_worst_hours(
        self,
        hourly_pnl: dict[int, float],
        top_n: int = 3,
    ) -> list[int]:
        """손실이 가장 큰 시간대 반환."""
        sorted_hours = sorted(hourly_pnl.items(), key=lambda x: x[1])
        return [h for h, _ in sorted_hours[:top_n]]

    def _get_worst_weekdays(
        self,
        weekday_pnl: dict[int, float],
        top_n: int = 2,
    ) -> list[int]:
        """손실이 가장 큰 요일 반환."""
        sorted_days = sorted(weekday_pnl.items(), key=lambda x: x[1])
        return [d for d, _ in sorted_days[:top_n]]

    def _analyze_consecutive_losses(
        self,
        returns: pd.Series,  # type: ignore[type-arg]
    ) -> tuple[int, float]:
        """연속 손실 분석.

        Returns:
            (최대 연속 손실, 평균 연속 손실)
        """
        is_loss = (returns < 0).astype(int)

        # 연속 손실 그룹 식별
        loss_groups = (is_loss != is_loss.shift()).cumsum()
        loss_streaks = is_loss.groupby(loss_groups).sum()

        # 손실 streak만 필터링 (값이 0보다 큰 것)
        loss_streaks = loss_streaks[loss_streaks > 0]

        if len(loss_streaks) == 0:
            return 0, 0.0

        max_consecutive = int(loss_streaks.max())
        avg_consecutive = float(loss_streaks.mean())

        return max_consecutive, avg_consecutive
