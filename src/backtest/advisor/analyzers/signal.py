"""Signal quality analyzer.

시그널의 예측력과 효율성을 분석합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.backtest.advisor.models import SignalQuality

if TYPE_CHECKING:
    import pandas as pd

    from src.models.backtest import BacktestResult

# 보유 기간 분류 임계값 (시간)
_HOURS_1 = 1
_HOURS_4 = 4
_HOURS_24 = 24
_HOURS_168 = 168  # 1주일


class SignalAnalyzer:
    """시그널 품질 분석기.

    시그널의 적중률, 손익비, 보유 기간 등을 분석합니다.

    Example:
        >>> analyzer = SignalAnalyzer()
        >>> result = analyzer.analyze(backtest_result, returns)
        >>> print(result.hit_rate)
    """

    def analyze(
        self,
        result: BacktestResult,
        returns: pd.Series,  # type: ignore[type-arg]
    ) -> SignalQuality:
        """시그널 품질 분석 수행.

        Args:
            result: 백테스트 결과
            returns: 수익률 시리즈

        Returns:
            SignalQuality 분석 결과
        """
        trades = result.trades
        metrics = result.metrics

        # 기본 지표
        hit_rate = metrics.win_rate
        profit_factor = metrics.profit_factor
        avg_win = metrics.avg_win if metrics.avg_win is not None else 0.0
        avg_loss = metrics.avg_loss if metrics.avg_loss is not None else 0.0

        # 손익비 계산
        risk_reward_ratio: float | None = None
        if avg_loss != 0:
            risk_reward_ratio = abs(avg_win / avg_loss)

        # 보유 기간 분석
        avg_holding, holding_dist = self._analyze_holding_periods(trades)

        # 시그널 효율 (거래 횟수 / 기간)
        signal_count = len(returns)  # 대략적인 시그널 수
        trade_count = len(trades)

        return SignalQuality(
            hit_rate=hit_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            risk_reward_ratio=risk_reward_ratio,
            avg_holding_periods=avg_holding,
            holding_distribution=holding_dist,
            signal_count=signal_count,
            trade_count=trade_count,
        )

    def _analyze_holding_periods(
        self,
        trades: tuple,  # type: ignore[type-arg]
    ) -> tuple[float, dict[str, int]]:
        """보유 기간 분석.

        Returns:
            (평균 보유 기간, 분포)
        """
        if not trades:
            return 0.0, {"<1h": 0, "1h-4h": 0, "4h-1d": 0, "1d-1w": 0, ">1w": 0}

        holding_periods: list[float] = []

        for trade in trades:
            if trade.exit_time and trade.entry_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                holding_periods.append(duration)

        if not holding_periods:
            return 0.0, {"<1h": 0, "1h-4h": 0, "4h-1d": 0, "1d-1w": 0, ">1w": 0}

        avg_holding = sum(holding_periods) / len(holding_periods)

        # 분포 계산
        distribution = {"<1h": 0, "1h-4h": 0, "4h-1d": 0, "1d-1w": 0, ">1w": 0}
        for hours in holding_periods:
            if hours < _HOURS_1:
                distribution["<1h"] += 1
            elif hours < _HOURS_4:
                distribution["1h-4h"] += 1
            elif hours < _HOURS_24:
                distribution["4h-1d"] += 1
            elif hours < _HOURS_168:
                distribution["1d-1w"] += 1
            else:
                distribution[">1w"] += 1

        return avg_holding, distribution
