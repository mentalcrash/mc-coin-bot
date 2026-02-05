"""Regime profile analyzer.

다양한 시장 레짐에서의 전략 성과를 분석합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.backtest.advisor.models import RegimeProfile

if TYPE_CHECKING:
    from src.models.backtest import BacktestResult

# 최소 데이터 포인트 수
_MIN_DATA_POINTS = 2
# 연환산 팩터 (암호화폐는 365일 거래)
_ANNUALIZATION_DAYS = 365


class RegimeAnalyzer:
    """레짐 프로파일 분석기.

    Bull/Bear/Sideways, 고변동성/저변동성 등 다양한 레짐에서의
    전략 성과를 분석합니다.

    Example:
        >>> analyzer = RegimeAnalyzer()
        >>> result = analyzer.analyze(backtest_result, returns, benchmark_returns)
        >>> print(result.weakest_regime)
    """

    def __init__(
        self,
        trend_lookback: int = 20,
        bull_threshold: float = 0.05,
        bear_threshold: float = -0.05,
        vol_lookback: int = 20,
    ) -> None:
        """RegimeAnalyzer 초기화.

        Args:
            trend_lookback: 추세 판단 기간 (기본값 20)
            bull_threshold: Bull 판단 임계값 (기본값 5%)
            bear_threshold: Bear 판단 임계값 (기본값 -5%)
            vol_lookback: 변동성 계산 기간 (기본값 20)
        """
        self._trend_lookback = trend_lookback
        self._bull_threshold = bull_threshold
        self._bear_threshold = bear_threshold
        self._vol_lookback = vol_lookback

    def analyze(
        self,
        result: BacktestResult,
        returns: pd.Series,  # type: ignore[type-arg]
        benchmark_returns: pd.Series,  # type: ignore[type-arg]
    ) -> RegimeProfile:
        """레짐 프로파일 분석 수행.

        Args:
            result: 백테스트 결과 (메타데이터용)
            returns: 전략 수익률 시리즈
            benchmark_returns: 벤치마크 수익률 시리즈

        Returns:
            RegimeProfile 분석 결과
        """
        # 레짐 분류
        trend_regime = self._classify_trend_regime(benchmark_returns)
        vol_regime = self._classify_vol_regime(returns)

        # 레짐별 Sharpe 계산
        bull_sharpe = self._calculate_regime_sharpe(returns, trend_regime == "bull")
        bear_sharpe = self._calculate_regime_sharpe(returns, trend_regime == "bear")
        sideways_sharpe = self._calculate_regime_sharpe(returns, trend_regime == "sideways")

        high_vol_sharpe = self._calculate_regime_sharpe(returns, vol_regime == "high")
        low_vol_sharpe = self._calculate_regime_sharpe(returns, vol_regime == "low")

        # 레짐 분포 계산
        regime_distribution = self._calculate_regime_distribution(trend_regime)

        # 가장 약한 레짐 식별
        regime_sharpes = {
            "bull": bull_sharpe,
            "bear": bear_sharpe,
            "sideways": sideways_sharpe,
        }
        weakest_regime = min(regime_sharpes, key=lambda k: regime_sharpes[k])

        return RegimeProfile(
            bull_sharpe=bull_sharpe,
            bear_sharpe=bear_sharpe,
            sideways_sharpe=sideways_sharpe,
            high_vol_sharpe=high_vol_sharpe,
            low_vol_sharpe=low_vol_sharpe,
            regime_distribution=regime_distribution,
            weakest_regime=weakest_regime,
        )

    def _classify_trend_regime(
        self,
        benchmark_returns: pd.Series,  # type: ignore[type-arg]
    ) -> pd.Series:  # type: ignore[type-arg]
        """추세 레짐 분류.

        Returns:
            레짐 시리즈 ("bull", "bear", "sideways")
        """
        rolling_return = benchmark_returns.rolling(window=self._trend_lookback).sum()

        regime = pd.Series(index=benchmark_returns.index, dtype=str)
        regime[:] = "sideways"
        regime[rolling_return > self._bull_threshold] = "bull"
        regime[rolling_return < self._bear_threshold] = "bear"

        return regime

    def _classify_vol_regime(
        self,
        returns: pd.Series,  # type: ignore[type-arg]
    ) -> pd.Series:  # type: ignore[type-arg]
        """변동성 레짐 분류.

        Returns:
            레짐 시리즈 ("high", "low")
        """
        rolling_vol = returns.rolling(window=self._vol_lookback).std()
        median_vol = rolling_vol.median()

        regime = pd.Series(index=returns.index, dtype=str)
        regime[:] = "low"
        regime[rolling_vol > median_vol] = "high"

        return regime

    def _calculate_regime_sharpe(
        self,
        returns: pd.Series,  # type: ignore[type-arg]
        mask: pd.Series,  # type: ignore[type-arg]
    ) -> float:
        """특정 레짐에서의 Sharpe Ratio 계산."""
        regime_returns = returns[mask]

        if len(regime_returns) < _MIN_DATA_POINTS:
            return 0.0

        mean_return = regime_returns.mean()
        std_return = regime_returns.std()

        if std_return == 0 or np.isnan(std_return):
            return 0.0

        # 연환산
        sharpe = (mean_return * _ANNUALIZATION_DAYS) / (std_return * np.sqrt(_ANNUALIZATION_DAYS))

        return float(sharpe)

    def _calculate_regime_distribution(
        self,
        trend_regime: pd.Series,  # type: ignore[type-arg]
    ) -> dict[str, float]:
        """레짐 분포 계산."""
        counts = trend_regime.value_counts(normalize=True)
        return {str(regime): float(pct * 100) for regime, pct in counts.items()}
