"""Information Coefficient (IC) Quick Check module.

전략 구현 전 지표 예측력을 사전 검증합니다.
Rank IC, IC IR, Hit Rate 등을 분석하여 P4 FAIL 가능성을 사전 필터링합니다.

Usage:
    >>> from src.backtest.ic_analyzer import ICAnalyzer
    >>> result = ICAnalyzer.analyze(indicator_series, forward_returns)
    >>> print(result.verdict)  # PASS or FAIL
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from scipy import stats


class ICVerdict(StrEnum):
    """IC 분석 판정 결과."""

    PASS = "PASS"  # noqa: S105
    FAIL = "FAIL"


# ─── Pass Thresholds ────────────────────────────────────────────────

IC_ABS_THRESHOLD = 0.02
IC_IR_ABS_THRESHOLD = 0.1
HIT_RATE_THRESHOLD = 52.0
_ROLLING_IC_WINDOW = 60
_DECAY_QUARTERS = 4  # 최근 1년 (252 bars) / 63 bars per quarter
_MIN_SAMPLES_RANK_IC = 10
_MIN_SAMPLES_IC_IR = 2


@dataclass(frozen=True)
class ICResult:
    """IC 분석 결과."""

    rank_ic: float
    rank_ic_pvalue: float
    ic_ir: float
    ic_decay_stable: bool
    hit_rate: float
    verdict: ICVerdict


@dataclass(frozen=True)
class ICBatchEntry:
    """IC batch 개별 결과."""

    indicator_name: str
    result: ICResult | None
    error: str | None = None


@dataclass(frozen=True)
class ICBatchResult:
    """IC batch 전체 결과."""

    entries: list[ICBatchEntry]
    total: int
    passed: int
    failed: int
    skipped: int


class ICAnalyzer:
    """Information Coefficient 분석기.

    지표와 forward return 간 Spearman rank correlation을 측정합니다.
    """

    @staticmethod
    def analyze(indicator: pd.Series, forward_returns: pd.Series) -> ICResult:  # type: ignore[type-arg]
        """전체 IC 분석 실행.

        Args:
            indicator: 지표 시리즈
            forward_returns: 다음 기간 수익률 시리즈

        Returns:
            ICResult with verdict
        """
        ic, pvalue = ICAnalyzer.rank_ic(indicator, forward_returns)
        rolling = ICAnalyzer.rolling_ic(indicator, forward_returns)
        ir = ICAnalyzer.ic_ir(rolling)
        stable = ICAnalyzer.ic_decay_stable(rolling)
        hr = ICAnalyzer.hit_rate(indicator, forward_returns)

        passed = (
            abs(ic) > IC_ABS_THRESHOLD
            and abs(ir) > IC_IR_ABS_THRESHOLD
            and stable
            and hr > HIT_RATE_THRESHOLD
        )

        return ICResult(
            rank_ic=ic,
            rank_ic_pvalue=pvalue,
            ic_ir=ir,
            ic_decay_stable=stable,
            hit_rate=hr,
            verdict=ICVerdict.PASS if passed else ICVerdict.FAIL,
        )

    @staticmethod
    def rank_ic(indicator: pd.Series, forward_returns: pd.Series) -> tuple[float, float]:  # type: ignore[type-arg]
        """Spearman rank correlation 계산.

        Args:
            indicator: 지표 시리즈
            forward_returns: forward return 시리즈

        Returns:
            (correlation, p-value) tuple
        """
        aligned = pd.DataFrame({"ind": indicator, "ret": forward_returns}).dropna()
        if len(aligned) < _MIN_SAMPLES_RANK_IC:
            return 0.0, 1.0

        spearman_result = stats.spearmanr(aligned["ind"], aligned["ret"])
        corr = float(spearman_result[0])  # type: ignore[arg-type]
        pvalue = float(spearman_result[1])  # type: ignore[arg-type]
        return corr, pvalue

    @staticmethod
    def rolling_ic(  # type: ignore[type-arg]
        indicator: pd.Series,
        forward_returns: pd.Series,
        window: int = _ROLLING_IC_WINDOW,
    ) -> pd.Series:  # type: ignore[type-arg]
        """Rolling Spearman rank correlation 시리즈.

        Args:
            indicator: 지표 시리즈
            forward_returns: forward return 시리즈
            window: rolling window 크기

        Returns:
            rolling IC 시리즈
        """
        aligned = pd.DataFrame({"ind": indicator, "ret": forward_returns}).dropna()
        n = len(aligned)

        ic_values: list[float] = []
        indices: list[object] = []

        for i in range(window, n + 1):
            chunk = aligned.iloc[i - window : i]
            sp = stats.spearmanr(chunk["ind"], chunk["ret"])
            ic_values.append(float(sp[0]))  # type: ignore[arg-type]
            indices.append(aligned.index[i - 1])

        return pd.Series(ic_values, index=indices, name="rolling_ic")

    @staticmethod
    def ic_ir(rolling_ic_series: pd.Series) -> float:  # type: ignore[type-arg]
        """IC Information Ratio (IC의 Sharpe ratio).

        Args:
            rolling_ic_series: rolling IC 시리즈

        Returns:
            IC IR 값
        """
        clean = rolling_ic_series.dropna()
        if len(clean) < _MIN_SAMPLES_IC_IR:
            return 0.0

        std = float(clean.std())
        if std == 0 or np.isnan(std):
            return 0.0

        return float(clean.mean()) / std

    @staticmethod
    def ic_decay_stable(rolling_ic_series: pd.Series) -> bool:  # type: ignore[type-arg]
        """최근 1년 rolling IC 분기별 부호 일관성.

        최근 252 bar의 rolling IC를 분기별(63 bar)로 나누어
        모든 분기의 평균 IC 부호가 동일하면 stable.

        Args:
            rolling_ic_series: rolling IC 시리즈

        Returns:
            True if decay stable
        """
        clean = rolling_ic_series.dropna()
        quarter_size = 63
        total_needed = quarter_size * _DECAY_QUARTERS

        if len(clean) < total_needed:
            # 데이터 부족 시 전체 부호 일관성으로 판단
            if len(clean) < quarter_size:
                return True  # 판단 불가 시 pass
            recent = clean.iloc[-len(clean) :]
            n_quarters = max(1, len(recent) // quarter_size)
            signs = []
            for q in range(n_quarters):
                chunk = recent.iloc[q * quarter_size : (q + 1) * quarter_size]
                if len(chunk) > 0:
                    signs.append(np.sign(float(chunk.mean())))
            if not signs:
                return True
            return bool(all(s == signs[0] for s in signs) and signs[0] != 0)

        recent = clean.iloc[-total_needed:]
        signs = []
        for q in range(_DECAY_QUARTERS):
            chunk = recent.iloc[q * quarter_size : (q + 1) * quarter_size]
            signs.append(np.sign(float(chunk.mean())))

        # 모든 분기 부호 동일 & 0이 아님
        return bool(all(s == signs[0] for s in signs) and signs[0] != 0)

    @staticmethod
    def hit_rate(indicator: pd.Series, forward_returns: pd.Series) -> float:  # type: ignore[type-arg]
        """방향 일치율 (%).

        indicator 부호와 forward_return 부호가 일치하는 비율.

        Args:
            indicator: 지표 시리즈
            forward_returns: forward return 시리즈

        Returns:
            hit rate (0~100)
        """
        aligned = pd.DataFrame({"ind": indicator, "ret": forward_returns}).dropna()
        # 0 부호 제외 (방향성 없는 데이터)
        mask = (aligned["ind"] != 0) & (aligned["ret"] != 0)
        active = aligned[mask]

        if len(active) == 0:
            return 0.0

        matches = (np.sign(active["ind"]) == np.sign(active["ret"])).sum()
        return float(matches) / len(active) * 100
