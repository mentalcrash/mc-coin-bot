"""Deflated Sharpe Ratio (DSR) implementation.

Bailey & Lopez de Prado (2014) 논문 기반.
다중 테스트 보정된 Sharpe Ratio를 계산합니다.

높은 n_trials(테스트한 전략/파라미터 수) → 높은 기대 최대 Sharpe →
더 엄격한 판정 기준으로 과적합을 방지합니다.

Reference:
    Bailey, D.H. & Lopez de Prado, M. (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias,
    Backtest Overfitting and Non-Normality"
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

_MIN_OBSERVATIONS = 2


def expected_max_sharpe(
    n_trials: int,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Expected Maximum Sharpe Ratio under multiple testing.

    n_trials개의 독립 전략 중 최고 Sharpe의 기대값을 계산합니다.
    (정규 분포에서 n개 표본의 기대 최대값)

    Args:
        n_trials: 테스트한 전략/파라미터 수
        skewness: 수익률 왜도 (0 = 정규분포)
        kurtosis: 수익률 첨도 (3 = 정규분포)

    Returns:
        Expected maximum Sharpe ratio (E[max(SR)])
    """
    if n_trials <= 1:
        return 0.0

    # E[max] of standard normal for n samples
    # Approximation using Euler-Mascheroni constant (0.5772)
    euler_mascheroni = 0.5772156649

    z1 = float(stats.norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    e_max = (1 - euler_mascheroni) * z1 + euler_mascheroni * z2

    # Non-normality correction (Cornish-Fisher expansion)
    excess_kurtosis = kurtosis - 3.0
    correction = (
        e_max
        + (skewness / 6.0) * (e_max**2 - 1)
        + (excess_kurtosis / 24.0) * (e_max**3 - 3 * e_max)
        - (skewness**2 / 36.0) * (2 * e_max**3 - 5 * e_max)
    )

    return max(0.0, correction)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_std: float | None = None,
) -> float:
    """Deflated Sharpe Ratio 계산.

    관측된 Sharpe가 다중 테스트로 인한 우연이 아닌지 검정합니다.
    DSR > 1이면 통계적으로 유의미합니다.

    Args:
        observed_sharpe: 관측된 Sharpe Ratio
        n_trials: 테스트한 전략/파라미터 수
        n_observations: 관측치 수 (데이터 포인트)
        skewness: 수익률 왜도
        kurtosis: 수익률 첨도
        sharpe_std: Sharpe의 표준편차 (None이면 1/sqrt(n)으로 근사)

    Returns:
        Deflated Sharpe Ratio (높을수록 유의미)
    """
    if n_trials < 1 or n_observations < _MIN_OBSERVATIONS:
        return 0.0

    # Expected maximum Sharpe under null hypothesis
    e_max_sr = expected_max_sharpe(n_trials, skewness=skewness, kurtosis=kurtosis)

    # Standard error of Sharpe Ratio
    if sharpe_std is not None:
        se = sharpe_std
    else:
        se = _sharpe_standard_error(
            observed_sharpe, n_observations, skewness=skewness, kurtosis=kurtosis
        )

    if se <= 0:
        return 0.0

    # DSR = PSR(SR*) = Φ((SR_obs - E[max(SR)]) / SE(SR))
    z_score = (observed_sharpe - e_max_sr) / se
    return float(stats.norm.cdf(z_score))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_observations: int,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio (PSR).

    관측된 Sharpe가 벤치마크를 초과할 확률을 계산합니다.

    Args:
        observed_sharpe: 관측된 Sharpe Ratio
        benchmark_sharpe: 벤치마크 Sharpe Ratio
        n_observations: 관측치 수
        skewness: 수익률 왜도
        kurtosis: 수익률 첨도

    Returns:
        PSR (0~1, 높을수록 유의미)
    """
    if n_observations < _MIN_OBSERVATIONS:
        return 0.0

    se = _sharpe_standard_error(
        observed_sharpe, n_observations, skewness=skewness, kurtosis=kurtosis
    )

    if se <= 0:
        return 0.0

    z_score = (observed_sharpe - benchmark_sharpe) / se
    return float(stats.norm.cdf(z_score))


def _sharpe_standard_error(
    sharpe: float,
    n_observations: int,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Sharpe Ratio의 표준오차 계산.

    Lo (2002) 공식 기반:
    SE(SR) = sqrt((1 - γ₁ * SR + (γ₂ - 1)/4 * SR²) / (n - 1))

    Args:
        sharpe: Sharpe Ratio
        n_observations: 관측치 수
        skewness: 수익률 왜도
        kurtosis: 수익률 첨도

    Returns:
        Sharpe Ratio의 표준오차
    """
    if n_observations < _MIN_OBSERVATIONS:
        return 0.0

    excess_kurtosis = kurtosis - 3.0
    variance = (1.0 - skewness * sharpe + ((excess_kurtosis + 2.0) / 4.0) * sharpe**2) / (
        n_observations - 1
    )

    return float(np.sqrt(max(0.0, variance)))
