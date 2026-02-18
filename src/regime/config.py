"""Regime Detector Configuration.

레짐 감지에 사용되는 설정 모델과 라벨 정의.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing, StrEnum
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RegimeLabel(StrEnum):
    """시장 레짐 라벨.

    Attributes:
        TRENDING: 추세장 (ER 높음, 방향성 강함)
        RANGING: 횡보장 (ER 낮음, 변동성 낮음)
        VOLATILE: 고변동장 (RV 높음, ER 낮음)
    """

    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"


class RegimeDetectorConfig(BaseModel):
    """레짐 감지 설정.

    RV Ratio와 Efficiency Ratio를 기반으로 시장 레짐을 분류합니다.

    Attributes:
        rv_short_window: 단기 RV 윈도우 (bar 수)
        rv_long_window: 장기 RV 윈도우 (bar 수)
        er_window: Efficiency Ratio 윈도우 (bar 수)
        er_trending_threshold: ER 추세 판별 임계값
        rv_expansion_threshold: RV 확장 판별 임계값
        min_hold_bars: 최소 레짐 유지 bar 수 (hysteresis)

    Example:
        >>> config = RegimeDetectorConfig(
        ...     rv_short_window=5,
        ...     rv_long_window=20,
        ...     er_window=10,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    rv_short_window: int = Field(
        default=5,
        ge=2,
        le=50,
        description="단기 RV 윈도우 (bar 수)",
    )
    rv_long_window: int = Field(
        default=20,
        ge=5,
        le=200,
        description="장기 RV 윈도우 (bar 수)",
    )
    er_window: int = Field(
        default=10,
        ge=3,
        le=100,
        description="Efficiency Ratio 윈도우 (bar 수)",
    )
    er_trending_threshold: float = Field(
        default=0.40,
        ge=0.1,
        le=0.95,
        description="ER 추세 판별 임계값 (높을수록 엄격)",
    )
    rv_expansion_threshold: float = Field(
        default=1.3,
        ge=1.0,
        le=3.0,
        description="RV 확장 판별 임계값 (rv_ratio > 이 값이면 변동 확대)",
    )
    min_hold_bars: int = Field(
        default=5,
        ge=1,
        le=20,
        description="최소 레짐 유지 bar 수 (hysteresis)",
    )

    @model_validator(mode="after")
    def validate_windows(self) -> Self:
        """윈도우 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: rv_short_window >= rv_long_window일 경우
        """
        if self.rv_short_window >= self.rv_long_window:
            msg = (
                f"rv_short_window ({self.rv_short_window}) must be "
                f"< rv_long_window ({self.rv_long_window})"
            )
            raise ValueError(msg)
        return self


class HMMDetectorConfig(BaseModel):
    """HMM 기반 레짐 감지 설정.

    GaussianHMM expanding window training으로 Bull/Bear/Sideways를 분류합니다.
    전략 관련 필드(short_mode, hedge 등)를 제거한 순수 HMM 설정입니다.

    Attributes:
        n_states: HMM 상태 수
        min_train_window: 최소 학습 윈도우 (bar 수)
        retrain_interval: 재학습 간격 (bar 수)
        n_iter: HMM 학습 반복 횟수
        vol_window: 롤링 변동성 윈도우
        use_log_returns: 로그 수익률 사용 여부
    """

    model_config = ConfigDict(frozen=True)

    n_states: int = Field(default=3, ge=2, le=5, description="HMM 상태 수")
    min_train_window: int = Field(
        default=252, ge=100, le=500, description="최소 학습 윈도우 (bar 수)"
    )
    retrain_interval: int = Field(default=21, ge=1, le=63, description="재학습 간격 (bar 수)")
    n_iter: int = Field(default=100, ge=50, le=500, description="HMM 학습 반복 횟수")
    vol_window: int = Field(default=20, ge=5, le=60, description="롤링 변동성 윈도우")
    use_log_returns: bool = Field(default=True, description="로그 수익률 사용 여부")
    sliding_window: int = Field(
        default=0, ge=0, le=2000, description="학습 윈도우 (0=expanding, 504=2yr)"
    )
    decay_half_life: int = Field(
        default=0, ge=0, le=500, description="가중치 반감기 (0=uniform, 126=6mo)"
    )

    @property
    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
        return self.min_train_window + 1


class VolStructureDetectorConfig(BaseModel):
    """Vol-Structure 기반 레짐 감지 설정.

    단기/장기 변동성 비율과 정규화된 모멘텀으로 레짐을 분류합니다.
    전략 관련 필드를 제거한 순수 감지기 설정입니다.

    Attributes:
        vol_short_window: 단기 변동성 윈도우
        vol_long_window: 장기 변동성 윈도우
        mom_window: 모멘텀 윈도우
    """

    model_config = ConfigDict(frozen=True)

    vol_short_window: int = Field(default=10, ge=5, le=30, description="단기 변동성 윈도우")
    vol_long_window: int = Field(default=60, ge=30, le=120, description="장기 변동성 윈도우")
    mom_window: int = Field(default=20, ge=10, le=60, description="모멘텀 윈도우")

    @model_validator(mode="after")
    def validate_windows(self) -> Self:
        """윈도우 일관성 검증."""
        if self.vol_short_window >= self.vol_long_window:
            msg = (
                f"vol_short_window ({self.vol_short_window}) must be "
                f"< vol_long_window ({self.vol_long_window})"
            )
            raise ValueError(msg)
        return self

    @property
    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
        return max(self.vol_long_window, self.mom_window) + 1


class MSARDetectorConfig(BaseModel):
    """MSAR (Markov-Switching AutoRegression) 감지기 설정.

    statsmodels MarkovAutoregression을 사용하여 레짐을 분류합니다.

    Attributes:
        k_regimes: 레짐 수
        order: AR 차수
        switching_ar: 레짐별 AR 계수 전환 여부
        switching_variance: 레짐별 분산 전환 여부
        sliding_window: 학습 윈도우 (0=expanding, 504=2yr)
        retrain_interval: 재학습 간격 (bar 수)
        min_train_window: 최소 학습 윈도우 (bar 수)
        use_log_returns: 로그 수익률 사용 여부
    """

    model_config = ConfigDict(frozen=True)

    k_regimes: int = Field(default=3, ge=2, le=5, description="레짐 수")
    order: int = Field(default=2, ge=1, le=5, description="AR 차수")
    switching_ar: bool = Field(default=True, description="레짐별 AR 계수 전환")
    switching_variance: bool = Field(default=True, description="레짐별 분산 전환")
    sliding_window: int = Field(
        default=504, ge=0, le=2000, description="학습 윈도우 (0=expanding, 504=2yr)"
    )
    retrain_interval: int = Field(default=21, ge=1, le=63, description="재학습 간격 (bar 수)")
    min_train_window: int = Field(
        default=252, ge=100, le=500, description="최소 학습 윈도우 (bar 수)"
    )
    use_log_returns: bool = Field(default=True, description="로그 수익률 사용 여부")

    @property
    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
        return self.min_train_window + 1


class MetaLearnerConfig(BaseModel):
    """Meta-Learner 앙상블 설정.

    LogisticRegression walk-forward stacking으로 감지기 확률을 결합합니다.

    Attributes:
        regularization: LogisticRegression C 파라미터 (작을수록 강한 정규화)
        train_window: walk-forward 학습 윈도우 (bar 수)
        retrain_interval: 재학습 간격 (bar 수)
        forward_return_window: 사후 라벨 생성 윈도우 (bar 수)
        trending_threshold: TRENDING 판별 임계값 (|return| > threshold)
        volatile_threshold: VOLATILE 판별 임계값 (|return| > threshold)
    """

    model_config = ConfigDict(frozen=True)

    regularization: float = Field(default=0.1, gt=0.0, le=10.0, description="LogisticRegression C")
    train_window: int = Field(default=504, ge=100, le=2000, description="walk-forward 학습 윈도우")
    retrain_interval: int = Field(default=63, ge=1, le=252, description="재학습 간격")
    forward_return_window: int = Field(default=20, ge=1, le=60, description="사후 라벨 생성 윈도우")
    trending_threshold: float = Field(
        default=0.03, ge=0.0, le=0.2, description="TRENDING 판별 임계값"
    )
    volatile_threshold: float = Field(
        default=0.05, ge=0.0, le=0.3, description="VOLATILE 판별 임계값"
    )


class DerivativesDetectorConfig(BaseModel):
    """Derivatives 기반 보조 레짐 감지기 설정.

    Funding rate z-score, OI 변화율, funding persistence로
    leverage 축적/해소와 cascade risk를 감지합니다.

    Attributes:
        funding_zscore_window: funding rate rolling z-score 윈도우
        oi_change_window: OI 변화율 기간
        funding_persistence_window: 연속 동방향 funding rate 감시 윈도우
        cascade_risk_threshold: 고위험 cascade risk 임계값
    """

    model_config = ConfigDict(frozen=True)

    funding_zscore_window: int = Field(default=7, ge=3, le=30)
    oi_change_window: int = Field(default=1, ge=1, le=7)
    funding_persistence_window: int = Field(default=14, ge=5, le=30)
    cascade_risk_threshold: float = Field(default=0.7, ge=0.3, le=1.0)

    @property
    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
        return max(self.funding_zscore_window, self.funding_persistence_window) + 1


class EnsembleRegimeDetectorConfig(BaseModel):
    """앙상블 레짐 감지 설정.

    Rule-Based + HMM + Vol-Structure + MSAR + Derivatives 감지기의
    가중 확률 블렌딩 설정입니다.

    Attributes:
        rule_based: Rule-Based 감지기 설정
        hmm: HMM 감지기 설정 (None이면 비활성)
        vol_structure: Vol-Structure 감지기 설정 (None이면 비활성)
        msar: MSAR 감지기 설정 (None이면 비활성)
        derivatives: Derivatives 감지기 설정 (None이면 비활성)
        weight_rule_based: Rule-Based 가중치
        weight_hmm: HMM 가중치
        weight_vol_structure: Vol-Structure 가중치
        weight_msar: MSAR 가중치
        weight_derivatives: Derivatives 가중치
        min_hold_bars: 최소 레짐 유지 bar 수 (hysteresis)
    """

    model_config = ConfigDict(frozen=True)

    rule_based: RegimeDetectorConfig = Field(default_factory=RegimeDetectorConfig)
    hmm: HMMDetectorConfig | None = Field(default=None, description="HMM 설정 (None=비활성)")
    vol_structure: VolStructureDetectorConfig | None = Field(
        default=None, description="Vol-Structure 설정 (None=비활성)"
    )

    msar: MSARDetectorConfig | None = Field(default=None, description="MSAR 설정 (None=비활성)")
    derivatives: DerivativesDetectorConfig | None = Field(
        default=None, description="Derivatives 설정 (None=비활성)"
    )

    weight_rule_based: float = Field(default=1.0, ge=0.0, le=1.0)
    weight_hmm: float = Field(default=0.0, ge=0.0, le=1.0)
    weight_vol_structure: float = Field(default=0.0, ge=0.0, le=1.0)
    weight_msar: float = Field(default=0.0, ge=0.0, le=1.0)
    weight_derivatives: float = Field(default=0.0, ge=0.0, le=1.0)

    min_hold_bars: int = Field(default=5, ge=1, le=20, description="최소 레짐 유지 bar 수")

    ensemble_method: Literal["weighted_average", "meta_learner"] = Field(
        default="weighted_average", description="앙상블 방식"
    )
    meta_learner: MetaLearnerConfig | None = Field(
        default=None, description="Meta-learner 설정 (ensemble_method='meta_learner' 시 필수)"
    )

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        """활성 감지기 가중치 합 = 1.0 검증 (weighted_average 모드)."""
        # meta_learner 모드에서는 가중치 검증 불필요
        if self.ensemble_method == "meta_learner":
            if self.meta_learner is None:
                msg = "meta_learner config is required when ensemble_method='meta_learner'"
                raise ValueError(msg)
            return self

        active_total = self.weight_rule_based
        if self.hmm is not None:
            active_total += self.weight_hmm
        if self.vol_structure is not None:
            active_total += self.weight_vol_structure
        if self.msar is not None:
            active_total += self.weight_msar
        if self.derivatives is not None:
            active_total += self.weight_derivatives

        tolerance = 1e-6
        if active_total > 0 and abs(active_total - 1.0) > tolerance:
            msg = f"Active detector weights must sum to 1.0, got {active_total:.4f}"
            raise ValueError(msg)

        return self
