"""Backtest result models.

이 모듈은 백테스트 결과를 표현하는 Pydantic 모델을 정의합니다.
VectorBT 출력과 QuantStats 분석 결과를 통합하여 저장합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, Decimal for prices
    - #10 Python Standards: Modern typing (X | None, list[])
    - #25 QuantStats Standards: 성과 지표 정의
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

# NOTE: TYPE_CHECKING import removed - pandas DataFrame은 별도 파일로 저장
# BetaAttributionResult에서 rolling_attribution_df는 Parquet으로 관리


class TradeRecord(BaseModel):
    """개별 거래 기록.

    백테스트에서 발생한 개별 거래(진입/청산)를 기록합니다.

    Attributes:
        entry_time: 진입 시각
        exit_time: 청산 시각 (오픈 포지션이면 None)
        symbol: 거래 심볼
        direction: 거래 방향 ("LONG" / "SHORT")
        entry_price: 진입가
        exit_price: 청산가 (오픈 포지션이면 None)
        size: 거래 수량
        pnl: 손익 (청산 후)
        pnl_pct: 손익률 (%)
        fees: 수수료
    """

    model_config = ConfigDict(frozen=True)

    entry_time: datetime
    exit_time: datetime | None = None
    symbol: str
    direction: str = Field(..., pattern="^(LONG|SHORT)$")
    entry_price: Decimal = Field(..., gt=0)
    exit_price: Decimal | None = Field(default=None, gt=0)
    size: Decimal = Field(..., gt=0)
    pnl: Decimal | None = None
    pnl_pct: float | None = None
    fees: Decimal = Field(default=Decimal(0))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_closed(self) -> bool:
        """포지션이 청산되었는지 여부."""
        return self.exit_time is not None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_profitable(self) -> bool:
        """수익 거래 여부."""
        if self.pnl is None:
            return False
        return self.pnl > 0


class PerformanceMetrics(BaseModel):
    """성과 지표 집합.

    백테스트 결과의 핵심 성과 지표들을 저장합니다.
    QuantStats 및 VectorBT에서 계산된 값들을 통합합니다.

    Attributes:
        total_return: 총 수익률 (%)
        cagr: 연평균 복리 수익률 (%)
        sharpe_ratio: 샤프 비율
        sortino_ratio: 소르티노 비율
        max_drawdown: 최대 낙폭 (%)
        calmar_ratio: 칼마 비율 (CAGR / MDD)
        win_rate: 승률 (%)
        profit_factor: 수익 팩터 (총 이익 / 총 손실)
        avg_win: 평균 수익 (%)
        avg_loss: 평균 손실 (%)
        total_trades: 총 거래 횟수
        winning_trades: 승리 거래 횟수
        losing_trades: 패배 거래 횟수
        avg_trade_duration: 평균 거래 기간 (시간)
        volatility: 연간 변동성 (%)
        skewness: 수익률 왜도
        kurtosis: 수익률 첨도
    """

    model_config = ConfigDict(frozen=True)

    # 수익률 지표
    total_return: float = Field(..., description="총 수익률 (%)")
    cagr: float = Field(..., description="연평균 복리 수익률 (%)")

    # 위험 조정 지표
    sharpe_ratio: float = Field(..., description="샤프 비율")
    sortino_ratio: float | None = Field(default=None, description="소르티노 비율")
    calmar_ratio: float | None = Field(default=None, description="칼마 비율")

    # 낙폭 지표
    max_drawdown: float = Field(..., description="최대 낙폭 (%)")
    avg_drawdown: float | None = Field(default=None, description="평균 낙폭 (%)")
    max_drawdown_duration: int | None = Field(default=None, description="최장 낙폭 기간 (일)")

    # 거래 통계
    win_rate: float = Field(..., ge=0, le=100, description="승률 (%)")
    profit_factor: float | None = Field(default=None, description="수익 팩터")
    avg_win: float | None = Field(default=None, description="평균 수익 (%)")
    avg_loss: float | None = Field(default=None, description="평균 손실 (%)")
    total_trades: int = Field(..., ge=0, description="총 거래 횟수")
    winning_trades: int = Field(..., ge=0, description="승리 거래 횟수")
    losing_trades: int = Field(..., ge=0, description="패배 거래 횟수")

    # 기간/빈도 지표
    avg_trade_duration: float | None = Field(default=None, description="평균 거래 기간 (시간)")
    trades_per_year: float | None = Field(default=None, description="연간 거래 횟수")

    # 변동성 및 분포
    volatility: float | None = Field(default=None, description="연간 변동성 (%)")
    skewness: float | None = Field(default=None, description="수익률 왜도")
    kurtosis: float | None = Field(default=None, description="수익률 첨도")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def risk_reward_ratio(self) -> float | None:
        """리스크/리워드 비율."""
        if self.avg_win is None or self.avg_loss is None or self.avg_loss == 0:
            return None
        return abs(self.avg_win / self.avg_loss)


class BenchmarkComparison(BaseModel):
    """벤치마크 대비 비교 결과.

    전략 성과를 Buy & Hold 또는 다른 벤치마크와 비교합니다.

    Attributes:
        benchmark_name: 벤치마크 이름 (예: "BTC Buy & Hold")
        benchmark_return: 벤치마크 총 수익률 (%)
        alpha: 초과 수익률 (전략 - 벤치마크)
        beta: 시장 민감도 (베타)
        correlation: 상관계수
        information_ratio: 정보 비율
        tracking_error: 추적 오차 (%)
    """

    model_config = ConfigDict(frozen=True)

    benchmark_name: str = Field(..., description="벤치마크 이름")
    benchmark_return: float = Field(..., description="벤치마크 총 수익률 (%)")
    alpha: float = Field(..., description="초과 수익률 (%)")
    beta: float | None = Field(default=None, description="베타")
    correlation: float | None = Field(default=None, ge=-1, le=1, description="상관계수")
    information_ratio: float | None = Field(default=None, description="정보 비율")
    tracking_error: float | None = Field(default=None, description="추적 오차 (%)")


# =============================================================================
# Diagnostic Models (Signal Pipeline Analysis)
# =============================================================================

# Type alias for signal suppression reasons
type SignalSuppressionReason = Literal[
    "none",
    "trend_filter",
    "deadband",
    "low_volatility",
    "leverage_cap",
    "rebalance_threshold",
    "stop_loss",
]


class SignalDiagnosticRecord(BaseModel):
    """매 캔들마다 기록되는 시그널 진단 데이터.

    전략이 왜 시장 상승분을 포착하지 못하는지 추적하기 위한 진단 레코드입니다.
    각 필터 단계에서의 시그널 상태와 최종 포지션 결정 과정을 기록합니다.

    Attributes:
        timestamp: 캔들 타임스탬프
        symbol: 거래 심볼

        Market State:
            close_price: 종가
            realized_vol_annualized: 실현 변동성 (연율화)
            benchmark_return: 시장(BTC) 수익률

        Signal Generation:
            raw_momentum: 원시 모멘텀 값 (lookback 기간 수익률)
            vol_scalar: vol_target / realized_vol
            scaled_momentum: raw_momentum * vol_scalar

        Filter Decisions:
            trend_regime: 추세 국면 (1=Uptrend, -1=Downtrend, 0=Neutral)
            signal_before_trend_filter: Trend filter 적용 전 시그널
            signal_after_trend_filter: Trend filter 적용 후 시그널
            deadband_applied: Deadband로 신호가 억제되었는지 여부
            signal_after_deadband: Deadband 적용 후 시그널

        Position Sizing:
            raw_target_weight: 필터 적용 전 목표 비중
            leverage_capped_weight: max_leverage 적용 후 비중
            final_target_weight: 최종 목표 비중

        Execution:
            rebalance_triggered: 리밸런싱 threshold 통과 여부
            stop_loss_triggered: Stop loss 발동 여부

        Attribution:
            signal_suppression_reason: 시그널 억제 원인

    Example:
        >>> record = SignalDiagnosticRecord(
        ...     timestamp=datetime.now(UTC),
        ...     symbol="BTC/USDT",
        ...     close_price=Decimal("50000"),
        ...     realized_vol_annualized=0.65,
        ...     benchmark_return=0.02,
        ...     raw_momentum=0.15,
        ...     vol_scalar=0.62,
        ...     scaled_momentum=0.093,
        ...     trend_regime=1,
        ...     signal_before_trend_filter=0.093,
        ...     signal_after_trend_filter=0.093,
        ...     deadband_applied=False,
        ...     signal_after_deadband=0.093,
        ...     raw_target_weight=0.093,
        ...     leverage_capped_weight=0.093,
        ...     final_target_weight=0.093,
        ...     rebalance_triggered=True,
        ...     stop_loss_triggered=False,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # === Identifiers ===
    timestamp: datetime
    symbol: str

    # === Market State ===
    close_price: Decimal = Field(..., gt=0, description="종가")
    realized_vol_annualized: float = Field(..., ge=0, description="실현 변동성 (연율화)")
    benchmark_return: float = Field(..., description="시장(BTC) 수익률")

    # === Signal Generation ===
    raw_momentum: float = Field(..., description="원시 모멘텀 값")
    vol_scalar: float = Field(..., ge=0, description="변동성 스케일러")
    scaled_momentum: float = Field(..., description="스케일링된 모멘텀")

    # === Filter Decisions (Critical for Beta diagnosis) ===
    trend_regime: Literal[1, -1, 0] = Field(..., description="추세 국면")
    signal_before_trend_filter: float = Field(..., description="Trend filter 적용 전 시그널")
    signal_after_trend_filter: float = Field(..., description="Trend filter 적용 후 시그널")
    deadband_applied: bool = Field(..., description="Deadband 적용 여부")
    signal_after_deadband: float = Field(..., description="Deadband 적용 후 시그널")

    # === Position Sizing ===
    raw_target_weight: float = Field(..., description="필터 적용 전 목표 비중")
    leverage_capped_weight: float = Field(..., description="max_leverage 적용 후 비중")
    final_target_weight: float = Field(..., description="최종 목표 비중")

    # === Execution ===
    rebalance_triggered: bool = Field(..., description="리밸런싱 threshold 통과 여부")
    stop_loss_triggered: bool = Field(default=False, description="Stop loss 발동 여부")

    # === Attribution ===
    signal_suppression_reason: SignalSuppressionReason = Field(
        default="none", description="시그널 억제 원인"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_signal_suppressed(self) -> bool:
        """시그널이 억제되었는지 여부."""
        return self.signal_suppression_reason != "none"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def beta_contribution(self) -> float:
        """Beta 기여도 (final_target_weight * benchmark_return)."""
        return self.final_target_weight * self.benchmark_return


class BetaAttributionResult(BaseModel):
    """Beta 분해 분석 결과.

    각 필터가 Beta에 미치는 영향을 정량화한 결과입니다.
    Rolling window 기반으로 계산된 Beta 값과 각 단계별 손실을 포함합니다.

    Attributes:
        potential_beta: 필터 없이 모든 신호를 실행했을 때의 예상 Beta
        beta_after_trend_filter: Trend filter 적용 후 Beta
        beta_after_deadband: Deadband 적용 후 Beta
        realized_beta: 실제 실현된 Beta

        lost_to_trend_filter: Trend filter로 인해 손실된 Beta
        lost_to_deadband: Deadband로 인해 손실된 Beta
        lost_to_vol_scaling: Vol scaling으로 인해 손실된 Beta

        analysis_window: 분석에 사용된 rolling window 크기
        total_periods: 분석 기간 총 캔들 수

    Example:
        >>> result = BetaAttributionResult(
        ...     potential_beta=0.85,
        ...     beta_after_trend_filter=0.65,
        ...     beta_after_deadband=0.55,
        ...     realized_beta=0.40,
        ...     lost_to_trend_filter=0.20,
        ...     lost_to_deadband=0.10,
        ...     lost_to_vol_scaling=0.15,
        ...     analysis_window=60,
        ...     total_periods=365,
        ... )
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # === Beta Values at Each Stage ===
    potential_beta: float = Field(..., description="필터 없이 예상되는 Beta")
    beta_after_trend_filter: float = Field(..., description="Trend filter 적용 후 Beta")
    beta_after_deadband: float = Field(..., description="Deadband 적용 후 Beta")
    realized_beta: float = Field(..., description="실제 실현된 Beta")

    # === Beta Losses (Attribution) ===
    lost_to_trend_filter: float = Field(..., description="Trend filter로 인한 Beta 손실")
    lost_to_deadband: float = Field(..., description="Deadband로 인한 Beta 손실")
    lost_to_vol_scaling: float = Field(..., description="Vol scaling으로 인한 Beta 손실")

    # === Metadata ===
    analysis_window: int = Field(default=60, ge=1, description="Rolling window 크기")
    total_periods: int = Field(default=0, ge=0, description="분석 기간 총 캔들 수")

    # NOTE: rolling_attribution_df는 Pydantic 직렬화 문제로 별도 저장
    # DataFrame은 to_parquet()로 별도 파일에 저장 권장

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_beta_loss(self) -> float:
        """총 Beta 손실량."""
        return self.potential_beta - self.realized_beta

    @computed_field  # type: ignore[prop-decorator]
    @property
    def beta_retention_ratio(self) -> float:
        """Beta 보존 비율 (realized / potential)."""
        if self.potential_beta == 0:
            return 0.0
        return self.realized_beta / self.potential_beta

    def summary(self) -> dict[str, str]:
        """요약 정보 반환."""
        return {
            "potential_beta": f"{self.potential_beta:.3f}",
            "realized_beta": f"{self.realized_beta:.3f}",
            "beta_retention": f"{self.beta_retention_ratio:.1%}",
            "lost_to_trend_filter": f"{self.lost_to_trend_filter:.3f}",
            "lost_to_deadband": f"{self.lost_to_deadband:.3f}",
            "lost_to_vol_scaling": f"{self.lost_to_vol_scaling:.3f}",
        }


class BacktestConfig(BaseModel):
    """백테스트 설정 기록.

    백테스트 실행 시 사용된 설정을 기록합니다.
    재현성(Reproducibility)을 위해 모든 파라미터를 저장합니다.

    Attributes:
        strategy_name: 전략 이름
        symbol: 거래 심볼
        timeframe: 타임프레임
        start_date: 시작일
        end_date: 종료일
        initial_capital: 초기 자본
        maker_fee: 메이커 수수료 (%)
        taker_fee: 테이커 수수료 (%)
        slippage: 슬리피지 (%)
        strategy_params: 전략별 파라미터
    """

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal = Field(default=Decimal(10000))
    maker_fee: float = Field(default=0.0002, ge=0, description="메이커 수수료 (%)")
    taker_fee: float = Field(default=0.0004, ge=0, description="테이커 수수료 (%)")
    slippage: float = Field(default=0.0005, ge=0, description="슬리피지 (%)")
    strategy_params: dict[str, Any] = Field(
        default_factory=dict,
        description="전략별 파라미터",
    )

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """datetime에 UTC timezone 적용."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class BacktestResult(BaseModel):
    """백테스트 결과 종합.

    백테스트 실행 결과의 모든 정보를 통합하여 저장합니다.
    JSON 직렬화하여 파일로 저장하거나 DB에 기록할 수 있습니다.

    Attributes:
        config: 백테스트 설정
        metrics: 성과 지표
        benchmark: 벤치마크 비교 결과
        trades: 거래 기록 목록
        equity_curve_path: Equity Curve 데이터 파일 경로
        report_path: QuantStats HTML 리포트 경로
        created_at: 결과 생성 시각

    Example:
        >>> result = BacktestResult(
        ...     config=config,
        ...     metrics=metrics,
        ...     benchmark=benchmark,
        ...     trades=trades,
        ... )
        >>> result.model_dump_json()  # JSON 직렬화
    """

    model_config = ConfigDict(frozen=True)

    config: BacktestConfig = Field(..., description="백테스트 설정")
    metrics: PerformanceMetrics = Field(..., description="성과 지표")
    benchmark: BenchmarkComparison | None = Field(default=None, description="벤치마크 비교")
    trades: tuple[TradeRecord, ...] = Field(
        default_factory=tuple,
        description="거래 기록",
    )
    equity_curve_path: str | None = Field(default=None, description="Equity Curve 파일 경로")
    report_path: str | None = Field(default=None, description="QuantStats 리포트 경로")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="결과 생성 시각",
    )

    @property
    def total_trades(self) -> int:
        """총 거래 횟수."""
        return len(self.trades)

    @property
    def duration_days(self) -> int:
        """백테스트 기간 (일)."""
        delta = self.config.end_date - self.config.start_date
        return delta.days

    def passed_minimum_criteria(
        self,
        min_sharpe: float = 1.0,
        max_mdd: float = 40.0,
        min_win_rate: float = 40.0,
    ) -> bool:
        """최소 성과 기준 통과 여부.

        Args:
            min_sharpe: 최소 샤프 비율
            max_mdd: 최대 허용 낙폭 (%) - 양수로 입력
            min_win_rate: 최소 승률 (%)

        Returns:
            모든 기준 통과 시 True
        """
        return (
            self.metrics.sharpe_ratio >= min_sharpe
            and self.metrics.max_drawdown <= max_mdd  # MDD는 양수
            and self.metrics.win_rate >= min_win_rate
        )

    def summary(self) -> dict[str, Any]:
        """요약 정보 반환.

        Returns:
            핵심 지표만 포함된 딕셔너리
        """
        return {
            "strategy": self.config.strategy_name,
            "symbol": self.config.symbol,
            "period": f"{self.config.start_date.date()} ~ {self.config.end_date.date()}",
            "total_return": f"{self.metrics.total_return:.2f}%",
            "cagr": f"{self.metrics.cagr:.2f}%",
            "sharpe_ratio": f"{self.metrics.sharpe_ratio:.2f}",
            "max_drawdown": f"{self.metrics.max_drawdown:.2f}%",
            "win_rate": f"{self.metrics.win_rate:.1f}%",
            "total_trades": self.metrics.total_trades,
            "passed_criteria": self.passed_minimum_criteria(),
        }


# =============================================================================
# Multi-Asset Backtest Models
# =============================================================================


class MultiAssetConfig(BaseModel):
    """멀티에셋 백테스트 설정 기록.

    멀티에셋 포트폴리오 백테스트 시 사용된 설정을 기록합니다.

    Attributes:
        strategy_name: 전략 이름
        symbols: 심볼 목록
        timeframe: 타임프레임
        start_date: 시작일
        end_date: 종료일
        initial_capital: 초기 자본
        asset_weights: 심볼별 배분 비중
        strategy_params: 전략 파라미터
    """

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    symbols: tuple[str, ...] = Field(..., min_length=1, description="심볼 목록")
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal = Field(default=Decimal(100_000))
    asset_weights: dict[str, float] = Field(..., description="심볼별 배분 비중")
    asset_allocation_method: str | None = Field(
        default=None,
        description="동적 에셋 배분 방법 (None=정적)",
    )
    maker_fee: float = Field(default=0.0002, ge=0)
    taker_fee: float = Field(default=0.0004, ge=0)
    slippage: float = Field(default=0.0005, ge=0)
    strategy_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """datetime에 UTC timezone 적용."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class MultiAssetBacktestResult(BaseModel):
    """멀티에셋 백테스트 결과.

    포트폴리오 전체 성과와 심볼별 분해 분석을 포함합니다.

    Attributes:
        config: 백테스트 설정
        portfolio_metrics: 포트폴리오 전체 성과 지표
        per_symbol_metrics: 심볼별 성과 지표
        correlation_matrix: 심볼 간 수익률 상관행렬 (JSON 직렬화 가능)
        contribution: 심볼별 수익 기여도 (%)
        created_at: 결과 생성 시각
    """

    model_config = ConfigDict(frozen=True)

    config: MultiAssetConfig = Field(..., description="백테스트 설정")
    portfolio_metrics: PerformanceMetrics = Field(..., description="포트폴리오 전체 성과")
    per_symbol_metrics: dict[str, PerformanceMetrics] = Field(..., description="심볼별 성과")
    correlation_matrix: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="상관행렬"
    )
    contribution: dict[str, float] = Field(
        default_factory=dict, description="심볼별 수익 기여도 (%)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="결과 생성 시각",
    )

    @property
    def duration_days(self) -> int:
        """백테스트 기간 (일)."""
        return (self.config.end_date - self.config.start_date).days

    def summary(self) -> dict[str, Any]:
        """요약 정보 반환."""
        return {
            "strategy": self.config.strategy_name,
            "symbols": list(self.config.symbols),
            "n_assets": len(self.config.symbols),
            "period": f"{self.config.start_date.date()} ~ {self.config.end_date.date()}",
            "total_return": f"{self.portfolio_metrics.total_return:.2f}%",
            "cagr": f"{self.portfolio_metrics.cagr:.2f}%",
            "sharpe_ratio": f"{self.portfolio_metrics.sharpe_ratio:.2f}",
            "max_drawdown": f"{self.portfolio_metrics.max_drawdown:.2f}%",
            "win_rate": f"{self.portfolio_metrics.win_rate:.1f}%",
        }
