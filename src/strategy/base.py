"""BaseStrategy ABC (Abstract Base Class).

이 모듈은 모든 전략이 구현해야 하는 추상 기반 클래스를 정의합니다.
일관된 인터페이스를 통해 백테스팅, EDA, 실전 트레이딩에서
동일한 전략 코드를 재사용할 수 있습니다.

Rules Applied:
    - #10 Python Standards: Modern typing, ABC pattern
    - #12 Data Engineering: Vectorization protocol
    - #26 VectorBT Standards: Broadcasting compatible output
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


class BaseStrategy(ABC):
    """모든 전략이 구현해야 하는 추상 기반 클래스.

    전략 구현 시 이 클래스를 상속받아 필수 메서드를 구현합니다.
    Stateless 설계 원칙에 따라, 전략은 시장 데이터만 입력받고
    시그널만 출력합니다. 포지션/자산 상태는 관리하지 않습니다.

    Design Principles:
        1. Stateless: 전략 내부에 상태(포지션, 잔고)를 저장하지 않음
        2. Vectorized: 모든 계산은 벡터 연산으로 수행 (for 루프 금지)
        3. Shift(1) Rule: 미래 참조 편향 방지를 위해 전봉 기준 시그널 생성
        4. Reusable: 백테스트/라이브 동일 코드 사용

    Example:
        >>> class MyStrategy(BaseStrategy):
        ...     @property
        ...     def name(self) -> str:
        ...         return "MyStrategy"
        ...
        ...     @property
        ...     def required_columns(self) -> list[str]:
        ...         return ["close", "volume"]
        ...
        ...     def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        ...         df["sma"] = df["close"].rolling(20).mean()
        ...         return df
        ...
        ...     def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        ...         # 시그널 생성 로직
        ...         ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """전략 고유 이름.

        Returns:
            전략 식별자 문자열 (예: "VW-TSMOM", "Breakout")
        """
        ...

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """전략에 필요한 DataFrame 컬럼 목록.

        preprocess() 호출 전 입력 DataFrame에 반드시 존재해야 하는 컬럼들입니다.
        일반적으로 OHLCV 컬럼 (open, high, low, close, volume) 중 필요한 것들.

        Returns:
            필수 컬럼 이름 리스트
        """
        ...

    @property
    def config(self) -> BaseModel | None:
        """전략 설정 (Pydantic 모델).

        전략별 파라미터를 저장하는 Pydantic 설정 모델입니다.
        서브클래스에서 오버라이드하여 구체적인 설정 타입을 반환합니다.

        Returns:
            전략 설정 모델 또는 None
        """
        return None

    @property
    def params(self) -> dict[str, Any]:
        """전략 파라미터 딕셔너리.

        백테스트 결과 저장 시 사용됩니다.
        config가 있으면 model_dump()로 변환, 없으면 빈 딕셔너리.

        Returns:
            파라미터 딕셔너리
        """
        if self.config is not None:
            return self.config.model_dump()
        return {}

    def validate_input(self, df: pd.DataFrame) -> None:
        """입력 DataFrame 유효성 검증.

        Args:
            df: 검증할 DataFrame

        Raises:
            ValueError: 필수 컬럼 누락 또는 데이터 문제 시
        """
        if df.empty:
            msg = "Input DataFrame is empty"
            raise ValueError(msg)

        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            msg = f"Missing required columns: {missing_cols}"
            raise ValueError(msg)

        # DatetimeIndex 확인
        if not isinstance(df.index, pd.DatetimeIndex):
            msg = "DataFrame index must be DatetimeIndex"
            raise TypeError(msg)

        # NaN 검사 (필수 컬럼만)
        for col in self.required_columns:
            col_series: pd.Series = df[col]  # type: ignore[assignment]
            if col_series.isna().all():  # type: ignore[truthy-bool]
                msg = f"Column '{col}' contains all NaN values"
                raise ValueError(msg)

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        원본 OHLCV 데이터에 전략에 필요한 지표를 계산하여 추가합니다.
        모든 계산은 벡터화된 연산을 사용해야 합니다 (for 루프 금지).

        Important:
            - 원본 DataFrame을 수정하지 말고 복사본을 반환할 것
            - 로그 수익률 계산 시 np.log() 사용
            - Rolling 계산 시 min_periods 파라미터 고려

        Args:
            df: OHLCV DataFrame (DatetimeIndex 필수)

        Returns:
            지표가 추가된 새로운 DataFrame
        """
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """매매 시그널 생성.

        전처리된 데이터에서 진입/청산 시그널과 강도를 계산합니다.
        VectorBT 및 QuantStats와 호환되는 표준 출력을 반환합니다.

        Important:
            - Shift(1) Rule: 미래 참조 편향 방지를 위해 전봉 기준으로 시그널 생성
            - entries/exits는 bool Series
            - direction은 -1, 0, 1 값을 가지는 int Series
            - strength는 포지션 사이징에 사용되는 float Series

        Args:
            df: 전처리된 DataFrame (preprocess() 출력)

        Returns:
            StrategySignals NamedTuple (entries, exits, direction, strength)

        Example:
            >>> signals = strategy.generate_signals(df)
            >>> signals.entries  # pd.Series[bool]
            >>> signals.strength  # pd.Series[float]
        """
        ...

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StrategySignals]:
        """전략 실행 (전처리 + 시그널 생성).

        validate → preprocess → generate_signals 파이프라인을 실행합니다.

        Args:
            df: 원본 OHLCV DataFrame

        Returns:
            (전처리된 DataFrame, 시그널) 튜플

        Raises:
            ValueError: 입력 데이터 검증 실패 시
        """
        self.validate_input(df)
        processed_df = self.preprocess(df)
        signals = self.generate_signals(processed_df)
        return processed_df, signals

    def run_incremental(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StrategySignals]:
        """Incremental 모드 전략 실행 (최신 시그널만 효율적으로 계산).

        기본 구현은 run()에 위임합니다. 전략별 최적화가 필요한 경우
        (예: CTREND의 Rolling ElasticNet) 서브클래스에서 오버라이드합니다.

        Args:
            df: 원본 OHLCV DataFrame

        Returns:
            (전처리된 DataFrame, 시그널) 튜플
        """
        return self.run(df)

    @classmethod
    def recommended_config(cls) -> dict[str, Any]:
        """이 전략에 권장되는 PortfolioManagerConfig 설정을 반환합니다.

        서브클래스에서 오버라이드하여 전략별 최적 설정을 제공합니다.
        기본 구현은 빈 딕셔너리를 반환합니다 (PortfolioManagerConfig 기본값 사용).

        NOTE: Portfolio 인스턴스 생성은 CLI 또는 상위 레이어의 책임입니다.
        Strategy는 Portfolio를 직접 생성하지 않고 설정만 제안합니다.

        Returns:
            PortfolioManagerConfig 생성에 필요한 키워드 인자 딕셔너리

        Example:
            >>> config_kwargs = strategy_class.recommended_config()
            >>> from src.portfolio import Portfolio
            >>> portfolio = Portfolio.create(
            ...     initial_capital=Decimal("10000"),
            ...     config=PortfolioManagerConfig(**config_kwargs),
            ... )
        """
        return {}

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        """파라미터 딕셔너리로 전략 인스턴스를 생성합니다.

        run_parameter_sweep 등에서 범용적으로 사용됩니다.
        서브클래스에서 Config 모델을 거쳐야 하는 경우 오버라이드합니다.

        Args:
            **params: 전략 생성 파라미터

        Returns:
            전략 인스턴스
        """
        return cls(**params)  # type: ignore[call-arg]

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 전략 정보를 반환합니다.

        서브클래스에서 오버라이드하여 전략별 핵심 파라미터를 노출합니다.
        기본 구현은 config의 모든 파라미터를 문자열로 변환합니다.

        Returns:
            파라미터명-값 딕셔너리
        """
        if self.config is not None:
            return {k: str(v) for k, v in self.config.model_dump().items()}
        return {}

    def __repr__(self) -> str:
        """전략 문자열 표현."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"
