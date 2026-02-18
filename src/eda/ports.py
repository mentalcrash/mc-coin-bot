"""EDA Port Protocol 정의.

DataFeed와 Executor의 명시적 인터페이스를 정의합니다.
structural subtyping으로 기존 구현체가 자동으로 만족합니다.

Ports:
    - DataFeedPort: 데이터 피드 인터페이스 (HistoricalDataFeed, LiveDataFeed)
    - ExecutorPort: 주문 실행기 인터페이스 (BacktestExecutor, ShadowExecutor, LiveExecutor)
    - DerivativesProviderPort: 파생상품 데이터 제공 인터페이스
    - MacroProviderPort: 매크로 경제 데이터 제공 인터페이스
    - OptionsProviderPort: 옵션 데이터 제공 인터페이스
    - DerivExtProviderPort: 확장 파생상품 데이터 제공 인터페이스
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    from src.core.event_bus import EventBus
    from src.core.events import FillEvent, OrderRequestEvent


@runtime_checkable
class DataFeedPort(Protocol):
    """데이터 피드 인터페이스.

    HistoricalDataFeed, LiveDataFeed 등이 구현합니다.
    """

    async def start(self, bus: EventBus) -> None:
        """데이터 리플레이/스트리밍을 시작합니다."""
        ...

    async def stop(self) -> None:
        """데이터 피드를 중지합니다."""
        ...

    @property
    def bars_emitted(self) -> int:
        """발행된 총 BarEvent 수."""
        ...


@runtime_checkable
class ExecutorPort(Protocol):
    """주문 실행기 인터페이스.

    BacktestExecutor, ShadowExecutor, LiveExecutor 등이 구현합니다.
    """

    async def execute(self, order: OrderRequestEvent) -> FillEvent | None:
        """주문 실행.

        Args:
            order: 검증된 주문 요청

        Returns:
            체결 결과 (None이면 체결 실패)
        """
        ...


@runtime_checkable
class DerivativesProviderPort(Protocol):
    """파생상품 데이터 제공 인터페이스.

    Backtest: DerivativesDataService 기반 precomputed 데이터
    Live: LiveDerivativesFeed 기반 REST polling 데이터
    """

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """DataFrame에 derivatives 컬럼 추가 (precomputed merge_asof).

        Args:
            df: OHLCV DataFrame
            symbol: 거래 심볼

        Returns:
            derivatives 컬럼이 추가된 DataFrame
        """
        ...

    def get_derivatives_columns(self, symbol: str) -> dict[str, float] | None:
        """최신 캐시된 derivatives 값 반환 (live fallback).

        Args:
            symbol: 거래 심볼

        Returns:
            {column_name: value} 또는 None (데이터 없음)
        """
        ...


@runtime_checkable
class OnchainProviderPort(Protocol):
    """On-chain 데이터 제공 인터페이스.

    Backtest: OnchainDataService 기반 precomputed 데이터
    Live: LiveOnchainFeed 기반 Silver 캐시 데이터
    """

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """DataFrame에 on-chain 컬럼 추가 (precomputed merge_asof).

        Args:
            df: OHLCV DataFrame
            symbol: 거래 심볼

        Returns:
            on-chain 컬럼이 추가된 DataFrame
        """
        ...

    def get_onchain_columns(self, symbol: str) -> dict[str, float] | None:
        """최신 캐시된 on-chain 값 반환 (live fallback).

        Args:
            symbol: 거래 심볼

        Returns:
            {column_name: value} 또는 None (데이터 없음)
        """
        ...


@runtime_checkable
class MacroProviderPort(Protocol):
    """매크로 경제 데이터 제공 인터페이스.

    Backtest: MacroDataService 기반 precomputed 데이터
    Live: LiveMacroFeed 기반 Silver 캐시 데이터
    """

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """DataFrame에 macro 컬럼 추가 (precomputed merge_asof).

        Args:
            df: OHLCV DataFrame
            symbol: 거래 심볼

        Returns:
            macro 컬럼이 추가된 DataFrame
        """
        ...

    def get_macro_columns(self, symbol: str) -> dict[str, float] | None:
        """최신 캐시된 macro 값 반환 (live fallback).

        Args:
            symbol: 거래 심볼

        Returns:
            {column_name: value} 또는 None (데이터 없음)
        """
        ...


@runtime_checkable
class OptionsProviderPort(Protocol):
    """옵션 데이터 제공 인터페이스.

    Backtest: OptionsDataService 기반 precomputed 데이터
    Live: LiveOptionsFeed 기반 Silver 캐시 데이터
    """

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """DataFrame에 options 컬럼 추가 (precomputed merge_asof).

        Args:
            df: OHLCV DataFrame
            symbol: 거래 심볼

        Returns:
            options 컬럼이 추가된 DataFrame
        """
        ...

    def get_options_columns(self, symbol: str) -> dict[str, float] | None:
        """최신 캐시된 options 값 반환 (live fallback).

        Args:
            symbol: 거래 심볼

        Returns:
            {column_name: value} 또는 None (데이터 없음)
        """
        ...


@runtime_checkable
class DerivExtProviderPort(Protocol):
    """확장 파생상품 데이터 제공 인터페이스.

    Backtest: DerivExtDataService 기반 precomputed 데이터
    Live: LiveDerivExtFeed 기반 Silver 캐시 데이터
    """

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """DataFrame에 deriv_ext 컬럼 추가 (precomputed merge_asof).

        Args:
            df: OHLCV DataFrame
            symbol: 거래 심볼

        Returns:
            deriv_ext 컬럼이 추가된 DataFrame
        """
        ...

    def get_deriv_ext_columns(self, symbol: str) -> dict[str, float] | None:
        """최신 캐시된 deriv_ext 값 반환 (live fallback).

        Args:
            symbol: 거래 심볼

        Returns:
            {column_name: value} 또는 None (데이터 없음)
        """
        ...


@runtime_checkable
class FeatureStorePort(Protocol):
    """공통 지표 캐시 서비스 인터페이스.

    Backtest: precompute() → enrich_dataframe() (vectorized, timestamp join)
    Live: register() → _on_bar() → get_feature_columns() (incremental)
    """

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """캐시된 지표 컬럼을 DataFrame에 timestamp join.

        Args:
            df: OHLCV DataFrame
            symbol: 거래 심볼

        Returns:
            지표 컬럼이 추가된 DataFrame
        """
        ...

    def get_feature_columns(self, symbol: str) -> dict[str, float] | None:
        """최신 캐시된 지표 값 반환 (live fallback).

        Args:
            symbol: 거래 심볼

        Returns:
            {column_name: value} 또는 None (데이터 없음)
        """
        ...
