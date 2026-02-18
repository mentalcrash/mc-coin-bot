"""Backtest Request DTO (Command Pattern).

이 모듈은 백테스트 실행 요청을 캡슐화하는 DTO를 정의합니다.
Command Pattern을 적용하여 실행에 필요한 모든 정보를 하나의 객체로 묶습니다.

Design Principles:
    - Immutable: 요청 객체는 생성 후 변경 불가
    - Self-contained: 실행에 필요한 모든 정보 포함
    - Decoupled: Engine은 요청 객체만 받아서 처리

Rules Applied:
    - #10 Python Standards: Modern typing, dataclass
    - #11 Pydantic Modeling: 검증 가능한 DTO
"""

from dataclasses import dataclass

from src.backtest.analyzer import PerformanceAnalyzer
from src.data.market_data import MarketDataSet, MultiSymbolData
from src.orchestrator.asset_allocator import AssetAllocationConfig
from src.portfolio.portfolio import Portfolio
from src.strategy.base import BaseStrategy


@dataclass(frozen=True)
class BacktestRequest:
    """백테스트 실행 요청 (Command Pattern).

    백테스트 실행에 필요한 모든 정보를 캡슐화합니다.
    불변 객체로 설계되어 실행 도중 변경되지 않습니다.

    Attributes:
        data: 시장 데이터셋 (메타데이터 + OHLCV)
        strategy: 전략 인스턴스
        portfolio: 포트폴리오 (초기 자본 + 설정)
        analyzer: 성과 분석기 (선택, None이면 기본값 사용)

    Example:
        >>> request = BacktestRequest(
        ...     data=market_data,
        ...     strategy=TSMOMStrategy(),
        ...     portfolio=Portfolio.create(initial_capital=10000),
        ...     analyzer=PerformanceAnalyzer(),
        ... )
        >>> result = BacktestEngine().run(request)
    """

    data: MarketDataSet
    strategy: BaseStrategy
    portfolio: Portfolio
    analyzer: PerformanceAnalyzer | None = None

    def __repr__(self) -> str:
        """문자열 표현."""
        return (
            f"BacktestRequest("
            f"symbol={self.data.symbol!r}, "
            f"timeframe={self.data.timeframe!r}, "
            f"strategy={self.strategy.name!r}, "
            f"capital=${self.portfolio.initial_capital:,.0f})"
        )


@dataclass(frozen=True)
class MultiAssetBacktestRequest:
    """멀티에셋 백테스트 실행 요청.

    여러 심볼을 하나의 포트폴리오로 백테스트하기 위한 요청 객체입니다.
    모든 심볼에 동일한 전략을 독립적으로 적용합니다.

    Attributes:
        data: 멀티 심볼 데이터
        strategy: 전략 인스턴스 (모든 심볼에 동일 적용)
        portfolio: 포트폴리오 (초기 자본 + PM 설정)
        weights: 심볼별 배분 비중 (None이면 Equal Weight 1/N)
        analyzer: 성과 분석기 (선택)

    Example:
        >>> request = MultiAssetBacktestRequest(
        ...     data=multi_data,
        ...     strategy=TSMOMStrategy(),
        ...     portfolio=Portfolio.create(initial_capital=100000),
        ... )
        >>> result = BacktestEngine().run_multi(request)
    """

    data: MultiSymbolData
    strategy: BaseStrategy
    portfolio: Portfolio
    weights: dict[str, float] | None = None
    asset_allocation: AssetAllocationConfig | None = None
    analyzer: PerformanceAnalyzer | None = None

    def __post_init__(self) -> None:
        """weights와 asset_allocation 상호 배타 검증."""
        if self.weights is not None and self.asset_allocation is not None:
            msg = "weights and asset_allocation are mutually exclusive"
            raise ValueError(msg)

    @property
    def asset_weights(self) -> dict[str, float]:
        """실제 자산 배분 비중 (EW fallback 포함)."""
        if self.weights is not None:
            return self.weights
        weight = 1.0 / self.data.n_assets
        return dict.fromkeys(self.data.symbols, weight)

    def __repr__(self) -> str:
        """문자열 표현."""
        return (
            f"MultiAssetBacktestRequest("
            f"symbols={self.data.symbols}, "
            f"timeframe={self.data.timeframe!r}, "
            f"strategy={self.strategy.name!r}, "
            f"capital=${self.portfolio.initial_capital:,.0f})"
        )
