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
from src.data.market_data import MarketDataSet
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
