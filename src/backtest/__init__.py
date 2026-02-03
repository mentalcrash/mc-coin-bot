"""Backtesting module for strategy evaluation.

이 모듈은 VectorBT 기반 백테스팅 인프라를 제공합니다.
Clean Architecture 원칙에 따라 설계되었습니다.

Core Components:
    - BacktestEngine: Stateless 백테스트 실행자
    - BacktestRequest: 실행 요청 DTO (Command Pattern)
    - PerformanceAnalyzer: 성과 분석 전담

Example:
    >>> from src.backtest import BacktestEngine, BacktestRequest, PerformanceAnalyzer
    >>> from src.data import MarketDataService, MarketDataRequest
    >>> from src.portfolio import Portfolio
    >>> from src.strategy.tsmom import TSMOMStrategy
    >>>
    >>> # 데이터 로드
    >>> data = MarketDataService().get(MarketDataRequest(...))
    >>>
    >>> # 백테스트 요청 생성
    >>> request = BacktestRequest(
    ...     data=data,
    ...     strategy=TSMOMStrategy(),
    ...     portfolio=Portfolio.create(initial_capital=10000),
    ... )
    >>>
    >>> # 실행
    >>> result = BacktestEngine().run(request)
    >>> print(result.metrics.sharpe_ratio)
"""

from src.backtest.analyzer import PerformanceAnalyzer
from src.backtest.cost_model import CostModel
from src.backtest.engine import BacktestEngine, run_parameter_sweep
from src.backtest.metrics import (
    calculate_all_metrics,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_drawdown_series,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_rolling_sharpe,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
    calculate_volatility,
    calculate_win_rate,
)
from src.backtest.reporter import (
    generate_quantstats_report,
    generate_report_from_backtest_result,
    print_performance_summary,
)
from src.backtest.request import BacktestRequest
from src.portfolio import Portfolio, PortfolioManagerConfig

__all__ = [
    "BacktestEngine",
    "BacktestRequest",
    "CostModel",
    "PerformanceAnalyzer",
    "Portfolio",
    "PortfolioManagerConfig",
    "calculate_all_metrics",
    "calculate_cagr",
    "calculate_calmar_ratio",
    "calculate_drawdown_series",
    "calculate_max_drawdown",
    "calculate_profit_factor",
    "calculate_rolling_sharpe",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_total_return",
    "calculate_volatility",
    "calculate_win_rate",
    "generate_quantstats_report",
    "generate_report_from_backtest_result",
    "print_performance_summary",
    "run_parameter_sweep",
]
