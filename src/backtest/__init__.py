"""Backtesting module for strategy evaluation.

이 모듈은 VectorBT 기반 백테스팅 인프라를 제공합니다.
전략 성과 평가, 비용 모델링, 성과 지표 계산을 담당합니다.

Example:
    >>> from src.backtest import BacktestEngine, PortfolioManagerConfig
    >>> from src.strategy.tsmom import TSMOMStrategy
    >>>
    >>> # 기본 설정 사용
    >>> engine = BacktestEngine(initial_capital=10000)
    >>> result = engine.run(strategy=TSMOMStrategy(), data=ohlcv_df)
    >>>
    >>> # 보수적 설정 사용
    >>> engine = BacktestEngine(
    ...     portfolio_config=PortfolioManagerConfig.conservative(),
    ...     initial_capital=10000,
    ... )
    >>> result = engine.run(strategy=TSMOMStrategy(), data=ohlcv_df)
    >>> print(result.metrics.sharpe_ratio)
"""

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
from src.portfolio.config import PortfolioManagerConfig

__all__ = [
    "BacktestEngine",
    "CostModel",
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
