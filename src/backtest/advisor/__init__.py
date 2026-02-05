"""Strategy Advisor module.

전략 빌드업을 위한 코치/가이드 시스템을 제공합니다.

Usage:
    >>> from src.backtest.advisor import StrategyAdvisor
    >>>
    >>> advisor = StrategyAdvisor()
    >>> report = advisor.analyze(
    ...     result=backtest_result,
    ...     returns=strategy_returns,
    ...     benchmark_returns=benchmark_returns,
    ... )
    >>> print(report.summary())
"""

from src.backtest.advisor.advisor import StrategyAdvisor
from src.backtest.advisor.analyzers import (
    LossAnalyzer,
    OverfitAnalyzer,
    RegimeAnalyzer,
    SignalAnalyzer,
)
from src.backtest.advisor.models import (
    AdvisorReport,
    ImprovementSuggestion,
    LossConcentration,
    OverfitScore,
    RegimeProfile,
    SignalQuality,
)
from src.backtest.advisor.suggestions import SuggestionEngine

__all__ = [
    "AdvisorReport",
    "ImprovementSuggestion",
    "LossAnalyzer",
    "LossConcentration",
    "OverfitAnalyzer",
    "OverfitScore",
    "RegimeAnalyzer",
    "RegimeProfile",
    "SignalAnalyzer",
    "SignalQuality",
    "StrategyAdvisor",
    "SuggestionEngine",
]
