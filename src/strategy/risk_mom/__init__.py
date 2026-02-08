"""Risk-Managed Momentum Strategy.

TSMOM + Barroso-Santa-Clara variance scaling.
"""

from src.strategy.risk_mom.config import RiskMomConfig
from src.strategy.risk_mom.preprocessor import preprocess
from src.strategy.risk_mom.signal import generate_signals
from src.strategy.risk_mom.strategy import RiskMomStrategy

__all__ = [
    "RiskMomConfig",
    "RiskMomStrategy",
    "generate_signals",
    "preprocess",
]
