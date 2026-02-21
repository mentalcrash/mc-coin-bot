"""DEX Activity Momentum Strategy.

DEX 거래량 7D/30D 변화율 기반 on-chain 활동 모멘텀.
stablecoin 무관 독립 데이터 소스.

Components:
    - DexMomConfig: Pydantic frozen config
    - preprocess: DEX volume ROC 계산
    - generate_signals: 7D/30D ROC 방향 기반 시그널
    - DexMomStrategy: @register("dex-mom")
"""

from src.strategy.dex_mom.config import DexMomConfig
from src.strategy.dex_mom.preprocessor import preprocess
from src.strategy.dex_mom.signal import generate_signals
from src.strategy.dex_mom.strategy import DexMomStrategy

__all__ = [
    "DexMomConfig",
    "DexMomStrategy",
    "generate_signals",
    "preprocess",
]
