"""Flow Imbalance Strategy Implementation.

BVC + OFI + VPIN → flow-driven signals (1H).

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.flow_imbalance.config import FlowImbalanceConfig, ShortMode
from src.strategy.flow_imbalance.preprocessor import preprocess
from src.strategy.flow_imbalance.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("flow-imbalance")
class FlowImbalanceStrategy(BaseStrategy):
    """Flow Imbalance Strategy.

    BVC(Bulk Volume Classification)로 매수/매도 볼륨을 분류하고,
    OFI와 VPIN으로 주문 흐름 불균형을 감지하는 1H 전략입니다.

    Key Features:
        - BVC로 bar 내 매수/매도 볼륨 추정
        - OFI로 주문 흐름 방향 감지
        - VPIN proxy로 정보비대칭 활동 확인
        - Timeout으로 장기 포지션 방지

    Attributes:
        _config: Flow Imbalance 설정 (FlowImbalanceConfig)
    """

    def __init__(self, config: FlowImbalanceConfig | None = None) -> None:
        """FlowImbalanceStrategy 초기화."""
        self._config = config or FlowImbalanceConfig()

    @classmethod
    def from_params(cls, **params: Any) -> FlowImbalanceStrategy:
        """파라미터로 FlowImbalanceStrategy 생성."""
        config = FlowImbalanceConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Flow-Imbalance"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> FlowImbalanceConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산."""
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """매매 시그널 생성."""
        return generate_signals(df, self._config)

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (1H bars)."""
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """Flow Imbalance 전략에 권장되는 PortfolioManagerConfig 설정."""
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.10,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 3.0,
        }

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 핵심 파라미터."""
        cfg = self._config

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "ofi_window": f"{cfg.ofi_window}h",
            "ofi_entry": f"{cfg.ofi_entry_threshold:.2f}",
            "vpin_threshold": f"{cfg.vpin_threshold:.2f}",
            "timeout": f"{cfg.timeout_bars}h",
            "mode": mode_str,
        }
