"""VPIN Flow Toxicity Strategy Implementation.

BVC + VPIN으로 informed trading을 감지하고 flow 방향 추종.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.vpin_flow.config import ShortMode, VPINFlowConfig
from src.strategy.vpin_flow.preprocessor import preprocess
from src.strategy.vpin_flow.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("vpin-flow")
class VPINFlowStrategy(BaseStrategy):
    """VPIN Flow Toxicity Strategy.

    BVC(Bulk Volume Classification)로 buy/sell volume을 근사하고,
    VPIN으로 정보거래 확률을 측정하여 flow 방향을 추종합니다.

    Key Features:
        - BVC로 tick data 없이 buy/sell volume 추정
        - VPIN으로 정보거래 독성(toxicity) 측정
        - 고독성 시: informed trading 방향 추종
        - 저독성 시: 관망 (안정적 시장)

    Attributes:
        _config: VPIN Flow 설정 (VPINFlowConfig)
    """

    def __init__(self, config: VPINFlowConfig | None = None) -> None:
        """VPINFlowStrategy 초기화."""
        self._config = config or VPINFlowConfig()

    @classmethod
    def from_params(cls, **params: Any) -> VPINFlowStrategy:
        """파라미터로 VPINFlowStrategy 생성."""
        config = VPINFlowConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "VPIN-Flow"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> VPINFlowConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산."""
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """매매 시그널 생성."""
        return generate_signals(df, self._config)

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수)."""
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """VPIN Flow 전략에 권장되는 PortfolioManagerConfig 설정."""
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
            "n_buckets": str(cfg.n_buckets),
            "threshold_high": f"{cfg.threshold_high:.2f}",
            "threshold_low": f"{cfg.threshold_low:.2f}",
            "flow_direction_period": f"{cfg.flow_direction_period}d",
            "mode": mode_str,
        }
