"""VWAP Disposition Momentum Strategy Implementation.

Rolling VWAP CGO 기반 disposition effect로 매매 시그널 생성.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.vwap_disposition.config import ShortMode, VWAPDispositionConfig
from src.strategy.vwap_disposition.preprocessor import preprocess
from src.strategy.vwap_disposition.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("vwap-disposition")
class VWAPDispositionStrategy(BaseStrategy):
    """VWAP Disposition Momentum Strategy.

    Rolling VWAP를 시장 참여자의 평균 취득가(cost basis)로 사용하여,
    Capital Gains Overhang(CGO)에 따른 disposition effect를 활용합니다.

    Key Features:
        - Behavioral Finance 기반 (disposition effect)
        - Rolling VWAP as cost basis proxy
        - CGO < -threshold → capitulation reversal (LONG)
        - CGO > +threshold → profit-taking pressure (SHORT)
        - Middle zone → momentum direction follow
        - Volume confirmation for extreme zone signals

    Attributes:
        _config: VWAP Disposition 설정 (VWAPDispositionConfig)
    """

    def __init__(self, config: VWAPDispositionConfig | None = None) -> None:
        """VWAPDispositionStrategy 초기화."""
        self._config = config or VWAPDispositionConfig()

    @classmethod
    def from_params(cls, **params: Any) -> VWAPDispositionStrategy:
        """파라미터로 VWAPDispositionStrategy 생성."""
        config = VWAPDispositionConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "VWAP-Disposition"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> VWAPDispositionConfig:
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
        """VWAP Disposition 전략에 권장되는 PortfolioManagerConfig 설정."""
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
            "vwap_window": str(cfg.vwap_window),
            "overhang_high": f"{cfg.overhang_high:.2f}",
            "overhang_low": f"{cfg.overhang_low:.2f}",
            "vol_confirm": str(cfg.use_volume_confirm),
            "mode": mode_str,
        }
