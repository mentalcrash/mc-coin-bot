"""Variance Ratio Regime Strategy Implementation.

Lo-MacKinlay VR test로 random walk hypothesis 검정 후 regime별 전략 적용.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.vr_regime.config import ShortMode, VRRegimeConfig
from src.strategy.vr_regime.preprocessor import preprocess
from src.strategy.vr_regime.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("vr-regime")
class VRRegimeStrategy(BaseStrategy):
    """Variance Ratio Regime Strategy.

    Lo-MacKinlay Variance Ratio test로 random walk hypothesis를 검정합니다.
    VR > 1 → trending (momentum), VR < 1 → mean-reverting (contrarian).

    Key Features:
        - Non-parametric random walk test 기반
        - Heteroscedastic-robust z-stat (Lo-MacKinlay)
        - Trending regime: momentum 추종
        - Mean-reverting regime: contrarian
        - Random walk: 관망

    Attributes:
        _config: VR Regime 설정 (VRRegimeConfig)
    """

    def __init__(self, config: VRRegimeConfig | None = None) -> None:
        """VRRegimeStrategy 초기화."""
        self._config = config or VRRegimeConfig()

    @classmethod
    def from_params(cls, **params: Any) -> VRRegimeStrategy:
        """파라미터로 VRRegimeStrategy 생성."""
        config = VRRegimeConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "VR-Regime"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> VRRegimeConfig:
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
        """VR Regime 전략에 권장되는 PortfolioManagerConfig 설정."""
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
            "vr_window": f"{cfg.vr_window}d",
            "vr_k": str(cfg.vr_k),
            "significance_z": f"{cfg.significance_z:.2f}",
            "heteroscedastic": str(cfg.use_heteroscedastic),
            "mode": mode_str,
        }
