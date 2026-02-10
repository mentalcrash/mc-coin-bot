"""Range Compression Breakout Strategy Implementation.

NR 패턴 + range ratio squeeze 후 breakout 방향 추종.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.range_squeeze.config import RangeSqueezeConfig, ShortMode
from src.strategy.range_squeeze.preprocessor import preprocess
from src.strategy.range_squeeze.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("range-squeeze")
class RangeSqueezeStrategy(BaseStrategy):
    """Range Compression Breakout Strategy.

    NR7 패턴(N일 중 최소 range)과 range ratio를 사용하여
    vol compression을 감지하고 breakout 방향을 추종하는 전략입니다.

    Key Features:
        - NR 패턴으로 range compression 감지
        - Range ratio로 상대적 squeeze 수준 측정
        - Squeeze 해소 시 breakout 방향 추종
        - Vol-target 기반 포지션 사이징

    Attributes:
        _config: Range Squeeze 설정 (RangeSqueezeConfig)
    """

    def __init__(self, config: RangeSqueezeConfig | None = None) -> None:
        """RangeSqueezeStrategy 초기화."""
        self._config = config or RangeSqueezeConfig()

    @classmethod
    def from_params(cls, **params: Any) -> RangeSqueezeStrategy:
        """파라미터로 RangeSqueezeStrategy 생성."""
        config = RangeSqueezeConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Range-Squeeze"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> RangeSqueezeConfig:
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
        """Range Squeeze 전략에 권장되는 PortfolioManagerConfig 설정."""
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
            "nr_period": f"{cfg.nr_period}d",
            "lookback": f"{cfg.lookback}d",
            "squeeze_threshold": f"{cfg.squeeze_threshold:.2f}",
            "mode": mode_str,
        }
