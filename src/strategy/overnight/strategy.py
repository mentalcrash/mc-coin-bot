"""Overnight Seasonality Strategy Implementation.

시간대 기반 crypto overnight effect를 포착하는 전략입니다.
BaseStrategy를 상속받아 백테스팅, EDA, 라이브 트레이딩에서 동일하게 사용됩니다.

Rules Applied:
    - #03 Strategy Architecture: BaseStrategy 상속, @register 등록
    - #11 Pydantic Modeling: Config 기반 파라미터 관리
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.overnight.config import OvernightConfig
from src.strategy.overnight.preprocessor import preprocess
from src.strategy.overnight.signal import generate_signals
from src.strategy.registry import register
from src.strategy.tsmom.config import ShortMode

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("overnight")
class OvernightStrategy(BaseStrategy):
    """Overnight Seasonality 전략.

    특정 UTC 시간대(기본 22:00-00:00)에 진입하고 청산하는
    시간 기반 전략입니다. Crypto 시장의 야간 계절성 효과를 포착합니다.

    Key Features:
        - 시간대 기반 진입/청산 (자정 넘김 지원)
        - 변동성 스케일링으로 포지션 크기 조절
        - 선택적 변동성 필터로 고변동성 구간 강화
        - 1H 타임프레임에 최적화

    Attributes:
        _config: Overnight 설정 (OvernightConfig)

    Example:
        >>> from src.strategy.overnight import OvernightStrategy, OvernightConfig
        >>>
        >>> strategy = OvernightStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: OvernightConfig | None = None) -> None:
        """OvernightStrategy 초기화.

        Args:
            config: Overnight 설정. None이면 기본 설정 사용.
        """
        self._config = config or OvernightConfig()

    @classmethod
    def from_params(cls, **params: Any) -> OvernightStrategy:
        """파라미터로 전략 생성 (parameter sweep용).

        Args:
            **params: OvernightConfig 파라미터

        Returns:
            새 OvernightStrategy 인스턴스
        """
        config = OvernightConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Overnight"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> OvernightConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        Args:
            df: OHLCV DataFrame (DatetimeIndex 필수)

        Returns:
            지표가 추가된 DataFrame
        """
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """매매 시그널 생성.

        Args:
            df: 전처리된 DataFrame (preprocess() 출력)

        Returns:
            StrategySignals NamedTuple
        """
        return generate_signals(df, self._config)

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """Overnight 전략에 권장되는 PortfolioManagerConfig 설정.

        - 야간 세션만 포지션 보유하므로 낮은 레버리지
        - Trailing stop 비활성화 (짧은 보유 기간)
        - 낮은 rebalance threshold

        Returns:
            PortfolioManagerConfig 생성에 필요한 키워드 인자 딕셔너리
        """
        return {
            "max_leverage_cap": 1.5,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.05,
            "use_trailing_stop": False,
        }

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 핵심 파라미터.

        Returns:
            파라미터명-값 딕셔너리
        """
        cfg = self._config

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        result = {
            "session": f"{cfg.entry_hour:02d}:00-{cfg.exit_hour:02d}:00 UTC",
            "vol_target": f"{cfg.vol_target:.0%}",
            "vol_window": f"{cfg.vol_window}h",
            "mode": mode_str,
        }

        if cfg.use_vol_filter:
            result["vol_filter"] = (
                f"ratio>{cfg.vol_filter_threshold:.1f} (lookback={cfg.vol_filter_lookback})"
            )

        return result

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self._config.warmup_periods()
