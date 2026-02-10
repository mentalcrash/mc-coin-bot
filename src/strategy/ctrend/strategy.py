"""CTREND Strategy Implementation.

ML Elastic Net Trend Factor 전략을 구현합니다.
28개 기술적 지표를 Elastic Net으로 결합하여 수익률을 예측합니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization (ML loop은 예외)
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.ctrend.config import CTRENDConfig
from src.strategy.ctrend.preprocessor import preprocess
from src.strategy.ctrend.signal import generate_signals
from src.strategy.registry import register
from src.strategy.tsmom.config import ShortMode

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("ctrend")
class CTRENDStrategy(BaseStrategy):
    """CTREND (ML Elastic Net Trend Factor) Strategy.

    28개 기술적 지표를 Rolling Elastic Net 회귀로 결합하여
    forward return을 예측하고 매매 시그널을 생성하는 전략입니다.

    Key Features:
        - 28 technical features (MACD, RSI, CCI, Stochastic, OBV, etc.)
        - Rolling Elastic Net regression for return prediction
        - Volatility targeting for risk management
        - ShortMode support (DISABLED/HEDGE_ONLY/FULL)

    Attributes:
        _config: CTREND 설정 (CTRENDConfig)

    Example:
        >>> from src.strategy.ctrend import CTRENDStrategy, CTRENDConfig
        >>> config = CTRENDConfig(training_window=252, alpha=0.5)
        >>> strategy = CTRENDStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: CTRENDConfig | None = None) -> None:
        """CTRENDStrategy 초기화.

        Args:
            config: CTREND 설정. None이면 기본 설정 사용.
        """
        self._config = config or CTRENDConfig()

    @classmethod
    def from_params(cls, **params: Any) -> CTRENDStrategy:
        """파라미터로 CTRENDStrategy 생성.

        Args:
            **params: CTRENDConfig 생성 파라미터

        Returns:
            CTRENDStrategy 인스턴스
        """
        config = CTRENDConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "CTREND"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["close", "high", "low", "volume"]

    @property
    def config(self) -> CTRENDConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 28개 feature 계산.

        OHLCV 데이터에 28개 기술적 feature와 변동성 스케일러,
        forward return을 계산하여 추가합니다.

        Calculated Columns:
            - feat_* (28개): 기술적 feature
            - returns: 수익률
            - realized_vol: 실현 변동성
            - vol_scalar: 변동성 스케일러
            - forward_return: 학습 타겟 (미래 수익률)

        Args:
            df: OHLCV DataFrame (DatetimeIndex 필수)

        Returns:
            지표가 추가된 DataFrame
        """
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """매매 시그널 생성 (Rolling Elastic Net).

        Args:
            df: 전처리된 DataFrame (preprocess() 출력)

        Returns:
            StrategySignals NamedTuple
        """
        return generate_signals(df, self._config)

    def run_incremental(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StrategySignals]:
        """Incremental 모드: 최신 시그널만 효율적으로 계산.

        전체 ElasticNet 루프 대신 마지막 2 fits만 수행하여
        EDA bar-by-bar 실행 시 O(n²) → O(n) 으로 최적화합니다.

        Args:
            df: 원본 OHLCV DataFrame

        Returns:
            (전처리된 DataFrame, 시그널) 튜플
        """
        self.validate_input(df)
        processed_df = self.preprocess(df)
        signals = generate_signals(processed_df, self._config, predict_last_only=True)
        return processed_df, signals

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """CTREND 전략에 권장되는 PortfolioManagerConfig 설정.

        - ML 예측 기반이므로 보수적 레버리지
        - 넓은 rebalance threshold (예측 빈도 고려)
        - Trailing stop으로 급락 방어

        Returns:
            PortfolioManagerConfig 생성에 필요한 키워드 인자 딕셔너리
        """
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.10,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 3.0,
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

        return {
            "training_window": f"{cfg.training_window}d",
            "prediction_horizon": f"{cfg.prediction_horizon}d",
            "alpha": f"{cfg.alpha:.2f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_str,
        }
