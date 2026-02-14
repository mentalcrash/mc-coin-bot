"""Ensemble Strategy Implementation.

여러 서브 전략의 시그널을 집계하여 앙상블 포지션을 결정하는 메타 전략.
EnsembleStrategy IS-A BaseStrategy → 기존 인프라(BacktestEngine, EDARunner, CLI) 변경 ZERO.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.strategy.base import BaseStrategy
from src.strategy.ensemble.config import EnsembleConfig, ShortMode, SubStrategySpec
from src.strategy.ensemble.preprocessor import preprocess
from src.strategy.ensemble.signal import generate_signals
from src.strategy.registry import get_strategy, register

if TYPE_CHECKING:
    from src.strategy.types import StrategySignals

logger = logging.getLogger(__name__)


@register("ensemble")
class EnsembleStrategy(BaseStrategy):
    """앙상블 메타 전략.

    여러 서브 전략의 시그널을 수집하고 설정된 aggregation 방법으로
    결합하여 하나의 시그널을 생성합니다.

    Key Features:
        - IS-A BaseStrategy: 기존 인프라와 완전 호환
        - 4가지 aggregation: EW, Inverse-Vol, Majority-Vote, Strategy-Momentum
        - Sub-strategy 실패 허용: 최소 1개 성공 시 정상 동작
        - Warmup 동기화: max(sub warmup) + aggregation lookback

    Attributes:
        _config: 앙상블 설정
        _sub_strategies: 서브 전략 인스턴스 목록
        _strategy_names: 서브 전략 이름 목록
        _weights: 정적 가중치 Series

    Example:
        >>> from src.strategy.ensemble import EnsembleStrategy, EnsembleConfig
        >>> config = EnsembleConfig(
        ...     strategies=(
        ...         SubStrategySpec(name="tsmom"),
        ...         SubStrategySpec(name="donchian-ensemble"),
        ...     ),
        ... )
        >>> strategy = EnsembleStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        """EnsembleStrategy 초기화.

        Args:
            config: 앙상블 설정. None이면 ValueError (최소 2개 전략 필수).

        Raises:
            ValueError: config가 None인 경우
        """
        if config is None:
            msg = "EnsembleConfig is required (at least 2 sub-strategies)"
            raise ValueError(msg)

        self._config = config
        self._sub_strategies: list[BaseStrategy] = []
        self._strategy_names: list[str] = []
        self._weights = pd.Series(dtype=float)

        self._init_sub_strategies()

    def _init_sub_strategies(self) -> None:
        """Registry에서 서브 전략 인스턴스를 생성."""
        names: list[str] = []
        strategies: list[BaseStrategy] = []
        weight_dict: dict[str, float] = {}

        for spec in self._config.strategies:
            strategy_cls = get_strategy(spec.name)
            strategy = strategy_cls.from_params(**spec.params) if spec.params else strategy_cls()

            names.append(spec.name)
            strategies.append(strategy)
            weight_dict[spec.name] = spec.weight

        self._strategy_names = names
        self._sub_strategies = strategies
        self._weights = pd.Series(weight_dict)

        logger.info(
            "Ensemble initialized | strategies=%s, aggregation=%s",
            names,
            self._config.aggregation.value,
        )

    @classmethod
    def from_params(cls, **params: Any) -> EnsembleStrategy:
        """파라미터로 EnsembleStrategy 생성.

        YAML의 sub_strategies 리스트를 SubStrategySpec 튜플로 변환합니다.

        Args:
            **params: 앙상블 설정 파라미터
                - strategies: list[dict] 형태의 서브 전략 목록

        Returns:
            EnsembleStrategy 인스턴스
        """
        raw_strategies = params.pop("strategies", [])
        specs = tuple(SubStrategySpec(**s) for s in raw_strategies)
        config = EnsembleConfig(strategies=specs, **params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Ensemble"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록 (최소 close)."""
        return ["close"]

    @property
    def config(self) -> EnsembleConfig:
        """앙상블 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 (vol_scalar 계산).

        Args:
            df: OHLCV DataFrame

        Returns:
            vol_scalar가 추가된 DataFrame
        """
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """앙상블 시그널 생성.

        Args:
            df: 전처리된 DataFrame (vol_scalar 포함)

        Returns:
            StrategySignals NamedTuple
        """
        return generate_signals(
            df,
            self._sub_strategies,
            self._strategy_names,
            self._weights,
            self._config,
        )

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간.

        max(sub-strategy warmup) + aggregation lookback.

        Returns:
            필요한 캔들 수
        """
        sub_warmups = [
            strategy.warmup_periods()  # type: ignore[attr-defined]
            for strategy in self._sub_strategies
            if hasattr(strategy, "warmup_periods")
        ]

        max_sub_warmup = max(sub_warmups) if sub_warmups else 0
        return max_sub_warmup + self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """앙상블 전략에 권장되는 PortfolioManagerConfig 설정."""
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.05,
        }

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 핵심 파라미터."""
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "strategies": ", ".join(self._strategy_names),
            "num_strategies": str(len(self._sub_strategies)),
            "aggregation": cfg.aggregation.value,
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
