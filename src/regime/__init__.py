"""Regime Detection Module (공유 인프라).

레짐 감지를 위한 유틸리티 모듈입니다.
개별 전략이 preprocess()에서 import하여 레짐 컬럼을 DataFrame에 추가하고,
generate_signals()에서 레짐에 따라 파라미터/포지션 사이징을 조절합니다.

Components:
    - RegimeDetectorConfig: 레짐 감지 설정 (Pydantic 모델)
    - RegimeLabel: 레짐 라벨 (TRENDING, RANGING, VOLATILE)
    - RegimeDetector: 레짐 분류기 (vectorized + incremental)
    - RegimeState: 개별 bar 레짐 상태
    - add_regime_columns: 편의 API (DataFrame에 레짐 컬럼 추가)
    - EnsembleRegimeDetector: 앙상블 레짐 분류기 (Rule + HMM + Vol + MSAR)
    - add_ensemble_regime_columns: 앙상블 편의 API
    - MSARDetectorConfig: MSAR 감지기 설정
    - MetaLearnerConfig: Meta-learner 앙상블 설정

Example:
    >>> from src.regime import add_regime_columns, RegimeDetectorConfig
    >>> df = add_regime_columns(df, RegimeDetectorConfig())
    >>> df["regime_label"]  # "trending", "ranging", "volatile"
"""

from src.regime.config import (
    EnsembleRegimeDetectorConfig,
    HMMDetectorConfig,
    MetaLearnerConfig,
    MSARDetectorConfig,
    RegimeDetectorConfig,
    RegimeLabel,
    VolStructureDetectorConfig,
)
from src.regime.detector import RegimeDetector, RegimeState, add_regime_columns
from src.regime.ensemble import EnsembleRegimeDetector, add_ensemble_regime_columns

__all__ = [
    "EnsembleRegimeDetector",
    "EnsembleRegimeDetectorConfig",
    "HMMDetectorConfig",
    "MSARDetectorConfig",
    "MetaLearnerConfig",
    "RegimeDetector",
    "RegimeDetectorConfig",
    "RegimeLabel",
    "RegimeState",
    "VolStructureDetectorConfig",
    "add_ensemble_regime_columns",
    "add_regime_columns",
]
