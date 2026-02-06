"""VW-TSMOM (Volume-Weighted Time Series Momentum) Strategy.

이 모듈은 거래량 가중 시계열 모멘텀 전략을 구현합니다.
학술 연구(SSRN #4825389)에 기반한 검증된 전략입니다.

Pure TSMOM + Vol Target 구현:
    1. vw_momentum = 거래량 가중 수익률 (lookback 기간)
    2. vol_scalar = vol_target / realized_vol
    3. direction = sign(vw_momentum)
    4. strength = direction * vol_scalar

Components:
    - TSMOMConfig: 전략 설정 (Pydantic 모델)
    - preprocess: 지표 계산 함수 (벡터화)
    - generate_signals: 시그널 생성 함수
    - TSMOMStrategy: 전략 클래스 (BaseStrategy 상속)

Example:
    >>> from src.strategy.tsmom import TSMOMStrategy, TSMOMConfig
    >>> config = TSMOMConfig(lookback=30, vol_target=0.40)
    >>> strategy = TSMOMStrategy(config)
    >>> processed_df, signals = strategy.run(ohlcv_df)
"""

from src.strategy.tsmom.config import MTFFilterConfig, MTFFilterMode, ShortMode, TSMOMConfig
from src.strategy.tsmom.diagnostics import (
    collect_diagnostics_from_signals,
    log_diagnostic_summary,
)
from src.strategy.tsmom.mtf_filter import (
    align_htf_to_ltf,
    apply_mtf_filter,
    compute_htf_trend,
)
from src.strategy.tsmom.preprocessor import preprocess
from src.strategy.tsmom.signal import (
    SignalsWithDiagnostics,
    generate_signals,
    generate_signals_with_diagnostics,
    get_current_signal,
)
from src.strategy.tsmom.strategy import TSMOMStrategy

__all__ = [
    "MTFFilterConfig",
    "MTFFilterMode",
    "ShortMode",
    "SignalsWithDiagnostics",
    "TSMOMConfig",
    "TSMOMStrategy",
    "align_htf_to_ltf",
    "apply_mtf_filter",
    "collect_diagnostics_from_signals",
    "compute_htf_trend",
    "generate_signals",
    "generate_signals_with_diagnostics",
    "get_current_signal",
    "log_diagnostic_summary",
    "preprocess",
]
