"""VW-TSMOM (Volume-Weighted Time Series Momentum) Strategy.

이 모듈은 거래량 가중 시계열 모멘텀 전략을 구현합니다.
학술 연구(SSRN #4825389)에 기반한 검증된 전략입니다.

Components:
    - TSMOMConfig: 전략 설정 (Pydantic 모델)
    - preprocess: 지표 계산 함수 (벡터화)
    - generate_signals: 시그널 생성 함수
    - TSMOMStrategy: 전략 클래스 (BaseStrategy 상속)
    - DiagnosticCollector: 시그널 진단 수집기

Example:
    >>> from src.strategy.tsmom import TSMOMStrategy, TSMOMConfig
    >>> config = TSMOMConfig(lookback=24, vol_target=0.15)
    >>> strategy = TSMOMStrategy(config)
    >>> processed_df, signals = strategy.run(ohlcv_df)
"""

from src.strategy.tsmom.config import TSMOMConfig
from src.strategy.tsmom.diagnostics import (
    DiagnosticCollector,
    collect_diagnostics_from_pipeline,
    log_diagnostic_summary,
    log_signal_diagnostic,
)
from src.strategy.tsmom.preprocessor import preprocess
from src.strategy.tsmom.signal import (
    SignalsWithDiagnostics,
    generate_signals,
    generate_signals_for_long_only,
    generate_signals_with_diagnostics,
    get_current_signal,
)
from src.strategy.tsmom.strategy import TSMOMStrategy

__all__ = [
    "DiagnosticCollector",
    "SignalsWithDiagnostics",
    "TSMOMConfig",
    "TSMOMStrategy",
    "collect_diagnostics_from_pipeline",
    "generate_signals",
    "generate_signals_for_long_only",
    "generate_signals_with_diagnostics",
    "get_current_signal",
    "log_diagnostic_summary",
    "log_signal_diagnostic",
    "preprocess",
]
