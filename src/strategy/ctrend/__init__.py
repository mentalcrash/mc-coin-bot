"""CTREND (ML Elastic Net Trend Factor) Strategy.

28개 기술적 지표를 Elastic Net 회귀로 결합하여 수익률을 예측하는 전략입니다.

Components:
    - CTRENDConfig: 전략 설정 (Pydantic 모델)
    - preprocess: 지표 계산 함수 (28 features)
    - generate_signals: Rolling Elastic Net 시그널 생성
    - CTRENDStrategy: 전략 클래스 (BaseStrategy 상속)

Example:
    >>> from src.strategy.ctrend import CTRENDStrategy, CTRENDConfig
    >>> config = CTRENDConfig(training_window=252, alpha=0.5)
    >>> strategy = CTRENDStrategy(config)
    >>> processed_df, signals = strategy.run(ohlcv_df)
"""

from src.strategy.ctrend.config import CTRENDConfig
from src.strategy.ctrend.preprocessor import preprocess
from src.strategy.ctrend.signal import generate_signals
from src.strategy.ctrend.strategy import CTRENDStrategy

__all__ = [
    "CTRENDConfig",
    "CTRENDStrategy",
    "generate_signals",
    "preprocess",
]
