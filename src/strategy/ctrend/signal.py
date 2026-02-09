"""CTREND Signal Generator (Rolling Elastic Net Prediction).

이 모듈은 28개 기술적 feature를 Rolling Elastic Net으로 결합하여
forward return을 예측하고 매매 시그널을 생성합니다.

Signal Generation Pipeline:
    1. Rolling window에서 feature X와 target y (forward_return) 추출
    2. ElasticNet으로 학습
    3. 현재 feature로 predicted return 계산
    4. signal = sign(predicted_return) * vol_scalar
    5. Shift(1) 적용 (미래 참조 편향 방지)

Rules Applied:
    - #12 Data Engineering: ML training loop은 벡터화 예외
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from src.strategy.ctrend.config import CTRENDConfig
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction, StrategySignals

logger = logging.getLogger(__name__)

# Elastic Net 학습에 필요한 최소 유효 샘플 수
_MIN_TRAIN_SAMPLES = 30


def generate_signals(
    df: pd.DataFrame,
    config: CTRENDConfig | None = None,
) -> StrategySignals:
    """CTREND 시그널 생성 (Rolling Elastic Net).

    Rolling window 방식으로 Elastic Net을 학습하여 forward return을 예측하고,
    예측 결과를 기반으로 매매 시그널을 생성합니다.

    For each bar t >= training_window:
        1. Extract features X[t-training_window:t] and forward_returns y[t-training_window:t]
        2. Filter out rows where y is NaN (last prediction_horizon rows of training window)
        3. Fit ElasticNet(l1_ratio=config.alpha) on X, y
        4. Predict using current features X[t] -> predicted_return
        5. Signal = sign(predicted_return) * vol_scalar
        6. Apply shift(1)

    NOTE: Rolling ML training requires loop -- cannot vectorize

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: feat_* (28개), forward_return, vol_scalar
        config: CTREND 설정. None이면 기본 설정 사용.

    Returns:
        StrategySignals NamedTuple:
            - entries: 진입 시그널 (bool Series)
            - exits: 청산 시그널 (bool Series)
            - direction: 방향 시리즈 (-1, 0, 1)
            - strength: 시그널 강도 (vol scaling 적용)

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    if config is None:
        config = CTRENDConfig()

    # Feature 컬럼 추출
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    if not feat_cols:
        msg = "No feature columns (feat_*) found. Run preprocess() first."
        raise ValueError(msg)

    required_cols = {"forward_return", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 데이터 추출
    n = len(df)
    training_window = config.training_window
    vol_scalar_series: pd.Series = df["vol_scalar"]  # type: ignore[assignment]
    forward_return_series: pd.Series = df["forward_return"]  # type: ignore[assignment]

    # Feature matrix (원본 numpy 배열로 변환)
    feature_matrix = df[feat_cols].to_numpy()  # shape: (n, 28)
    forward_returns = forward_return_series.to_numpy()

    # Prediction 결과 배열 (pre-allocate)
    predictions = np.full(n, np.nan)

    # NOTE: Rolling ML training requires loop -- cannot vectorize
    for t in range(training_window, n):
        # Training window: [t - training_window, t)
        start_idx = t - training_window

        x_train = feature_matrix[start_idx:t]
        y_train = forward_returns[start_idx:t]

        # NaN이 아닌 행만 사용 (forward_return은 마지막 prediction_horizon개가 NaN)
        valid_mask = ~(np.isnan(y_train) | np.any(np.isnan(x_train), axis=1))

        if valid_mask.sum() < _MIN_TRAIN_SAMPLES:
            continue

        x_valid = x_train[valid_mask]
        y_valid = y_train[valid_mask]

        # Elastic Net 학습
        model = ElasticNet(
            l1_ratio=config.alpha,
            alpha=0.01,  # regularization strength
            max_iter=1000,
            random_state=42,
        )

        try:
            model.fit(x_valid, y_valid)
        except Exception:
            logger.debug("ElasticNet fit failed at index %d", t)
            continue

        # 현재 feature로 예측
        x_current = feature_matrix[t : t + 1]
        if np.any(np.isnan(x_current)):
            continue

        pred = model.predict(x_current)[0]
        predictions[t] = pred

    # Prediction을 Series로 변환
    prediction_series = pd.Series(predictions, index=df.index)

    # Signal 계산: sign(predicted_return) * vol_scalar
    pred_direction = np.sign(prediction_series)
    scaled_signal: pd.Series = pred_direction * vol_scalar_series  # type: ignore[assignment]

    # Shift(1) 적용: 미래 참조 편향 방지
    signal_shifted: pd.Series = scaled_signal.shift(1)  # type: ignore[assignment]

    # Direction 계산
    direction_raw = pd.Series(np.sign(signal_shifted), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # Strength 계산
    strength = pd.Series(
        signal_shifted.fillna(0),
        index=df.index,
        name="strength",
    )

    # Short mode 처리
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)
    elif config.short_mode == ShortMode.HEDGE_ONLY:
        # HEDGE_ONLY는 drawdown 없이는 완전히 구현 불가 → DISABLED처럼 처리
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)
    # ShortMode.FULL: 모든 시그널 유지

    # Entry/Exit 시그널 생성
    prev_direction = direction.shift(1).fillna(0)

    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)

    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
    reversal = direction * prev_direction < 0

    exits = pd.Series(
        to_neutral | reversal,
        index=df.index,
        name="exits",
    )

    # 시그널 통계 로깅
    valid_strength = strength[strength != 0]
    long_count = int((strength > 0).sum())
    short_count = int((strength < 0).sum())

    if len(valid_strength) > 0:
        logger.info(
            "CTREND Signals | Total: %d, Long: %d (%.1f%%), Short: %d (%.1f%%)",
            len(valid_strength),
            long_count,
            long_count / len(valid_strength) * 100,
            short_count,
            short_count / len(valid_strength) * 100,
        )

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )
