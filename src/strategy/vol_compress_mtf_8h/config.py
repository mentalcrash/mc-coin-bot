"""Volatility Compression Breakout + Multi-TF 전략 설정 (8H).

Yang-Zhang 변동성 압축 감지 → Donchian 돌파 + 모멘텀 방향 합의로 진입,
변동성 팽창 시 퇴장. 8H TF에서 12H 모멘텀을 근사하는 Multi-TF 접근.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VolCompressMtf8hConfig(BaseModel):
    """Volatility Compression Breakout + Multi-TF 전략 설정 (8H).

    Yang-Zhang 변동성의 단기/장기 비율(vol_ratio)로 압축 구간을 감지하고,
    Donchian 돌파 + 모멘텀 방향 합의 시 진입한다.
    변동성 팽창(vol_ratio > expansion_threshold) 시 포지션을 청산한다.

    Signal Logic:
        1. vol_ratio = yz_short / yz_long
        2. compressed = vol_ratio < compression_threshold
        3. breakout = close > prev_dc_upper (long) or close < prev_dc_lower (short)
        4. 진입 = compressed & breakout & momentum 방향 합의
        5. 퇴장 = vol_ratio > expansion_threshold (변동성 팽창)

    Attributes:
        yz_short_window: 단기 Yang-Zhang 변동성 윈도우 (8H bars, ~3일).
        yz_long_window: 장기 Yang-Zhang 변동성 윈도우 (8H bars, ~21일).
        compression_threshold: 압축 감지 임계값 (short/long < this).
        expansion_threshold: 팽창 퇴장 임계값 (short/long > this).
        dc_lookback: Donchian channel 돌파 확인 lookback.
        mom_lookback: 모멘텀 방향 lookback (8H bars, ~5일, 12H 10-bar 근사).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 8H TF 연환산 계수 (1095).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    yz_short_window: int = Field(
        default=9,
        ge=3,
        le=30,
        description="Short-term Yang-Zhang vol window (8H bars, ~3 days)",
    )
    yz_long_window: int = Field(
        default=63,
        ge=20,
        le=200,
        description="Long-term Yang-Zhang vol window (8H bars, ~21 days)",
    )
    compression_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="Vol ratio threshold for compression detection (short/long < this = compressed)",
    )
    expansion_threshold: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        description="Vol ratio threshold for expansion exit (short/long > this = expanded)",
    )
    dc_lookback: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Donchian channel lookback for breakout confirmation",
    )
    mom_lookback: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Momentum lookback for direction (8H bars, ~5 days, approximates 12H 10-bar)",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=1095.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> VolCompressMtf8hConfig:
        if self.yz_short_window >= self.yz_long_window:
            msg = (
                f"yz_short_window ({self.yz_short_window}) must be "
                f"< yz_long_window ({self.yz_long_window})"
            )
            raise ValueError(msg)
        if self.compression_threshold >= self.expansion_threshold:
            msg = (
                f"compression_threshold ({self.compression_threshold}) must be "
                f"< expansion_threshold ({self.expansion_threshold})"
            )
            raise ValueError(msg)
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.yz_long_window,
                self.dc_lookback,
                self.mom_lookback,
                self.vol_window,
            )
            + 10
        )
