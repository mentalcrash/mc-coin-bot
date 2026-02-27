"""Wavelet-Channel 8H 전략 설정.

DWT(Discrete Wavelet Transform) denoised close에 3종 채널(Donchian/Keltner/BB) x
3스케일(22/66/132) 앙상블 breakout. 웨이블릿 denoising으로 노이즈 제거 후
채널 계산 → 로버스트 추세 감지.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class WaveletChannel8hConfig(BaseModel):
    """Wavelet-Channel 8H 전략 설정.

    DWT denoised close에 3종 채널(DC/KC/BB) x 3스케일(22/66/132) breakout 앙상블.
    9개 sub-signal의 평균(consensus) -> sign -> vol scaling.

    Signal Logic:
        1. close를 DWT denoising (approximation coefficients만 유지)
        2. denoised close로 3채널 x 3스케일 = 9 채널 계산
        3. raw close vs prev(denoised channel)로 breakout sub-signal 9개
        4. consensus = mean(signal_1, ..., signal_9)
        5. direction = sign(consensus) if |consensus| >= entry_threshold else 0
        6. strength = |consensus| * vol_scalar

    Attributes:
        wavelet_family: 웨이블릿 종류 (pywt 호환, e.g. 'db4', 'haar').
        wavelet_level: DWT 분해 레벨 (detail coefficients 제거 수).
        scale_short: 단기 스케일 (8H bars, ~3일).
        scale_mid: 중기 스케일 (8H bars, ~22일).
        scale_long: 장기 스케일 (8H bars, ~44일).
        bb_std: Bollinger Bands 표준편차 배수.
        kc_mult: Keltner Channels ATR 배수.
        entry_threshold: consensus 진입 임계값 (0~1).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 8H TF 연환산 계수 (365*3=1095).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Wavelet Parameters ---
    wavelet_family: str = Field(default="db4", description="웨이블릿 종류 (pywt 호환)")
    wavelet_level: int = Field(default=2, ge=1, le=6, description="DWT 분해 레벨")

    # --- Scale Parameters (8H bars) ---
    scale_short: int = Field(default=22, ge=5, le=100, description="단기 스케일 (~3일)")
    scale_mid: int = Field(default=66, ge=10, le=300, description="중기 스케일 (~22일)")
    scale_long: int = Field(default=132, ge=20, le=500, description="장기 스케일 (~44일)")

    # --- Channel Parameters ---
    bb_std: float = Field(default=3.5, ge=0.5, le=5.0)
    kc_mult: float = Field(default=2.04, ge=0.5, le=5.0)
    entry_threshold: float = Field(
        default=0.34,
        ge=0.0,
        le=1.0,
        description="consensus 진입 임계값. 0.34 = 9개 중 ~3개 합의로 진입",
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
    def _validate_cross_fields(self) -> WaveletChannel8hConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if not (self.scale_short < self.scale_mid < self.scale_long):
            msg = (
                f"scale_short ({self.scale_short}) < scale_mid ({self.scale_mid}) "
                f"< scale_long ({self.scale_long}) 필수"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수.

        scale_long + wavelet padding (2^level) + 10.
        """
        wavelet_padding = 2**self.wavelet_level
        return max(self.scale_long, self.vol_window) + wavelet_padding + 10
