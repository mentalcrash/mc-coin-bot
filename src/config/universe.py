"""Asset Universe — Tier 중앙 상수 정의.

모든 에셋 심볼을 한 곳에서 관리하여 CLI/백테스트/라이브에서 참조합니다.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tier 1: 기존 핵심 에셋 (높은 유동성, 전체 Derivatives 지원)
# ---------------------------------------------------------------------------
TIER1_SYMBOLS: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "ADA/USDT",
    "AVAX/USDT",
)

# ---------------------------------------------------------------------------
# Tier 2: 확장 에셋 (Funding Rate만 수집)
# ---------------------------------------------------------------------------
TIER2_SYMBOLS: tuple[str, ...] = (
    "XRP/USDT",
    "DOT/USDT",
    "POL/USDT",
    "UNI/USDT",
    "NEAR/USDT",
    "ATOM/USDT",
    "FIL/USDT",
    "LTC/USDT",
)

ALL_SYMBOLS: tuple[str, ...] = TIER1_SYMBOLS + TIER2_SYMBOLS

# ---------------------------------------------------------------------------
# MATIC → POL 심볼 전환 정보 (2024-09-10 Binance 기준)
# ---------------------------------------------------------------------------
SYMBOL_RENAME_MAP: dict[str, tuple[str, str]] = {
    "POL/USDT": ("MATIC/USDT", "2024-09-10"),  # (이전 심볼, 전환일)
}

# ---------------------------------------------------------------------------
# 연도 범위 상수
# ---------------------------------------------------------------------------
TIER1_OHLCV_START_YEAR = 2017
TIER2_OHLCV_START_YEAR = 2020
CURRENT_YEAR = 2026
DERIV_START_YEAR = 2020
