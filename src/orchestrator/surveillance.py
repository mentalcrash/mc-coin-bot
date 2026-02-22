"""MarketSurveillanceService — 동적 유니버스 관리.

주기적(주간) 시장 스캔으로 유동성/거래량 기준 에셋을 자동 선별합니다.
전역 서비스 1개가 유니버스를 관리하고, Orchestrator가 각 Pod에 전달합니다.

Rules Applied:
    - #10 Python Standards: Modern typing, named constants
    - #11 Pydantic Modeling: frozen=True, Field validators
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.exchange.binance_client import BinanceClient

# ── Constants ─────────────────────────────────────────────────────

_DEFAULT_SCAN_INTERVAL_HOURS = 168.0  # 7일
_DEFAULT_MIN_VOLUME_USD = 50_000_000.0  # $50M
_DEFAULT_MIN_LISTING_AGE_DAYS = 90
_DEFAULT_MAX_ASSETS_PER_POD = 10
_DEFAULT_MAX_TOTAL_ASSETS = 20

_DEFAULT_STABLECOIN_SYMBOLS: tuple[str, ...] = (
    "USDC/USDT",
    "BUSD/USDT",
    "FDUSD/USDT",
    "TUSD/USDT",
    "DAI/USDT",
    "USDP/USDT",
    "USDD/USDT",
    "PYUSD/USDT",
)

_MAX_SCAN_HISTORY = 100


# ── SurveillanceConfig ────────────────────────────────────────────


class SurveillanceConfig(BaseModel):
    """Market Surveillance 설정.

    Attributes:
        enabled: 활성화 여부
        scan_interval_hours: 스캔 주기 (시간)
        min_24h_volume_usd: 최소 24시간 거래대금 (USD)
        min_listing_age_days: 최소 상장 기간 (일)
        exclude_stablecoins: 스테이블코인 제외 여부
        stablecoin_symbols: 스테이블코인 심볼 목록
        max_assets_per_pod: Pod당 최대 에셋 수
        max_total_assets: 전체 최대 에셋 수
        quote_currency: Quote 통화
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=False, description="활성화 여부")
    scan_interval_hours: float = Field(
        default=_DEFAULT_SCAN_INTERVAL_HOURS,
        gt=0.0,
        description="스캔 주기 (시간)",
    )
    min_24h_volume_usd: float = Field(
        default=_DEFAULT_MIN_VOLUME_USD,
        ge=0.0,
        description="최소 24시간 거래대금 (USD)",
    )
    min_listing_age_days: int = Field(
        default=_DEFAULT_MIN_LISTING_AGE_DAYS,
        ge=0,
        description="최소 상장 기간 (일)",
    )
    exclude_stablecoins: bool = Field(
        default=True,
        description="스테이블코인 제외 여부",
    )
    stablecoin_symbols: tuple[str, ...] = Field(
        default=_DEFAULT_STABLECOIN_SYMBOLS,
        description="스테이블코인 심볼 목록",
    )
    max_assets_per_pod: int = Field(
        default=_DEFAULT_MAX_ASSETS_PER_POD,
        ge=1,
        description="Pod당 최대 에셋 수",
    )
    max_total_assets: int = Field(
        default=_DEFAULT_MAX_TOTAL_ASSETS,
        ge=1,
        description="전체 최대 에셋 수",
    )
    quote_currency: str = Field(
        default="USDT",
        description="Quote 통화",
    )


# ── ScanResult ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ScanResult:
    """Surveillance 스캔 결과 (불변).

    Attributes:
        timestamp: 스캔 시점
        qualified_symbols: 선정된 전체 심볼
        added: 신규 진입 심볼
        dropped: 탈락 심볼
        retained: 유지 심볼
        scan_duration_seconds: 스캔 소요 시간
        total_scanned: 스캔된 총 심볼 수
    """

    timestamp: datetime
    qualified_symbols: tuple[str, ...]
    added: tuple[str, ...]
    dropped: tuple[str, ...]
    retained: tuple[str, ...]
    scan_duration_seconds: float
    total_scanned: int


# ── MarketSurveillanceService ─────────────────────────────────────


class MarketSurveillanceService:
    """동적 유니버스 관리 서비스.

    주기적으로 시장을 스캔하여 유동성/거래량 기준으로 에셋을 선별합니다.
    BinanceClient.fetch_top_symbols()를 재사용하여 API 호출을 최소화합니다.

    Args:
        config: SurveillanceConfig
        client: BinanceClient 인스턴스
    """

    def __init__(
        self,
        config: SurveillanceConfig,
        client: BinanceClient,
    ) -> None:
        self._config = config
        self._client = client
        self._current_universe: set[str] = set()
        self._scan_history: list[ScanResult] = []
        self._last_scan_ts: datetime | None = None

    @property
    def config(self) -> SurveillanceConfig:
        """Surveillance 설정."""
        return self._config

    @property
    def current_universe(self) -> tuple[str, ...]:
        """현재 유니버스 (정렬)."""
        return tuple(sorted(self._current_universe))

    @property
    def last_scan_ts(self) -> datetime | None:
        """마지막 스캔 시점."""
        return self._last_scan_ts

    @property
    def scan_history(self) -> list[ScanResult]:
        """스캔 이력."""
        return list(self._scan_history)

    async def scan(self) -> ScanResult:
        """시장 스캔 실행.

        1. client.fetch_top_symbols()로 후보 조회
        2. _apply_filters(): stablecoin 제외, volume 필터
        3. 이전 _current_universe와 diff → added/dropped/retained 계산
        4. _current_universe 갱신, scan_history 추가

        Returns:
            ScanResult
        """
        start_time = time.monotonic()

        # 1. 후보 조회 (최대 total_assets * 2 + 여유)
        fetch_limit = self._config.max_total_assets * 3
        all_symbols = await self._client.fetch_top_symbols(
            quote=self._config.quote_currency,
            limit=fetch_limit,
        )

        # 2. 필터 적용
        qualified = self._apply_filters(all_symbols)

        # 3. max_total_assets 상한 적용
        qualified = qualified[: self._config.max_total_assets]

        # 4. Diff 계산
        qualified_set = set(qualified)
        previous = self._current_universe

        added = tuple(s for s in qualified if s not in previous)
        dropped = tuple(s for s in previous if s not in qualified_set)
        retained = tuple(s for s in qualified if s in previous)

        # 5. Universe 갱신
        self._current_universe = qualified_set
        self._last_scan_ts = datetime.now(UTC)

        scan_duration = time.monotonic() - start_time

        result = ScanResult(
            timestamp=self._last_scan_ts,
            qualified_symbols=tuple(qualified),
            added=added,
            dropped=dropped,
            retained=retained,
            scan_duration_seconds=scan_duration,
            total_scanned=len(all_symbols),
        )

        # 6. History 추가 (trim)
        self._scan_history.append(result)
        if len(self._scan_history) > _MAX_SCAN_HISTORY:
            self._scan_history = self._scan_history[-_MAX_SCAN_HISTORY:]

        msg = (
            "Surveillance scan complete: {} qualified"
            " ({} added, {} dropped, {} retained)"
            " from {} scanned in {:.1f}s"
        )
        logger.info(
            msg,
            len(qualified),
            len(added),
            len(dropped),
            len(retained),
            len(all_symbols),
            scan_duration,
        )

        return result

    def _apply_filters(self, symbols: list[str]) -> list[str]:
        """필터 적용: stablecoin 제외.

        Note: volume 필터는 fetch_top_symbols()가 이미 quoteVolume 기준
        정렬된 결과를 반환하므로, 추가적인 API 호출 없이 상위 N개로 처리.

        Args:
            symbols: fetch_top_symbols() 결과 (volume 내림차순)

        Returns:
            필터링된 심볼 리스트
        """
        cfg = self._config
        result: list[str] = []

        stablecoin_set = set(cfg.stablecoin_symbols) if cfg.exclude_stablecoins else set()

        for symbol in symbols:
            if symbol in stablecoin_set:
                continue
            result.append(symbol)

        return result

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """서비스 상태 직렬화."""
        return {
            "current_universe": sorted(self._current_universe),
            "last_scan_ts": self._last_scan_ts.isoformat() if self._last_scan_ts else None,
            "scan_history_count": len(self._scan_history),
        }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """서비스 상태 복원."""
        universe = data.get("current_universe")
        if isinstance(universe, list):
            self._current_universe = {str(s) for s in universe}

        ts = data.get("last_scan_ts")
        if isinstance(ts, str):
            self._last_scan_ts = datetime.fromisoformat(ts)
