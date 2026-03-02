"""BacktestSurveillanceSimulator — 백테스트용 동적 유니버스 시뮬레이터.

VolumeMatrix 기반으로 라이브 MarketSurveillanceService.scan()과
동일한 ScanResult를 생성합니다.

Rules Applied:
    - #10 Python Standards: type hints, named constants
    - MarketSurveillanceService._apply_filters() 로직 재사용
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from loguru import logger

from src.orchestrator.surveillance import ScanResult, SurveillanceConfig
from src.orchestrator.volume_matrix import VolumeMatrix, rank_at

# ── Constants ─────────────────────────────────────────────────────

_MAX_SCAN_HISTORY = 100


class BacktestSurveillanceSimulator:
    """백테스트용 Surveillance 시뮬레이터.

    VolumeMatrix 기반으로 라이브 MarketSurveillanceService.scan()과
    동일한 ScanResult를 생성합니다.

    Args:
        config: SurveillanceConfig
        volume_matrix: 사전 계산된 VolumeMatrix
        seed_symbols: Pod config 초기 심볼 (백테스트 시작 시 활성)
        available_symbols: 데이터 보유 전체 심볼
    """

    def __init__(
        self,
        config: SurveillanceConfig,
        volume_matrix: VolumeMatrix,
        seed_symbols: set[str],
        available_symbols: set[str],
    ) -> None:
        self._config = config
        self._volume_matrix = volume_matrix
        self._available_symbols = available_symbols
        self._current_universe: set[str] = set(seed_symbols)
        self._last_scan_ts: datetime | None = None
        self._scan_history: list[ScanResult] = []

    @property
    def config(self) -> SurveillanceConfig:
        """Surveillance 설정."""
        return self._config

    @property
    def current_universe(self) -> tuple[str, ...]:
        """현재 유니버스 (정렬)."""
        return tuple(sorted(self._current_universe))

    @property
    def scan_history(self) -> list[ScanResult]:
        """스캔 이력."""
        return list(self._scan_history)

    def should_scan(self, current_ts: datetime) -> bool:
        """scan_interval_hours 경과 여부.

        Args:
            current_ts: 현재 시점

        Returns:
            True면 스캔 실행
        """
        if not self._config.enabled:
            return False
        if self._last_scan_ts is None:
            return True
        elapsed = current_ts - self._last_scan_ts
        interval = timedelta(hours=self._config.scan_interval_hours)
        return elapsed >= interval

    def scan_at(self, timestamp: datetime) -> ScanResult:
        """volume 기반 스캔 → ScanResult 생성.

        1. rank_at(matrix, timestamp, top_n=max_total_assets * 2)
        2. stablecoin 필터
        3. available_symbols 교집합
        4. max_total_assets 상한
        5. diff(current_universe) → added/dropped/retained

        Args:
            timestamp: 스캔 시점

        Returns:
            ScanResult
        """
        # 1. Volume 순위 (여유분 포함)
        fetch_limit = self._config.max_total_assets * 3
        ranked = rank_at(
            self._volume_matrix,
            timestamp,
            rolling_window_days=max(1, int(self._config.scan_interval_hours / 24)),
            top_n=fetch_limit,
        )

        # 2. 필터 적용 (stablecoin 제외)
        qualified = self._apply_filters(ranked)

        # 3. available_symbols 교집합 (데이터 미보유 심볼 제외)
        qualified = [s for s in qualified if s in self._available_symbols]

        # 4. max_total_assets 상한
        qualified = qualified[: self._config.max_total_assets]

        # 5. Diff 계산
        qualified_set = set(qualified)
        previous = self._current_universe

        added = tuple(s for s in qualified if s not in previous)
        dropped = tuple(s for s in previous if s not in qualified_set)
        retained = tuple(s for s in qualified if s in previous)

        # 6. Universe 갱신
        self._current_universe = qualified_set
        self._last_scan_ts = timestamp

        result = ScanResult(
            timestamp=timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=UTC),
            qualified_symbols=tuple(qualified),
            added=added,
            dropped=dropped,
            retained=retained,
            scan_duration_seconds=0.0,  # 백테스트이므로 0
            total_scanned=len(ranked),
        )

        # 7. History 추가 (trim)
        self._scan_history.append(result)
        if len(self._scan_history) > _MAX_SCAN_HISTORY:
            self._scan_history = self._scan_history[-_MAX_SCAN_HISTORY:]

        msg = "Backtest surveillance scan at {}: {} qualified ({} added, {} dropped, {} retained)"
        logger.info(
            msg,
            timestamp,
            len(qualified),
            len(added),
            len(dropped),
            len(retained),
        )

        return result

    def _apply_filters(self, symbols: list[str]) -> list[str]:
        """stablecoin 필터 적용.

        MarketSurveillanceService._apply_filters()와 동일 로직.

        Args:
            symbols: volume 순위 심볼 리스트

        Returns:
            필터링된 심볼 리스트
        """
        cfg = self._config
        if not cfg.exclude_stablecoins:
            return symbols

        stablecoin_set = set(cfg.stablecoin_symbols)
        return [s for s in symbols if s not in stablecoin_set]
