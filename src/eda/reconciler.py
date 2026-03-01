"""PositionReconciler — 거래소 vs PM 포지션 교차 검증.

주기적으로 거래소의 실제 포지션과 PM의 내부 상태를 비교하여
불일치를 감지합니다. Optional auto-correction 모드로 PM 상태를
거래소 기준으로 보정할 수 있습니다.

Usage:
    reconciler = PositionReconciler()
    await reconciler.initial_check(pm, futures_client, symbols)
    # periodic loop (LiveRunner에서 관리)
    await reconciler.periodic_check(pm, futures_client, symbols)
    # auto-correction 활성화 (LIVE 모드 전용)
    reconciler = PositionReconciler(auto_correct=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from src.models.types import Direction
from src.notification.reconciler_formatters import DriftDetail

if TYPE_CHECKING:
    from src.eda.portfolio_manager import EDAPortfolioManager, Position
    from src.exchange.binance_futures_client import BinanceFuturesClient


@dataclass(frozen=True)
class ExchangePositionInfo:
    """거래소 포지션 상세 정보 (size + direction + entry_price)."""

    size: float
    direction: Direction
    entry_price: float


# Position drift 임계값 (2%): 이 이상 차이나면 CRITICAL
_DRIFT_THRESHOLD = 0.02

# Balance drift 임계값
_BALANCE_DRIFT_THRESHOLD = 0.05  # 5%: CRITICAL
_BALANCE_WARN_THRESHOLD = 0.02  # 2%: WARNING

# Auto-correction 임계값: drift가 이 이상이면 PM 상태를 거래소 기준으로 보정
_AUTO_CORRECT_THRESHOLD = 0.10  # 10%


class PositionReconciler:
    """거래소 vs PM 포지션 교차 검증.

    Args:
        auto_correct: True 시 drift > 10% 인 포지션의 PM size를 거래소 기준으로 보정.
                      기본 False (safety-first: 경고만 발행).
    """

    def __init__(self, *, auto_correct: bool = False) -> None:
        self._auto_correct = auto_correct
        self._corrections_applied: int = 0
        self._last_drift_details: list[DriftDetail] = []
        self._last_balance_drift_pct: float = 0.0

    @property
    def auto_correct_enabled(self) -> bool:
        """Auto-correction 활성화 여부."""
        return self._auto_correct

    @property
    def corrections_applied(self) -> int:
        """적용된 auto-correction 횟수."""
        return self._corrections_applied

    @property
    def last_drift_details(self) -> list[DriftDetail]:
        """마지막 비교에서 수집된 drift 상세 정보."""
        return self._last_drift_details

    @property
    def last_balance_drift_pct(self) -> float:
        """마지막 잔고 검증의 drift 비율 (%)."""
        return self._last_balance_drift_pct

    async def initial_check(
        self,
        pm: EDAPortfolioManager,
        futures_client: BinanceFuturesClient,
        symbols: list[str],
    ) -> list[str]:
        """시작 시 초기 포지션 검증.

        Args:
            pm: PortfolioManager
            futures_client: Futures client
            symbols: 트레이딩 심볼 리스트

        Returns:
            불일치 심볼 리스트 (비어 있으면 정상)
        """
        logger.info("PositionReconciler: Running initial check...")
        drifts = await self._compare(pm, futures_client, symbols)

        if drifts:
            logger.critical(
                "PositionReconciler: {} position drifts detected at startup! Symbols: {}",
                len(drifts),
                drifts,
            )
        else:
            logger.info("PositionReconciler: All positions match")

        return drifts

    async def periodic_check(
        self,
        pm: EDAPortfolioManager,
        futures_client: BinanceFuturesClient,
        symbols: list[str],
    ) -> list[str]:
        """주기적 포지션 검증.

        Args:
            pm: PortfolioManager
            futures_client: Futures client
            symbols: 트레이딩 심볼 리스트

        Returns:
            불일치 심볼 리스트
        """
        try:
            drifts = await self._compare(pm, futures_client, symbols)
        except Exception:
            logger.exception("PositionReconciler: Periodic check failed")
            return []
        else:
            if drifts:
                logger.warning(
                    "PositionReconciler: {} drift(s) detected: {}",
                    len(drifts),
                    drifts,
                )
            return drifts

    async def check_balance(
        self,
        pm: EDAPortfolioManager,
        futures_client: BinanceFuturesClient,
    ) -> float | None:
        """PM equity vs 거래소 잔고 비교.

        Args:
            pm: PortfolioManager
            futures_client: Futures client

        Returns:
            거래소 equity (float) 또는 실패 시 None
        """
        try:
            balance = await futures_client.fetch_balance()
            usdt_info = balance.get("USDT", {})
            exchange_equity = float(usdt_info.get("total", 0) if isinstance(usdt_info, dict) else 0)
        except Exception:
            logger.exception("PositionReconciler: Failed to fetch balance")
            return None

        pm_equity = pm.total_equity
        if pm_equity <= 0:
            self._last_balance_drift_pct = 0.0
            return exchange_equity

        drift = abs(pm_equity - exchange_equity) / pm_equity
        self._last_balance_drift_pct = drift * 100

        if drift > _BALANCE_DRIFT_THRESHOLD:
            logger.critical(
                "PositionReconciler: BALANCE DRIFT {:.1%} — PM=${:.0f} vs Exchange=${:.0f}",
                drift,
                pm_equity,
                exchange_equity,
            )
        elif drift > _BALANCE_WARN_THRESHOLD:
            logger.warning(
                "PositionReconciler: balance drift {:.1%} — PM=${:.0f} vs Exchange=${:.0f}",
                drift,
                pm_equity,
                exchange_equity,
            )

        return exchange_equity

    @staticmethod
    async def parse_exchange_positions(
        futures_client: BinanceFuturesClient,
        symbols: list[str],
    ) -> dict[str, tuple[float, Direction]]:
        """거래소 포지션을 파싱하여 {symbol: (size, Direction)} 맵 반환.

        One-way Mode에서 CCXT가 반환하는 side(long/short)를 파싱합니다.
        LONG과 SHORT 모두 있으면 LONG 우선.

        Args:
            futures_client: BinanceFuturesClient
            symbols: 트레이딩 심볼 리스트

        Returns:
            {symbol: (size, Direction)} — size=0이면 포지션 없음
        """
        from src.exchange.binance_futures_client import (
            BinanceFuturesClient as BinanceFuturesClient_,
        )

        futures_symbols = [BinanceFuturesClient_.to_futures_symbol(s) for s in symbols]
        exchange_positions = await futures_client.fetch_positions(futures_symbols)

        result: dict[str, tuple[float, Direction]] = {}
        for pos in exchange_positions:
            sym = str(pos.get("symbol", ""))
            spot_sym = sym.split(":")[0] if ":" in sym else sym
            contracts = abs(float(pos.get("contracts", 0)))
            side = str(pos.get("side", "")).lower()

            if contracts <= 0:
                continue

            if side == "long":
                result[spot_sym] = (contracts, Direction.LONG)
            elif side == "short" and spot_sym not in result:
                result[spot_sym] = (contracts, Direction.SHORT)

        return result

    @staticmethod
    async def parse_exchange_positions_full(
        futures_client: BinanceFuturesClient,
        symbols: list[str],
    ) -> dict[str, ExchangePositionInfo]:
        """거래소 포지션을 entry_price 포함하여 파싱 (One-way Mode).

        parse_exchange_positions()와 동일 로직 + entryPrice 추출.
        LONG과 SHORT 모두 있으면 LONG 우선.

        Args:
            futures_client: BinanceFuturesClient
            symbols: 트레이딩 심볼 리스트

        Returns:
            {symbol: ExchangePositionInfo}
        """
        from src.exchange.binance_futures_client import (
            BinanceFuturesClient as BinanceFuturesClient_,
        )

        futures_symbols = [BinanceFuturesClient_.to_futures_symbol(s) for s in symbols]
        exchange_positions = await futures_client.fetch_positions(futures_symbols)

        result: dict[str, ExchangePositionInfo] = {}
        for pos in exchange_positions:
            sym = str(pos.get("symbol", ""))
            spot_sym = sym.split(":")[0] if ":" in sym else sym
            contracts = abs(float(pos.get("contracts", 0)))
            side = str(pos.get("side", "")).lower()

            if contracts <= 0:
                continue

            entry_price = float(pos.get("entryPrice", 0) or 0)
            if entry_price <= 0:
                entry_price = float(pos.get("markPrice", 0) or 0)
            if entry_price <= 0:
                logger.warning(
                    "parse_full: skipping {} — no entryPrice/markPrice",
                    spot_sym,
                )
                continue

            if side == "long":
                result[spot_sym] = ExchangePositionInfo(
                    size=contracts,
                    direction=Direction.LONG,
                    entry_price=entry_price,
                )
            elif side == "short" and spot_sym not in result:
                result[spot_sym] = ExchangePositionInfo(
                    size=contracts,
                    direction=Direction.SHORT,
                    entry_price=entry_price,
                )

        return result

    @staticmethod
    async def parse_exchange_positions_hedge(
        futures_client: BinanceFuturesClient,
        symbols: list[str],
    ) -> dict[str, dict[str, float]]:
        """거래소 포지션을 Hedge mode용으로 파싱.

        Hedge mode에서 CCXT는 동일 심볼에 LONG/SHORT 별도 row를 반환합니다.

        Args:
            futures_client: BinanceFuturesClient
            symbols: 트레이딩 심볼 리스트

        Returns:
            {symbol: {"long_size": float, "short_size": float}}
        """
        from src.exchange.binance_futures_client import (
            BinanceFuturesClient as BinanceFuturesClient_,
        )

        futures_symbols = [BinanceFuturesClient_.to_futures_symbol(s) for s in symbols]
        exchange_positions = await futures_client.fetch_positions(futures_symbols)

        result: dict[str, dict[str, float]] = {}
        for pos in exchange_positions:
            sym = str(pos.get("symbol", ""))
            spot_sym = sym.split(":")[0] if ":" in sym else sym
            contracts = abs(float(pos.get("contracts", 0)))
            side = str(pos.get("side", "")).lower()

            if contracts <= 0:
                continue

            if spot_sym not in result:
                result[spot_sym] = {"long_size": 0.0, "short_size": 0.0}

            if side == "long":
                result[spot_sym]["long_size"] = contracts
            elif side == "short":
                result[spot_sym]["short_size"] = contracts

        return result

    @staticmethod
    async def parse_exchange_positions_hedge_full(
        futures_client: BinanceFuturesClient,
        symbols: list[str],
    ) -> dict[str, dict[str, ExchangePositionInfo | None]]:
        """거래소 포지션을 Hedge mode + entry_price 포함하여 파싱.

        Args:
            futures_client: BinanceFuturesClient
            symbols: 트레이딩 심볼 리스트

        Returns:
            {symbol: {"long": ExchangePositionInfo | None, "short": ExchangePositionInfo | None}}
        """
        from src.exchange.binance_futures_client import (
            BinanceFuturesClient as BinanceFuturesClient_,
        )

        futures_symbols = [BinanceFuturesClient_.to_futures_symbol(s) for s in symbols]
        exchange_positions = await futures_client.fetch_positions(futures_symbols)

        result: dict[str, dict[str, ExchangePositionInfo | None]] = {}
        for pos in exchange_positions:
            sym = str(pos.get("symbol", ""))
            spot_sym = sym.split(":")[0] if ":" in sym else sym
            contracts = abs(float(pos.get("contracts", 0)))
            side = str(pos.get("side", "")).lower()

            if contracts <= 0:
                continue

            entry_price = float(pos.get("entryPrice", 0) or 0)
            if entry_price <= 0:
                entry_price = float(pos.get("markPrice", 0) or 0)
            if entry_price <= 0:
                logger.warning(
                    "parse_hedge_full: skipping {} {} — no entryPrice/markPrice",
                    spot_sym,
                    side,
                )
                continue

            if spot_sym not in result:
                result[spot_sym] = {"long": None, "short": None}

            direction = Direction.LONG if side == "long" else Direction.SHORT
            info = ExchangePositionInfo(
                size=contracts, direction=direction, entry_price=entry_price
            )

            if side == "long":
                result[spot_sym]["long"] = info
            elif side == "short":
                result[spot_sym]["short"] = info

        return result

    async def _compare(
        self,
        pm: EDAPortfolioManager,
        futures_client: BinanceFuturesClient,
        symbols: list[str],
    ) -> list[str]:
        """거래소 vs PM 포지션 비교.

        auto_correct 활성화 시 drift > _AUTO_CORRECT_THRESHOLD 인 포지션의
        PM size를 거래소 기준으로 보정합니다.

        Returns:
            불일치가 발견된 심볼 리스트
        """
        from src.exchange.binance_futures_client import (
            BinanceFuturesClient as BinanceFuturesClient_,
        )

        futures_symbols = [BinanceFuturesClient_.to_futures_symbol(s) for s in symbols]
        exchange_positions = await futures_client.fetch_positions(futures_symbols)

        # One-way Mode: 심볼당 long/short 사이즈 매핑
        exchange_map: dict[str, dict[str, float]] = {}
        for pos in exchange_positions:
            sym = str(pos.get("symbol", ""))
            spot_sym = sym.split(":")[0] if ":" in sym else sym
            contracts = abs(float(pos.get("contracts", 0)))
            side = str(pos.get("side", "")).lower()

            if spot_sym not in exchange_map:
                exchange_map[spot_sym] = {"long_size": 0.0, "short_size": 0.0}

            if side == "long":
                exchange_map[spot_sym]["long_size"] = contracts
            elif side == "short":
                exchange_map[spot_sym]["short_size"] = contracts

        drifts: list[str] = []
        self._last_drift_details = []

        # PM 포지션 집계: composite key (pod_id|symbol) → raw symbol 기준 합산
        pm_aggregate = self._aggregate_pm_positions(pm, symbols)

        for symbol in symbols:
            pm_info = pm_aggregate.get(symbol, {"long_size": 0.0, "short_size": 0.0})
            ex_info = exchange_map.get(symbol, {"long_size": 0.0, "short_size": 0.0})

            # PM aggregate → _check_symbol_drift 호환 형식으로 변환
            pm_size, pm_dir = self._pm_aggregate_to_size_dir(pm_info)

            drift_reasons = self._check_symbol_drift(symbol, pm_size, pm_dir, ex_info)
            if drift_reasons:
                drifts.append(symbol)

                # DriftDetail 수집
                ex_size = self._get_exchange_size(pm_dir, ex_info)
                pm_side = "FLAT" if pm_dir == Direction.NEUTRAL else pm_dir.name
                ex_side = self._detect_exchange_side(ex_info)
                max_size = max(pm_size, ex_size)
                drift_pct = (abs(pm_size - ex_size) / max_size * 100) if max_size > 0 else 0.0
                is_orphan = (pm_size > 0) != (ex_size > 0)

                # Auto-correction은 composite key 환경에서 단일 Position에만 적용 가능
                corrected = False
                if self._auto_correct:
                    pm_pos = pm.positions.get(symbol)
                    if pm_pos is not None:
                        old_size = pm_pos.size
                        self._apply_correction(pm_pos, ex_size, pm_dir, ex_info)
                        corrected = pm_pos.size != old_size

                self._last_drift_details.append(
                    DriftDetail(
                        symbol=symbol,
                        pm_size=pm_size,
                        pm_side=pm_side,
                        exchange_size=ex_size,
                        exchange_side=ex_side,
                        drift_pct=drift_pct,
                        is_orphan=is_orphan,
                        auto_corrected=corrected,
                    )
                )

        # 요약 로그 (개별 로그 대신 한 줄로 통합)
        if drifts:
            parts = [f"{d.symbol}:{d.drift_pct:.1f}%" for d in self._last_drift_details]
            logger.warning("PositionReconciler: {} drift(s) — {}", len(drifts), " | ".join(parts))

        return drifts

    @staticmethod
    def _aggregate_pm_positions(
        pm: EDAPortfolioManager,
        symbols: list[str],
    ) -> dict[str, dict[str, float]]:
        """PM 포지션을 raw symbol 기준으로 집계 (composite key 지원).

        Hedge mode에서 composite key (pod_id|symbol)로 저장된 포지션을
        raw symbol 기준 long_size/short_size로 합산합니다.

        Returns:
            {symbol: {"long_size": float, "short_size": float}}
        """
        result: dict[str, dict[str, float]] = {}
        symbol_set = set(symbols)

        for key, pos in pm.positions.items():
            if not pos.is_open:
                continue
            raw = key.split("|", 1)[1] if "|" in key else key
            if raw not in symbol_set:
                continue
            if raw not in result:
                result[raw] = {"long_size": 0.0, "short_size": 0.0}
            if pos.direction == Direction.LONG:
                result[raw]["long_size"] += pos.size
            elif pos.direction == Direction.SHORT:
                result[raw]["short_size"] += pos.size

        return result

    @staticmethod
    def _pm_aggregate_to_size_dir(
        pm_info: dict[str, float],
    ) -> tuple[float, Direction]:
        """PM aggregate → (size, direction) 변환.

        _check_symbol_drift() 호환 형식으로 변환합니다.
        LONG과 SHORT 모두 있으면 LONG 우선 (거래소 비교 시 방향별 매칭).
        """
        long_size = pm_info["long_size"]
        short_size = pm_info["short_size"]
        if long_size > 0 and short_size <= 0:
            return long_size, Direction.LONG
        if short_size > 0 and long_size <= 0:
            return short_size, Direction.SHORT
        if long_size > 0 and short_size > 0:
            return long_size, Direction.LONG
        return 0.0, Direction.NEUTRAL

    @staticmethod
    def _detect_exchange_side(ex_info: dict[str, float]) -> str:
        """거래소 포지션 방향 감지."""
        has_long = ex_info["long_size"] > 0
        has_short = ex_info["short_size"] > 0
        if has_long and has_short:
            return "HEDGE"
        if has_long:
            return "LONG"
        if has_short:
            return "SHORT"
        return "FLAT"

    def _get_exchange_size(self, pm_dir: Direction, ex_info: dict[str, float]) -> float:
        """PM 방향에 맞는 거래소 포지션 크기 반환."""
        if pm_dir == Direction.LONG:
            return ex_info["long_size"]
        if pm_dir == Direction.SHORT:
            return ex_info["short_size"]
        return ex_info["long_size"] + ex_info["short_size"]

    def _apply_correction(
        self,
        pm_pos: Position,
        ex_size: float,
        pm_dir: Direction,
        ex_info: dict[str, float],
    ) -> None:
        """PM 포지션 크기를 거래소 기준으로 보정.

        drift가 _AUTO_CORRECT_THRESHOLD 이상일 때만 보정합니다.
        """
        pm_size = pm_pos.size
        max_size = max(pm_size, ex_size)
        if max_size <= 0:
            return

        drift_ratio = abs(pm_size - ex_size) / max_size
        if drift_ratio < _AUTO_CORRECT_THRESHOLD:
            return

        old_size = pm_pos.size
        pm_pos.size = ex_size

        # 포지션이 0이 되었으면 NEUTRAL로 설정
        if ex_size <= 0:
            pm_pos.direction = Direction.NEUTRAL
            pm_pos.avg_entry_price = 0.0

        self._corrections_applied += 1
        logger.warning(
            "PositionReconciler AUTO-CORRECT: {} size {:.6f} → {:.6f} (drift {:.1%})",
            pm_pos.symbol,
            old_size,
            ex_size,
            drift_ratio,
        )

    @staticmethod
    def _check_symbol_drift(
        symbol: str,
        pm_size: float,
        pm_dir: Direction,
        ex_info: dict[str, float],
    ) -> list[str]:
        """단일 심볼의 포지션 불일치 검사.

        Returns:
            drift 사유 리스트 (비어 있으면 정상)
        """

        # PM 방향에 맞는 거래소 사이즈 선택
        if pm_dir == Direction.LONG:
            ex_size = ex_info["long_size"]
        elif pm_dir == Direction.SHORT:
            ex_size = ex_info["short_size"]
        else:
            ex_size = ex_info["long_size"] + ex_info["short_size"]

        reasons: list[str] = []

        # 수량 불일치 (상대 비교)
        if pm_size > 0 or ex_size > 0:
            max_size = max(pm_size, ex_size)
            if max_size > 0:
                drift_ratio = abs(pm_size - ex_size) / max_size
                if drift_ratio > _DRIFT_THRESHOLD:
                    dir_label = pm_dir.value if pm_dir != Direction.NEUTRAL else "neutral"
                    reasons.append(
                        f"{symbol} size drift — PM={pm_size:.6f} ({dir_label}), Exchange={ex_size:.6f} ({drift_ratio * 100:.1f}%)"
                    )

        # 한쪽만 포지션 보유
        if (pm_size > 0) != (ex_size > 0):
            who_has = "PM" if pm_size > 0 else "Exchange"
            reasons.append(
                f"{symbol} — only {who_has} has position (PM={pm_size:.6f}, Exchange={ex_size:.6f})"
            )

        # 거래소에 반대방향 포지션 존재 감지
        if pm_dir == Direction.LONG and ex_info["short_size"] > 0:
            reasons.append(
                f"{symbol} — PM is LONG but exchange has SHORT ({ex_info['short_size']:.6f})"
            )
        elif pm_dir == Direction.SHORT and ex_info["long_size"] > 0:
            reasons.append(
                f"{symbol} — PM is SHORT but exchange has LONG ({ex_info['long_size']:.6f})"
            )

        return reasons
